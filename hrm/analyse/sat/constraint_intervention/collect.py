"""Collect fixed native trajectories from the local frozen SAT checkpoint."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import runtime, study


EVENT_FIELDS = ("delta_l2", "delta_rms", "source_l2", "source_rms", "destination_l2", "destination_rms")
EVENT_MODULE = np.array(["L"] * 6 + ["H"] + ["L"] * 6 + ["H"])
EVENT_H_CYCLE = np.repeat([1, 2], 7)
EVENT_L_STEP = np.tile([1, 2, 3, 4, 5, 6, 6], 2)


def validate_events(events, outer):
    if len(events) != 14 or [e["module"] for e in events] != EVENT_MODULE.tolist():
        raise RuntimeError("Native event order differs from 12 L and 2 H updates")
    for index, event in enumerate(events):
        if (event["outer"] != outer or event["h_cycle"] != EVENT_H_CYCLE[index]
                or event["l_step"] != EVENT_L_STEP[index]):
            raise RuntimeError("Native event indices do not match the fixed protocol")
        if any(np.asarray(event[k]).shape != (1,) or not np.isfinite(event[k]).all()
               for k in EVENT_FIELDS):
            raise RuntimeError("Nonfinite or unexpected per-case observation")


def completed_case(folder, split, case_id, run_hash):
    manifest_path = folder / f"case_{case_id:04d}.complete.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    expected = {f"case_{case_id:04d}.npz"}
    if split == "test":
        expected.add(f"case_{case_id:04d}_h4.pt")
    if (manifest.get("status") != "COMPLETE" or manifest.get("split") != split
            or manifest.get("case_id") != case_id or manifest.get("input_seed") != study.protocol()["trajectory"]["input_seed_base"] + case_id
            or manifest.get("forward_dtype") != "float32"
            or manifest.get("run_manifest_sha256") != run_hash
            or set(manifest.get("sha256", {})) != expected):
        raise RuntimeError(f"Invalid native completion manifest: {manifest_path}")
    for name, digest in manifest["sha256"].items():
        if study.sha256(folder / name) != digest:
            raise RuntimeError(f"Native artifact hash mismatch: {folder / name}")
    return True


@torch.no_grad()
def collect_case(model, arrays, split, case_id, device, protocol, run_hash):
    folder = study.RUN / "native" / split
    if completed_case(folder, split, case_id, run_hash):
        return {"split": split, "case_id": case_id, "resumed": True}
    started = time.perf_counter()
    seed = protocol["trajectory"]["input_seed_base"] + case_id
    batch = runtime.inference_batch(arrays, case_id, device, seed=seed,
                                    n_vars=protocol["data"]["n_vars"])
    runtime.bind_graph_mask(model, batch["mask"])
    carry = runtime.initial_carry(model, batch)
    steps = protocol["trajectory"]["outer_cycles"]
    features = {"H": [], "L": []}
    predictions, violated, legal, valid = [], [], [], []
    observations = {key: [] for key in EVENT_FIELDS}
    h4_hash = None
    for outer in range(1, steps + 1):
        events = []
        def observe(event, _states):
            events.append(event)
        carry, logits = runtime.forward_outer(model, carry, batch["x"], observer=observe, outer=outer)
        validate_events(events, outer)
        for key in ("z_H", "z_L"):
            if carry[key].dtype != torch.float32 or not bool(torch.isfinite(carry[key]).all()):
                raise RuntimeError(f"Native {key} dtype/finite check failed at {split}/{case_id}/{outer}")
        if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Native logits dtype/finite check failed")
        for name, feature in runtime.pooled_features(carry).items():
            features[name].append(feature[0].cpu().numpy())
        prediction = runtime.decode_assignment(logits, protocol["data"]["n_vars"])
        metrics = runtime.clause_metrics(batch["clauses"], prediction)
        predictions.append(prediction[0].cpu().numpy().astype(np.int8))
        violated.append(int(metrics["violated"][0]))
        legal.append(bool(metrics["legal"][0]))
        valid.append(bool(metrics["valid"][0]))
        for key in EVENT_FIELDS:
            observations[key].append([float(event[key][0]) for event in events])
        if split == "test" and outer == protocol["intervention"]["fork_after_outer"]:
            h4_path = folder / f"case_{case_id:04d}_h4.pt"
            study.write_torch(h4_path, {
                "carry": {key: tensor.detach().cpu().clone() for key, tensor in carry.items()},
                "case_id": case_id, "split": split, "outer": outer, "h_update": 4,
                "input_seed": seed, "forward_dtype": "float32",
                "checkpoint_sha256": runtime.verify_checkpoint()['sha256'],
                "run_manifest_sha256": run_hash,
            })
            h4_hash = study.sha256(h4_path)
    torch.cuda.synchronize(device)
    data = dict(
        case_id=np.int64(case_id), split=np.array(split), input_seed=np.int64(seed),
        outer=np.arange(1, steps + 1), forward_dtype=np.array("float32"),
        checkpoint_sha256=np.array(runtime.verify_checkpoint()['sha256']),
        event_module=EVENT_MODULE, event_h_cycle=EVENT_H_CYCLE, event_l_step=EVENT_L_STEP,
        predictions=np.stack(predictions), violated=np.array(violated),
        legal=np.array(legal), valid=np.array(valid),
        **{name: np.stack(values) for name, values in features.items()},
        **{f"event_{key}": np.array(values, dtype=np.float64) for key, values in observations.items()},
    )
    path = folder / f"case_{case_id:04d}.npz"
    study.write_npz(path, **data)
    hashes = {path.name: study.sha256(path)}
    if split == "test":
        if h4_hash is None:
            raise RuntimeError("Missing H4 fork state")
        hashes[f"case_{case_id:04d}_h4.pt"] = h4_hash
    metadata = dict(
        status="COMPLETE", split=split, case_id=case_id, input_seed=seed,
        forward_dtype="float32", sha256=hashes, run_manifest_sha256=run_hash,
        elapsed_seconds=time.perf_counter() - started, utc=datetime.now(timezone.utc).isoformat(),
    )
    study.write_json(folder / f"case_{case_id:04d}.complete.json", metadata)
    print(json.dumps(metadata), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.workers:
        parser.error("worker-index must be within workers")
    p = study.assert_frozen()
    if p["execution"]["forward_dtype"] != "float32":
        raise RuntimeError("This experiment is the approved FP32 mechanism analysis")
    torch.set_num_threads(2)
    if args.device.startswith("cuda"):torch.cuda.set_device(args.device)
    torch.manual_seed(p["statistics"]["bootstrap_seed"])
    model = runtime.load_model(args.device, forward_dtype="float32")
    run_hash = study.sha256(study.RUN / "run_manifest.json")
    for split in ("train", "valid", "test"):
        arrays = runtime.load_arrays(study.ROOT / p["data"]["root"] / f"{split}_n1024.npz")
        if len(arrays["clauses"]) != p["data"][split]:
            raise RuntimeError(f"Unexpected {split} puzzle count")
        for case_id in range(args.worker_index, p["data"][split], args.workers):
            collect_case(model, arrays, split, case_id, args.device, p, run_hash)
    study.write_json(study.RUN / "workers" / f"collect_{args.worker_index}.complete.json", {
        "status": "COMPLETE", "worker_index": args.worker_index, "workers": args.workers,
        "run_manifest_sha256": run_hash,
    })


if __name__ == "__main__":
    main()
