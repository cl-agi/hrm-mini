"""Frozen FP32 H4 interventions. Importing this file does not launch a run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

import hrm.analyse.sat.constraint_intervention.perturbations as perturbations
import hrm.analyse.sat.constraint_intervention.runtime as runtime
import hrm.analyse.sat.constraint_intervention.study as study


EVENT_MODULE = np.asarray(list("LLLLLLHLLLLLLH"))
EVENT_H_CYCLE = np.repeat([1, 2], 7)
EVENT_L_STEP = np.tile([1, 2, 3, 4, 5, 6, 6], 2)
NORM_FIELDS = ("delta_l2", "delta_rms", "source_l2", "source_rms",
               "destination_l2", "destination_rms")
TRACE_FIELDS = ("predictions", "violated", "legal", "valid") + tuple("event_"+k for k in NORM_FIELDS)
ARM_NAMES = ("critic_plus", "critic_minus") + tuple(f"random{k}_{sign}" for k in range(4) for sign in ("plus", "minus"))


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _clone_carry(carry):
    return {key: value.clone() for key, value in carry.items()}


def trace_continuation(model, carry, batch, *, start_outer, end_outer, n_vars):
    """Record compact native observations; never retain recurrent time states."""
    rows = {key: [] for key in TRACE_FIELDS}
    for outer in range(start_outer, end_outer+1):
        events = []

        def observe(event, _states):
            events.append(event)

        carry, logits = runtime.forward_outer(model, carry, batch["x"], observer=observe, outer=outer)
        if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Continuation logits must remain finite FP32")
        for key in ("z_H", "z_L"):
            if carry[key].dtype != torch.float32 or not bool(torch.isfinite(carry[key]).all()):
                raise RuntimeError(f"Continuation state must remain finite FP32: {key}")
        if [e["module"] for e in events] != EVENT_MODULE.tolist():
            raise RuntimeError("Unexpected native H/L event order")
        if [e["h_cycle"] for e in events] != EVENT_H_CYCLE.tolist() or [e["l_step"] for e in events] != EVENT_L_STEP.tolist():
            raise RuntimeError("Unexpected native H/L event coordinates")
        if any(e["outer"] != outer for e in events):
            raise RuntimeError("Observer outer index mismatch")
        prediction = runtime.decode_assignment(logits, n_vars)
        metrics = runtime.clause_metrics(batch["clauses"], prediction)
        rows["predictions"].append(_numpy(prediction[0]).astype(np.int8))
        for key in ("violated", "legal", "valid"):
            rows[key].append(_numpy(metrics[key][0]))
        for key in NORM_FIELDS:
            values = [np.asarray(e[key]) for e in events]
            if any(v.shape != (1,) or v.dtype != np.float64 for v in values):
                raise RuntimeError("Expected batch-one CPU FP64 event measurements")
            rows["event_"+key].append(np.asarray([v[0] for v in values], dtype=np.float64))
    result = {key: np.stack(value) for key, value in rows.items()}
    result.update(outer=np.arange(start_outer, end_outer+1, dtype=np.int64),
                  event_module=EVENT_MODULE, event_h_cycle=EVENT_H_CYCLE, event_l_step=EVENT_L_STEP)
    return result


def replay_comparison(replay, native):
    """Exact comparison at every continuation outer, including every event."""
    native_outer = native.get("outer", np.arange(1, len(native["predictions"])+1))
    lookup = {int(outer): index for index, outer in enumerate(native_outer)}
    if len(lookup) != len(native_outer) or any(int(outer) not in lookup for outer in replay["outer"]):
        raise ValueError("Native outer coordinates do not cover the continuation uniquely")
    indices = [lookup[int(outer)] for outer in replay["outer"]]
    failures = {}
    for key in TRACE_FIELDS:
        actual, expected = replay[key], native[key][indices]
        if actual.shape != expected.shape:
            failures[key] = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
        elif not np.array_equal(actual, expected):
            diff = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
            maximum = float(np.max(diff))
            failures[key] = {"mismatch_count": int(np.count_nonzero(actual != expected)),
                             "max_absolute_error": maximum if np.isfinite(maximum) else "nonfinite"}
    for key, expected in (("event_module", EVENT_MODULE), ("event_h_cycle", EVENT_H_CYCLE), ("event_l_step", EVENT_L_STEP)):
        if not np.array_equal(native[key], expected):
            failures[key] = {"reason": "native event coordinates differ"}
    return {"status": "PASSED" if not failures else "FAILED", "exact": not failures,
            "first_outer": int(replay["outer"][0]), "last_outer": int(replay["outer"][-1]), "failures": failures}


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _trace_schema(data, *, case_id, module, alpha, arm, input_seed, n_vars, end_outer):
    expected_scalar = {"case_id": case_id, "module": module, "alpha": alpha, "arm": arm,
                       "input_seed": input_seed, "forward_dtype": "float32", "geometry_passed": True,
                       "sign": 1 if arm.endswith("plus") else -1}
    for key, expected in expected_scalar.items():
        if key not in data or data[key].shape != () or data[key].item() != expected:
            raise ValueError(f"Invalid completed artifact scalar: {key}")
    count = end_outer-2
    if not np.array_equal(data["outer"], np.arange(3, end_outer+1)):
        raise ValueError("Invalid completed artifact outer coordinates")
    for key, expected in (("event_module", EVENT_MODULE), ("event_h_cycle", EVENT_H_CYCLE), ("event_l_step", EVENT_L_STEP)):
        if not np.array_equal(data[key], expected):
            raise ValueError(f"Invalid completed event coordinates: {key}")
    for key in TRACE_FIELDS:
        expected_shape = (count, n_vars) if key == "predictions" else ((count, 14) if key.startswith("event_") else (count,))
        if data[key].shape != expected_shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid completed trace field: {key}")
    if any(data[key].dtype != np.float64 for key in TRACE_FIELDS if key.startswith("event_")):
        raise ValueError("Completed event measurements must be float64")
    if not np.isin(data["predictions"], [0, 1, 2, 3]).all():
        raise ValueError("Completed predictions contain invalid vocabulary codes")
    if not np.array_equal(data["valid"], data["legal"] & (data["violated"] == 0)):
        raise ValueError("Completed SAT success fields disagree")


def validated_complete(path, expected, provenance):
    """Skip only a validated COMPLETE pair; preserve uncommitted NPZ on resume."""
    path = Path(path)
    audit_path = path.with_suffix(".json")
    if not path.exists() and not audit_path.exists():
        return False
    if path.is_file() and not audit_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        preserved = path.with_name(path.name+"."+stamp+".incomplete")
        if preserved.exists():
            raise RuntimeError(f"Incomplete evidence destination already exists: {preserved}")
        path.rename(preserved)
        print(json.dumps({"status": "INCOMPLETE_PRESERVED", "path": str(path), "preserved": str(preserved)}), flush=True)
        return False
    if not path.is_file() or not audit_path.is_file():
        raise RuntimeError(f"Partial intervention result requires inspection: {path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "COMPLETE" or audit.get("provenance") != provenance:
        raise RuntimeError(f"Invalid completion status/provenance: {audit_path}")
    if audit.get("sha256") != study.sha256(path):
        raise RuntimeError(f"Intervention artifact hash mismatch: {path}")
    geometry = audit.get("geometry_audit", {})
    geometry_path = path.parent/"geometry.json"
    if geometry.get("path") != str(geometry_path) or geometry.get("sha256") != study.sha256(geometry_path):
        raise RuntimeError(f"Intervention geometry audit hash/path mismatch: {path}")
    geometry_report = json.loads(geometry_path.read_text())
    raw_path = path.parent/"geometry.npz"
    if (geometry_report.get("status") != "PASSED" or geometry_report.get("provenance") != provenance
            or geometry_report.get("raw_sha256") != study.sha256(raw_path)
            or geometry.get("raw_npz_sha256") != study.sha256(raw_path)):
        raise RuntimeError(f"Intervention geometry gate/provenance mismatch: {path}")
    _trace_schema(_read_npz(path), **expected)
    return True


def geometry_arrays(generated):
    """Lossless raw audit arrays, no full hidden tensors for successful arms."""
    result = {}
    for name, arm in generated["arms"].items():
        for check_name in ("reference_check", "native_check"):
            check = arm[check_name]
            for key, value in check.items():
                if key == "failures":
                    for failure, mask in value.items():
                        result[f"{name}__{check_name}__failed_{failure}"] = _numpy(mask)
                else:
                    result[f"{name}__{check_name}__{key}"] = _numpy(value)
        for key, value in arm.items():
            if key.startswith("cast_"):
                result[f"{name}__{key}"] = _numpy(value)
    for name, check in generated["matching"].items():
        for key, value in check.items():
            result[f"matching__{name}__{key}"] = _numpy(value)
    return result


def _summary(check):
    result = {"passed": check["passed"], "failed_counts": {key: int(_numpy(mask).sum()) for key, mask in check["failures"].items()}}
    for key in ("token_norm_before", "token_norm_after", "token_norm_relative_error", "token_relative_displacement", "token_probe_delta", "token_probe_delta_normalized", "probe_before", "probe_after", "probe_delta", "probe_delta_normalized"):
        value = _numpy(check[key])
        result[key] = {"min": float(value.min()), "max": float(value.max()), "max_abs": float(np.abs(value).max())}
        result[key] = {k: v if np.isfinite(v) else "nonfinite" for k, v in result[key].items()}
    return result


def _failure(group, filename, report, arrays):
    raw_path = group/(filename+".npz")
    json_path = group/(filename+".json")
    if raw_path.exists() or json_path.exists():
        raise RuntimeError(f"Prior failure artifact retained: {json_path}")
    study.write_npz(raw_path, **arrays)
    study.write_json(json_path, {**report, "status": "FAILED", "raw_npz": raw_path.name, "raw_sha256": study.sha256(raw_path)})


def audit_group(group, generated, *, provenance, source_state, weight, covariance, alpha, seed):
    """Persist all ten audits before any arm of this group is continued."""
    group = Path(group)
    arrays = geometry_arrays(generated)
    report = {"status": "PASSED" if generated["passed"] else "FAILED", "alpha": alpha,
              "seed_base": seed, "control_seeds": [seed+k for k in range(4)],
              "native_dtype": generated["native_dtype"], "tolerances": generated["tolerances"],
              "provenance": provenance,
              "arms": {name: {key: _summary(arm[key]) for key in ("reference_check", "native_check")} for name, arm in generated["arms"].items()},
              "matching_passed": {name: value["passed"] for name, value in generated["matching"].items()}}
    if not generated["passed"]:
        arrays.update(source_state=_numpy(source_state), weight=_numpy(weight), covariance=_numpy(covariance))
        _failure(group, "geometry_failed", report, arrays)
        raise RuntimeError(f"Geometry gate FAILED; all arms retained in {group/'geometry_failed.json'}")
    raw_path, json_path = group/"geometry.npz", group/"geometry.json"
    if raw_path.exists() or json_path.exists():
        if not raw_path.is_file() or not json_path.is_file():
            raise RuntimeError(f"Partial geometry audit requires inspection: {group}")
        previous = json.loads(json_path.read_text())
        stored = _read_npz(raw_path)
        same = (previous.get("status") == "PASSED" and previous.get("provenance") == provenance
                and previous.get("raw_sha256") == study.sha256(raw_path)
                and stored.keys() == arrays.keys()
                and all(np.array_equal(stored[key], value) for key, value in arrays.items()))
        if not same:
            _failure(group, "geometry_recheck_failed", report, arrays)
            raise RuntimeError(f"Previously saved geometry audit changed: {group}")
    else:
        study.write_npz(raw_path, **arrays)
        study.write_json(json_path, {**report, "raw_npz": raw_path.name, "raw_sha256": study.sha256(raw_path)})
    return {"path": str(json_path), "sha256": study.sha256(json_path), "raw_npz_sha256": study.sha256(raw_path)}


def _arm_geometry(arm):
    result = {}
    for check_name, prefix in (("native_check", "geometry_"), ("reference_check", "reference_geometry_")):
        for key, value in arm[check_name].items():
            if key not in ("passed", "failures"):
                result[prefix+key] = _numpy(value).squeeze(axis=0)
    for key, value in arm.items():
        if key.startswith("cast_"):
            result[key] = _numpy(value).squeeze(axis=0)
    return result


def _load_probe(module, device, feature_dim):
    path = study.RUN/"probes"/(module+".npz")
    data = _read_npz(path)
    for key, shape in (("weight", (feature_dim,)), ("covariance", (feature_dim, feature_dim)), ("mean", (feature_dim,))):
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid probe {module}/{key}")
    intercept = float(np.asarray(data["intercept"]).item())
    if not np.isfinite(intercept):
        raise ValueError("Nonfinite probe intercept")
    return dict(weight=torch.as_tensor(data["weight"], device=device, dtype=torch.float64),
                covariance=torch.as_tensor(data["covariance"], device=device, dtype=torch.float64),
                intercept=intercept, sha256=study.sha256(path), path=str(path))


def _native_case(case_id, batch, protocol, device, manifest_hash):
    directory = study.RUN/"native"/"test"
    stem = f"case_{case_id:04d}"
    npz_path, h4_path = directory/(stem+".npz"), directory/(stem+"_h4.pt")
    complete = json.loads((directory/(stem+".complete.json")).read_text())
    seed = protocol["trajectory"]["input_seed_base"]+case_id
    for key, expected in (("status", "COMPLETE"), ("case_id", case_id), ("split", "test"),
                          ("input_seed", seed), ("forward_dtype", "float32"), ("run_manifest_sha256", manifest_hash)):
        if complete.get(key) != expected:
            raise ValueError(f"Native completion mismatch: {stem}/{key}")
    hashes = {path.name: study.sha256(path) for path in (npz_path, h4_path)}
    if complete.get("sha256") != hashes:
        raise ValueError(f"Native/H4 artifact hash mismatch: {stem}")
    saved = torch.load(h4_path, map_location="cpu", weights_only=True)
    for key, expected in (("case_id", case_id), ("outer", 2), ("input_seed", seed), ("forward_dtype", "float32"),
                          ("checkpoint_sha256", runtime.verify_checkpoint()['sha256']), ("run_manifest_sha256", manifest_hash)):
        if saved.get(key) != expected:
            raise ValueError(f"Saved H4 metadata mismatch: {key}")
    carry = saved["carry"]
    if set(carry) != {"z_H", "z_L", "mixture_weights"}:
        raise ValueError("Unexpected H4 carry keys")
    for key, value in carry.items():
        if value.dtype != torch.float32 or value.ndim != 3 or value.shape[0] != 1 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"H4 must contain finite expanded FP32 tensors: {key}")
    for key in ("z_H", "z_L"):
        if carry[key].shape != (1, batch["x"].shape[1], protocol["trajectory"]["feature_dim"]):
            raise ValueError(f"Saved H4 state shape mismatch: {key}")
    if not torch.equal(carry["mixture_weights"], batch["mix"].cpu()):
        raise ValueError("Regenerated deterministic mixture differs from the saved H4 mixture")
    native = _read_npz(npz_path)
    for key, expected in (("case_id", case_id), ("input_seed", seed), ("split", "test"),
                          ("forward_dtype", "float32"), ("checkpoint_sha256", runtime.verify_checkpoint()['sha256'])):
        if np.asarray(native[key]).shape != () or np.asarray(native[key]).item() != expected:
            raise ValueError(f"Native array metadata mismatch: {key}")
    if not np.array_equal(native["outer"], np.arange(1, protocol["trajectory"]["outer_cycles"]+1)) or native["predictions"].shape != (protocol["trajectory"]["outer_cycles"], protocol["data"]["n_vars"]):
        raise ValueError("Native arrays must contain the full 64-outer trajectory")
    return ({key: value.to(device).clone() for key, value in carry.items()}, native,
            {"native_npz_sha256": hashes[npz_path.name], "h4_sha256": hashes[h4_path.name]})


@torch.inference_mode()
def run_worker(device, worker_index, workers):
    protocol = study.assert_frozen()
    if workers < 1 or not 0 <= worker_index < workers:
        raise ValueError("Require workers >= 1 and 0 <= worker_index < workers")
    if protocol["execution"]["forward_dtype"] != "float32":
        raise RuntimeError("This intervention runner requires the frozen full-FP32 protocol")
    if protocol["intervention"]["fork_after_outer"] != 2 or protocol["intervention"]["fork_after_h_update"] != 4:
        raise RuntimeError("This runner implements the confirmed H4 fork only")
    if protocol["intervention"]["random_controls_per_absolute_dose"] != 8:
        raise RuntimeError("This runner requires exactly eight matched controls")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    if torch.device(device).type == "cuda":
        if str(device).startswith("cuda"):torch.cuda.set_device(device)
    tolerances = perturbations.Tolerances(**protocol["precision"]["formal_tolerances"])
    manifest_hash = study.sha256(study.RUN/"run_manifest.json")
    n_vars = protocol["data"]["n_vars"]
    data_path = study.ROOT/protocol["data"]["root"]/f"test_n{n_vars}.npz"
    arrays = runtime.load_arrays(data_path)
    if len(arrays["clauses"]) != protocol["data"]["test"] or protocol["data"]["intervention_count"] != len(arrays["clauses"]):
        raise ValueError("The full fixed test cohort must be retained")
    model = runtime.load_model(device, forward_dtype="float32")
    modules = protocol["intervention"]["modules"]
    if modules != ["H", "L"]:
        raise ValueError("Expected both separately fitted H and L critics")
    probes = {module: _load_probe(module, device, protocol["trajectory"]["feature_dim"]) for module in modules}
    doses = sorted(set(abs(float(x)) for x in protocol["intervention"]["signed_relative_displacements"]))
    if doses != [.002, .01, .03]:
        raise ValueError("Frozen intervention doses must be 0.002, 0.01, and 0.03")
    seed = int(protocol["intervention"]["four_direction_seed_base"])
    if protocol["intervention"]["random_control_seeds"] != [seed+k for k in range(4)]:
        raise ValueError("Control seed list disagrees with the four-direction base seed")
    end_outer = protocol["intervention"]["continue_through_outer"]
    if end_outer != protocol["trajectory"]["outer_cycles"] or end_outer<3:
        raise ValueError("Continuation must match the frozen trajectory budget and extend past the fork")
    for case_id in range(len(arrays["clauses"])):
        if case_id % workers != worker_index:
            continue
        study.assert_frozen()
        for probe in probes.values():
            if study.sha256(probe["path"]) != probe["sha256"]:
                raise RuntimeError("A fitted probe changed during intervention execution")
        input_seed = protocol["trajectory"]["input_seed_base"]+case_id
        batch = runtime.inference_batch(arrays, case_id, device, seed=input_seed, n_vars=n_vars)
        runtime.bind_graph_mask(model, batch["mask"])
        carry, native, native_hashes = _native_case(case_id, batch, protocol, device, manifest_hash)
        case_dir = study.RUN/"interventions"/"test"/f"case_{case_id:04d}"
        replay = trace_continuation(model, _clone_carry(carry), batch, start_outer=3, end_outer=end_outer, n_vars=n_vars)
        comparison = replay_comparison(replay, native)
        if not comparison["exact"]:
            _failure(case_dir, "native_replay_failed", {**comparison, **native_hashes, "case_id": case_id}, replay)
            raise RuntimeError(f"Native replay mismatch for case {case_id}; inspect saved failure")
        replay_path = case_dir/"native_replay.json"
        if not replay_path.exists():
            study.write_json(replay_path, {**comparison, **native_hashes, "case_id": case_id,
                                          "run_manifest_sha256": manifest_hash})
        for module in modules:
            probe = probes[module]
            source = carry["z_"+module]
            provenance = {**native_hashes, "run_manifest_sha256": manifest_hash, "probe_sha256": probe["sha256"]}
            for alpha in doses:
                group = case_dir/module/f"alpha_{alpha:g}"
                expected = dict(case_id=case_id, module=module, alpha=alpha, input_seed=input_seed, n_vars=n_vars, end_outer=end_outer)
                completed = {name: validated_complete(group/(name+".npz"), {**expected, "arm": name}, provenance) for name in ARM_NAMES}
                if all(completed.values()):
                    continue
                try:
                    generated = perturbations.generate_arms(source, probe["weight"], probe["covariance"], alpha, seed, tolerances=tolerances)
                except ValueError as error:
                    _failure(group, "geometry_unreachable", {"error": str(error), "provenance": provenance, "alpha": alpha, "seed_base": seed},
                             {"source_state": _numpy(source), "weight": _numpy(probe["weight"]), "covariance": _numpy(probe["covariance"])})
                    raise
                audit = audit_group(group, generated, provenance=provenance, source_state=source,
                                    weight=probe["weight"], covariance=probe["covariance"], alpha=alpha, seed=seed)
                for name, arm in generated["arms"].items():
                    if completed[name]:
                        continue
                    altered = _clone_carry(carry)
                    altered["z_"+module] = arm["state"]
                    trace = trace_continuation(model, altered, batch, start_outer=3, end_outer=end_outer, n_vars=n_vars)
                    before = float(arm["native_check"]["probe_before"][0])+probe["intercept"]
                    after = float(arm["native_check"]["probe_after"][0])+probe["intercept"]
                    payload = {**trace, **_arm_geometry(arm), "case_id": case_id, "input_seed": input_seed,
                               "module": module, "alpha": alpha, "arm": name, "sign": arm["sign"],
                               "forward_dtype": "float32", "geometry_passed": True,
                               "initial_probe_before": before, "initial_probe_after": after,
                               "initial_probe_change": float(arm["native_check"]["probe_delta"][0])}
                    payload = {key: np.asarray(value) for key, value in payload.items()}
                    _trace_schema(payload, **expected, arm=name)
                    path = group/(name+".npz")
                    study.write_npz(path, **payload)
                    study.write_json(path.with_suffix(".json"), {"status": "COMPLETE", "sha256": study.sha256(path),
                        "provenance": provenance, "geometry_audit": audit, "case_id": case_id, "input_seed": input_seed,
                        "module": module, "alpha": alpha, "arm": name, "sign": arm["sign"], "forward_dtype": "float32",
                        "initial_probe_before": before, "initial_probe_after": after,
                        "initial_probe_change": payload["initial_probe_change"].item(),
                        "geometry": _summary(arm["native_check"]), "reference_geometry": _summary(arm["reference_check"])})
                del generated
                print(json.dumps({"case_id": case_id, "module": module, "alpha": alpha, "status": "COMPLETE"}), flush=True)
        print(json.dumps({"case_id": case_id, "worker_index": worker_index, "status": "COMPLETE"}), flush=True)
    study.write_json(study.RUN/"workers"/f"intervene_{worker_index}.complete.json",
                     {"status": "COMPLETE", "worker_index": worker_index, "workers": workers,
                      "run_manifest_sha256": manifest_hash})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    run_worker(args.device, args.worker_index, args.workers)


if __name__ == "__main__":
    main()
