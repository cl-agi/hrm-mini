"""Frozen RandCSPBench NeuroSAT on the archived graph-coloring CNFs.

No training, restart, repair, checkpoint selection, or early stopping. The
official final-output result and the additional any-iteration result are
reported separately. Original source and data directories are read-only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from hrm.paths import repo_path, work_path
import hrm_neurosat
ROOT = work_path('sat/graph_coloring/neurosat')

import numpy as np
import torch

COMPAT = Path(hrm_neurosat.__file__).parent
CHECKPOINT = None  # Resolved from the evaluation configuration.
CHECKPOINT_SHA256 = "b0cb771fe146c9d245bedb5320030581bb9b0fa43c41aa66317aec072d7cff78"
DATA_ROOT = repo_path('data/sat/data')
N_VARIABLES = 1024
ITERATIONS = 2048
METRICS = ("full", "original", "graph")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def json_default(value):
    return str(value)


def check_source_equivalence():
    """Existing compatibility copy differs only in import statements."""
    def without_imports(path):
        tree = ast.parse(path.read_text())
        tree.body = [x for x in tree.body if not isinstance(x, (ast.Import, ast.ImportFrom))]
        return ast.dump(tree, include_attributes=False)

    manifest = json.loads(repo_path('third_party/neurosat/SOURCE.json').read_text())
    for name,row in manifest['source_files'].items():
        assert sha256(COMPAT/name) == row['vendored_sha256']
    from hrm_neurosat.scatter import scatter_sum
    # Verify duplicate destinations, unused destinations, scalar/vector values.
    index = torch.tensor([2, 0, 2, 1, 0])
    values = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    incidence = torch.nn.functional.one_hot(index, num_classes=4).T.float()
    torch.testing.assert_close(scatter_sum(values, index, dim_size=4), incidence @ values, rtol=0, atol=0)
    return {str(p):sha256(p) for p in COMPAT.glob("*.py")}


def load_model(device, checkpoint_path):
    from hrm_neurosat.gnn import GNN
    assert sha256(checkpoint_path) == CHECKPOINT_SHA256
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    opts = checkpoint["hyper_parameters"]["args"]
    assert opts.model == "neurosat" and opts.graph == "lcg" and opts.init_emb == "learned"
    assert opts.task == "assignment" and getattr(opts, "decoding", "standard") == "standard"
    opts.n_iterations = ITERATIONS
    model = GNN(opts)
    model.load_state_dict({key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()}, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, {"epoch": int(checkpoint["epoch"]), "global_step": int(checkpoint["global_step"]), "parameters": sum(p.numel() for p in model.parameters())}


def load_data(k):
    path = DATA_ROOT / f"k{k}" / "instances.npz"
    digest = sha256(path)
    assert digest == json.loads((path.parent / "data_manifest.json").read_text())["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("clauses", "edges")}
    assert data["clauses"].shape == (1000, 2304, k)
    assert data["edges"].shape == (1000, 512, 2)
    assert np.abs(data["clauses"]).max() == N_VARIABLES and (data["clauses"] != 0).all()
    return data, digest


def case_hashes(clauses, edges):
    # Exact archived formula slot order and graph edge order, little-endian i32.
    return [hashlib.sha256(c.astype("<i4").tobytes() + e.astype("<i4").tobytes()).hexdigest() for c, e in zip(clauses, edges)]


def make_graph(clauses, device, variables=N_VARIABLES, flat=False):
    """One incidence edge per distinct signed literal in each clause."""
    batch, n_clauses, width = clauses.shape
    keep = np.ones(clauses.shape, dtype=bool)
    for col in range(1, width):
        keep[:, :, col] = (clauses[:, :, col, None] != clauses[:, :, :col]).all(-1)
    degrees = keep.sum(-1)
    if flat:
        assert (degrees[:, :800].reshape(batch,200,4)==np.array([2,2,2,3])).all() and (degrees[:,800:]==2).all()
    else:assert (degrees[:, :256] == width).all() and (degrees[:, 256:] == 2).all()
    literal = 2 * (np.abs(clauses.astype(np.int64)) - 1) + (clauses < 0)
    literal += np.arange(batch)[:, None, None] * (2 * variables)
    clause = np.broadcast_to(np.arange(batch * n_clauses).reshape(batch, n_clauses, 1), clauses.shape)
    return SimpleNamespace(
        num_graphs=batch,
        l_size=torch.full((batch,), 2 * variables, device=device, dtype=torch.long),
        c_size=torch.full((batch,), n_clauses, device=device, dtype=torch.long),
        l_edge_index=torch.as_tensor(literal[keep], device=device),
        c_edge_index=torch.as_tensor(clause[keep], device=device),
    )


def trajectory(model, graph, iterations=ITERATIONS):
    """Instrument the unchanged eval recurrence with a readout after each step.

    Both messages use the previous literal/clause state, exactly as upstream
    NeuroSAT.forward_test. Readouts do not feed into the recurrence.
    """
    from hrm_neurosat.scatter import scatter_sum
    l_size, c_size = int(graph.l_size.sum()), int(graph.c_size.sum())
    variables=int(graph.l_size[0])//2
    if not bool((graph.l_size==2*variables).all()):raise ValueError("Mixed variable counts are unsupported")
    dim = model.opts.dim
    l_emb = model.l_init.repeat(l_size, 1)
    c_emb = model.c_init.repeat(c_size, 1)
    l_state = torch.zeros(l_size, dim, device=l_emb.device)
    c_state = torch.zeros(c_size, dim, device=c_emb.device)
    predictions = torch.empty((iterations, graph.num_graphs, variables), dtype=torch.bool, device=l_emb.device)
    core = model.gnn
    for iteration in range(iterations):
        l_msg_feat = core.l2c_msg_func(l_emb)
        l2c_msg = l_msg_feat[graph.l_edge_index]
        l2c_msg_aggr = scatter_sum(l2c_msg, graph.c_edge_index, dim=0, dim_size=c_size)
        c_msg_feat = core.c2l_msg_func(c_emb)
        c2l_msg = c_msg_feat[graph.c_edge_index]
        c2l_msg_aggr = scatter_sum(c2l_msg, graph.l_edge_index, dim=0, dim_size=l_size)
        positive, negative = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
        opposite = torch.cat([negative, positive], dim=1).reshape(l_size, -1)
        c_emb, c_state = core.c_update(l2c_msg_aggr, (c_emb, c_state))
        l_emb, l_state = core.l_update(torch.cat([c2l_msg_aggr, opposite], dim=1), (l_emb, l_state))
        probability = torch.sigmoid(model.l_readout(l_emb.reshape(-1, dim * 2)).reshape(graph.num_graphs, variables))
        predictions[iteration] = probability > 0.5
    return predictions, probability


def verify_trajectory(predictions, clauses, edges, k):
    """Independently check every archived candidate, using NumPy boolean CNF."""
    iterations, batch, variables = predictions.shape
    assert variables == N_VARIABLES
    full = np.zeros((iterations, batch), dtype=bool)
    original = np.zeros_like(full)
    graph = np.zeros_like(full)
    violations = np.zeros((iterations, batch), dtype=np.uint16)
    original_n, original_m = 256 * k, 256 + 512 * k
    for sample in range(batch):
        c, e = clauses[sample], edges[sample]
        for begin in range(0, iterations, 64):
            p = predictions[begin:begin + 64, sample]
            sat = (p[:, np.abs(c) - 1] == (c > 0)[None]).any(-1)
            full[begin:begin + len(p), sample] = sat.all(-1)
            original[begin:begin + len(p), sample] = sat[:, :original_m].all(-1)
            violations[begin:begin + len(p), sample] = (~sat).sum(-1)
            colors = p[:, :original_n].reshape(-1, 256, k).argmax(-1)
            graph[begin:begin + len(p), sample] = (colors[:, e[:, 0]] != colors[:, e[:, 1]]).all(-1)
    assert np.all(~full | original) and np.all(~original | graph)
    success = np.stack((full, original, graph), axis=-1)
    first = np.where(success.any(0), success.argmax(0) + 1, 0).astype(np.int32)
    witnesses = np.zeros((batch, len(METRICS), N_VARIABLES), dtype=bool)
    for sample in range(batch):
        for metric in range(len(METRICS)):
            if first[sample, metric]:
                witnesses[sample, metric] = predictions[first[sample, metric] - 1, sample]
    return {"success": success, "first_hit": first, "witnesses": witnesses, "violated_clauses": violations}


def execute_batch(model, clauses, edges, k, device):
    start = time.perf_counter()
    graph = make_graph(clauses, device)
    torch.cuda.synchronize(device)
    inference_start = time.perf_counter()
    predictions, _ = trajectory(model, graph)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    predictions = predictions.cpu().numpy()
    verified = verify_trajectory(predictions, clauses, edges, k)
    packed = np.packbits(predictions, axis=-1, bitorder="little")
    assert np.array_equal(np.unpackbits(packed, axis=-1, bitorder="little", count=N_VARIABLES).astype(bool), predictions)
    return packed, verified, {"inference_seconds": inference_seconds, "compute_verify_pack_seconds": time.perf_counter() - start}


def save_batch(path, ids, packed, verified, hashes, k, metadata, timing):
    path = Path(path)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp,
        case_ids=np.asarray(ids, dtype=np.int32), k=k, predictions_packbits=packed,
        first_hit=verified["first_hit"], witness_packbits=np.packbits(verified["witnesses"], axis=-1, bitorder="little"),
        success_packbits=np.packbits(verified["success"], axis=0, bitorder="little"),
        final_success=verified["success"][-1], violated_clauses=verified["violated_clauses"],
        run_config_sha256=metadata["run_config_sha256"], script_sha256=metadata["script_sha256"], seed_base=metadata["seed_base"],
        checkpoint_sha256=CHECKPOINT_SHA256, data_sha256=metadata["data_sha256"],
        case_sha256=np.asarray(hashes), iterations=ITERATIONS, variable_count=N_VARIABLES,
        metric_names=np.asarray(METRICS), bitorder="little", timing_json=json.dumps(timing),
    )
    temp.replace(path)


def preflight(model, device, metadata):
    out = ROOT / "preflight"
    out.mkdir(exist_ok=True)
    rows = []
    for k in (3, 4):
        data, digest = load_data(k)
        clauses, edges = data["clauses"][:1], data["edges"][:1]
        graph = make_graph(clauses, device)
        # Endpoint parity checks cover the explicit instrumented recurrence.
        checks = []
        for iterations in (1, 3, 8):
            model.opts.n_iterations = iterations
            expected = model(graph)
            saved, actual = trajectory(model, graph, iterations)
            torch.testing.assert_close(actual.reshape(-1), expected, rtol=1e-5, atol=1e-6)
            assert torch.equal(saved[-1].reshape(-1), expected > 0.5)
            checks.append({"iterations": iterations, "max_abs_error": float((actual.reshape(-1) - expected).abs().max())})
        model.opts.n_iterations = ITERATIONS
        packed, verified, timing = execute_batch(model, clauses, edges, k, device)
        meta = dict(metadata, data_sha256=digest)
        save_batch(out / f"k{k}_case0000.npz", [0], packed, verified, case_hashes(clauses, edges), k, meta, timing)
        rows.append({"k": k, "cases": 1, "iterations": ITERATIONS, "endpoint_parity": checks, **timing})
    report = dict(metadata, mode="engineering_preflight_excluded_from_formal_results", measurements=rows,
        estimated_2000_cases_inference_seconds_single_gpu=sum(x["inference_seconds"] * 1000 for x in rows),
        estimated_2000_cases_compute_verify_seconds_single_gpu=sum(x["compute_verify_pack_seconds"] * 1000 for x in rows),
        estimate_note="Two single-case timings, one per k; no formal run started. Parallel scaling and filesystem overhead are unmeasured.")
    atomic_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


def evaluate(model, device, metadata, args):
    data, digest = load_data(args.k)
    out = ROOT / "evaluation" / f"k{args.k}" / f"rank{args.rank:02d}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(metadata, data_sha256=digest, k=args.k, rank=args.rank, world_size=args.world_size, batch_size=args.batch_size,
        mode="formal_evaluation", case_sha256_encoding="SHA256(CNF little-endian int32 bytes || graph edges little-endian int32 bytes)")
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("run_config", "run_config_sha256", "seed_base", "data_sha256", "checkpoint_sha256", "script_sha256", "source_sha256", "k", "rank", "world_size", "batch_size"):
            assert old[key] == manifest[key], f"Resume manifest mismatch: {key}"
    else:
        atomic_json(manifest_path, manifest)
    assigned = np.arange(args.rank, 1000, args.world_size)
    started = time.perf_counter()
    for offset in range(0, len(assigned), args.batch_size):
        ids = assigned[offset:offset + args.batch_size]
        target = out / f"cases_{ids[0]:04d}_{ids[-1]:04d}.npz"
        if target.exists():
            with np.load(target) as archive:
                assert np.array_equal(archive["case_ids"], ids)
                assert str(archive["checkpoint_sha256"]) == CHECKPOINT_SHA256 and str(archive["data_sha256"]) == digest
                assert str(archive["run_config_sha256"]) == manifest["run_config_sha256"] and str(archive["script_sha256"]) == manifest["script_sha256"]
            continue
        c, e = data["clauses"][ids], data["edges"][ids]
        batch_start = time.perf_counter()
        packed, verified, timing = execute_batch(model, c, e, args.k, device)
        save_batch(target, ids, packed, verified, case_hashes(c, e), args.k, manifest, timing)
        status = dict(k=args.k, rank=args.rank, assigned_cases=len(assigned), completed_cases=offset + len(ids),
            last_case_ids=ids.tolist(), elapsed_seconds=time.perf_counter() - started,
            batch_wall_seconds=time.perf_counter() - batch_start, **timing)
        atomic_json(out / "progress.json", status)
        print(json.dumps(status), flush=True)
    atomic_json(out / "complete.json", dict(k=args.k, rank=args.rank, cases=len(assigned), elapsed_seconds=time.perf_counter() - started))


def main():
    global DATA_ROOT, CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "evaluate"))
    parser.add_argument("--config", type=Path, default=repo_path('data/sat/graph_coloring/neurosat.json'))
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA index; use CUDA_VISIBLE_DEVICES to allocate GPU(s).")
    parser.add_argument("--k", type=int, choices=(3, 4))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config["status"] == "ready", "Run configuration is not finalized"
    DATA_ROOT = repo_path(config["data_root"]).resolve()
    assert DATA_ROOT == repo_path('data/sat/data').resolve()
    assert config["starts"] == 1 and config["iterations"] == ITERATIONS
    CHECKPOINT = repo_path(config["checkpoint"]).resolve()
    assert config["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert isinstance(config["seed_base"], int) and 0 <= config["seed_base"] < 2**31
    assert 0 <= args.rank < args.world_size and args.batch_size > 0
    if args.mode == "evaluate":
        assert args.k is not None
    os.environ.setdefault("TMPDIR", str(ROOT / "tmp"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "cache" / "torchinductor"))
    ROOT.mkdir(parents=True,exist_ok=True)
    for path in (ROOT / "tmp", ROOT / "cache"):
        path.mkdir(exist_ok=True)
    source_hashes = check_source_equivalence()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.manual_seed(config["seed_base"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    model, checkpoint_info = load_model(device, CHECKPOINT)
    metadata = dict(run_config=str(args.config.resolve()), run_config_sha256=sha256(args.config), seed_base=config["seed_base"], checkpoint_path=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA256, checkpoint=checkpoint_info,
        data_root=str(DATA_ROOT), script_sha256=sha256(__file__), source_sha256=source_hashes,
        torch=str(torch.__version__), python=sys.version, device=torch.cuda.get_device_name(device),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), iterations=ITERATIONS, starts=1,
        initialization="learned, fixed", numeric_precision="float32; TF32 disabled; CUDA index_add may be numerically nondeterministic",
        graph_encoding="Boolean signed-literal incidence; repeated padding literals deduplicated; native four-literal clauses retained",
        assignment_decoder="sigmoid(l_readout) > 0.5", graph_decoder="first True color; color 0 if none True; no repair",
        metric_names=list(METRICS), metrics="All 2048 iterations retained and independently verified; final and any-iteration reported separately",
        inference_timing="CUDA synchronized; all recurrent iterations plus readouts; excludes loading, graph construction, CPU verification and storage",
        storage="predictions [iteration,batch,128 bytes], little-endian packbits over 1024 bool variables; success [256 bytes,batch,3 metrics], packbits over iterations; first_hit 1-based, zero means absent")
    with torch.inference_mode():
        if args.mode == "preflight":
            preflight(model, device, metadata)
        else:
            evaluate(model, device, metadata, args)


if __name__ == "__main__":
    main()
