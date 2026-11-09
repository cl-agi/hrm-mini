#!/usr/bin/env python3
"""Estimate the 16x16 causal H/L-to-future-readout Jacobian triangle.

For endpoint j, one VJP returns gradients for every recorded H_i and L_i,
i <= j. Thus shorter paths sharing an endpoint use the same backward pass.
Squared gains estimate ||J||_F^2 / d_out, matching earlier output-RMS JVPs.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch import Tensor, nn

RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "00_shared_runtime"
if str(RUNTIME_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(RUNTIME_ROOT))
from hrm.analyse.dynamics.output_sensitivity.observational import build_fixed_dataset, load_board_indices, load_model, load_run_spec
from hrm.analyse.dynamics.output_sensitivity.h_l_joint_transition_jacobian import advance_h
from hrm.analyse.dynamics.output_sensitivity.h_l_joint_readout_routing import readout_basis
from hrm.analyse.dynamics.output_sensitivity.h_l_state_rank_audit import atomic_csv, atomic_json

SCHEMA_VERSION = "h-l-output-triangle-vjp-v1"
OUTER_STEPS, H_CYCLES_PER_OUTER, L_STEPS_PER_H, OUTPUT_DIM = 16, 2, 6, 81 * 10


def rademacher_output(shape: torch.Size, device: torch.device, generator: torch.Generator) -> Tensor:
    value = torch.randint(0, 2, shape, device=device, generator=generator, dtype=torch.int8).float().mul_(2).sub_(1)
    return value - value.mean(dim=-1, keepdim=True)


def outer_transition(model: nn.Module, latent: Tensor, z_h: Tensor, z_l: Tensor) -> tuple[Tensor, Tensor]:
    for _ in range(H_CYCLES_PER_OUTER):
        old_h = z_h
        for _ in range(L_STEPS_PER_H):
            z_l = model.L_level(z_l + z_h + latent)
        z_h = advance_h(model, old_h, z_l)
    return z_h, z_l


def boundary_linearizations(model: nn.Module, x: Tensor):
    """Return clean joint boundary states and reusable VJPs for i -> i+1."""
    latent = model.embed(x).detach()
    z_h = torch.broadcast_to(model.initial_carry["z_H"], latent.shape).clone()
    z_l = torch.broadcast_to(model.initial_carry["z_L"], latent.shape).clone()
    with torch.no_grad():
        z_h, z_l = outer_transition(model, latent, z_h, z_l)
    hs, ls, vjps = [z_h.detach()], [z_l.detach()], []
    for _ in range(OUTER_STEPS - 1):
        def mapping(h: Tensor, l: Tensor) -> tuple[Tensor, Tensor]:
            return outer_transition(model, latent, h, l)
        (next_h, next_l), vjp_fn = torch.func.vjp(mapping, hs[-1], ls[-1])
        vjps.append(vjp_fn)
        hs.append(next_h.detach()); ls.append(next_l.detach())
    logits = [model.lm_head(h)[:, 1:].float() for h in hs]
    return hs, ls, logits, vjps



def reference_update_pool(model: nn.Module, dataset, device: torch.device) -> dict[str, Tensor]:
    collected: dict[str, list[Tensor]] = {"H": [], "L": []}
    with torch.no_grad():
        for batch in dataset.loader:
            x = batch[0].to(device); latent = model.embed(x)
            z_h = torch.broadcast_to(model.initial_carry["z_H"], latent.shape).clone()
            z_l = torch.broadcast_to(model.initial_carry["z_L"], latent.shape).clone()
            updates_h, updates_l = [], []
            for _ in range(OUTER_STEPS):
                old_h, old_l = z_h, z_l
                z_h, z_l = outer_transition(model, latent, z_h, z_l)
                updates_h.append((z_h - old_h).float()); updates_l.append((z_l - old_l).float())
            collected["H"].append(torch.stack(updates_h, 1)); collected["L"].append(torch.stack(updates_l, 1))
    result = {state: torch.cat(parts, 0) for state, parts in collected.items()}
    for value in result.values():
        value[:, :, 0] = 0; value.sub_(value.mean(0, keepdim=True))
    return result

def centered_readout_basis(model: nn.Module) -> tuple[Tensor, dict[str, float]]:
    weight = model.lm_head.weight.detach().to(torch.float32)
    centered = weight - weight.mean(0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    keep = singular > singular[0].clamp_min(1e-30) * max(centered.shape) * torch.finfo(centered.dtype).eps
    basis = vh[keep].T.contiguous()
    checks = {"centered_readout_rank": int(keep.sum().item()),
        "centered_readout_orthogonality_max_abs": float((basis.T @ basis - torch.eye(basis.shape[1], device=basis.device)).abs().max().item())}
    return basis, checks


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = {}
    for row in rows:
        key = (int(row["board_id"]), int(row["source_step"]), int(row["readout_step"]), str(row["source_state"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for key in sorted(grouped, key=lambda v: (v[0], v[2], v[1], v[3])):
        selected = grouped[key]; values = np.asarray([float(v["gain_sq"]) for v in selected]); mean_sq = float(values.mean())
        se_sq = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0; gain = math.sqrt(max(mean_sq, 0.0)); source = selected[0]
        item = {"board_id": key[0], "source_step": key[1], "readout_step": key[2], "horizon": key[2] - key[1],
            "source_state": key[3], "blank_count": int(source["blank_count"]), "final_exact": int(source["final_exact"]),
            "readout_exact": int(source["readout_exact"]), "probes": int(values.size), "mean_gain_sq": mean_sq,
            "mean_gain_sq_se": se_sq, "rms_gain": gain, "rms_gain_se_delta": 0.5 * se_sq / max(gain, 1e-30)}
        if "covariance_weighted_gain_sq" in source:
            item["mean_covariance_weighted_gain_sq"] = float(np.mean([float(v["covariance_weighted_gain_sq"]) for v in selected]))
            item["mean_covariance_normalized_gain_sq"] = float(np.mean([float(v["covariance_normalized_gain_sq"]) for v in selected]))
        if key[3] == "H":
            for field in ("p_r_gain_sq", "p_r_null_gain_sq", "current_gain_sq", "future_only_gain_sq"):
                component = np.asarray([float(v[field]) for v in selected]); item["mean_" + field] = float(component.mean())
            item["current_fraction"] = item["mean_current_gain_sq"] / max(mean_sq, 1e-30)
            item["p_r_fraction"] = item["mean_p_r_gain_sq"] / max(mean_sq, 1e-30)
        result.append(item)
    return result


def run_batch(model: nn.Module, x: Tensor, y: Tensor, board_ids: Sequence[int], probes: int,
              random_seed: int, p_r_q: Tensor, current_q: Tensor, reference_updates: dict[str, Tensor] | None) -> tuple[list[dict[str, Any]], dict[str, float]]:
    hs, ls, logits, transition_vjps = boundary_linearizations(model, x)
    target = y[:, 1:].long(); exact = [v.argmax(-1).eq(target).all(-1) for v in logits]
    final_exact, blanks, rows = exact[-1], x[:, 1:].eq(0).sum(-1), []
    diagonal_h_min, diagonal_l_max = math.inf, 0.0
    diagonal_future_only_max, diagonal_p_r_null_max = 0.0, 0.0
    weight = model.lm_head.weight.to(torch.float32)
    reference_variance = ({state: value[:, :, 1:].float().square().mean((0, 2, 3))
                           for state, value in reference_updates.items()} if reference_updates is not None else None)

    def append(source: int, endpoint: int, state: str, probe_index: int, gradient: Tensor) -> None:
        nonlocal diagonal_h_min, diagonal_l_max, diagonal_future_only_max, diagonal_p_r_null_max
        value = gradient[:, 1:].float(); gain_sq = value.square().sum((1, 2)) / OUTPUT_DIM
        components: dict[str, Tensor] = {}
        if reference_updates is not None:
            direction = reference_updates[state][:, source, 1:].float()
            dot = torch.einsum("btd,ktd->bk", value, direction)
            components["covariance_weighted_gain_sq"] = dot.square().mean(-1) / OUTPUT_DIM
            components["covariance_normalized_gain_sq"] = components["covariance_weighted_gain_sq"] / reference_variance[state][source].clamp_min(1e-30)
        if state == "H":
            p_r = (value @ p_r_q).square().sum((1, 2)) / OUTPUT_DIM
            current = (value @ current_q).square().sum((1, 2)) / OUTPUT_DIM
            components.update({"p_r_gain_sq": p_r, "p_r_null_gain_sq": (gain_sq - p_r).clamp_min(0),
                "current_gain_sq": current, "future_only_gain_sq": (gain_sq - current).clamp_min(0)})
        if source == endpoint and state == "H":
            diagonal_h_min = min(diagonal_h_min, float(gain_sq.min()))
            diagonal_future_only_max = max(diagonal_future_only_max, float(components["future_only_gain_sq"].max()))
            diagonal_p_r_null_max = max(diagonal_p_r_null_max, float(components["p_r_null_gain_sq"].max()))
        if source == endpoint and state == "L": diagonal_l_max = max(diagonal_l_max, float(gain_sq.max()))
        for bi, board_id in enumerate(board_ids):
            rows.append({"board_id": int(board_id), "source_step": source + 1,
                "readout_step": endpoint + 1, "horizon": endpoint - source,
                "source_state": state, "probe": probe_index, "blank_count": int(blanks[bi]),
                "final_exact": int(final_exact[bi]), "readout_exact": int(exact[endpoint][bi]),
                "gain_sq": float(gain_sq[bi]), **{name: float(component[bi]) for name, component in components.items()}})

    for endpoint in range(OUTER_STEPS):
        for probe_index in range(probes):
            probe_parts = []
            for board_id in board_ids:
                probe_generator = torch.Generator(device=x.device)
                probe_generator.manual_seed(random_seed + int(board_id) * 1_000_003 + endpoint * 10_007 + probe_index)
                probe_parts.append(rademacher_output(logits[endpoint][:1].shape, x.device, probe_generator))
            probe = torch.cat(probe_parts, dim=0)
            cot_h = torch.zeros_like(hs[endpoint], dtype=torch.float32)
            cot_h[:, 1:] = probe @ weight
            cot_l = torch.zeros_like(ls[endpoint], dtype=torch.float32)
            append(endpoint, endpoint, "H", probe_index, cot_h)
            append(endpoint, endpoint, "L", probe_index, cot_l)
            for source in range(endpoint - 1, -1, -1):
                cot_h, cot_l = transition_vjps[source]((cot_h, cot_l))
                append(source, endpoint, "H", probe_index, cot_h)
                append(source, endpoint, "L", probe_index, cot_l)
    return rows, {"minimum_diagonal_h_gain_sq": diagonal_h_min, "maximum_diagonal_l_gain_sq": diagonal_l_max,
        "maximum_diagonal_h_future_only_gain_sq": diagonal_future_only_max, "maximum_diagonal_h_p_r_null_gain_sq": diagonal_p_r_null_max}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, required=True); parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--boards", type=int, default=1)
    parser.add_argument("--board-offset", type=int, default=0); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--probes", type=int, default=4); parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--explicit-board-ids", type=int, nargs="*")
    parser.add_argument("--covariance-reference-boards", type=int, default=0); parser.add_argument("--covariance-reference-offset", type=int, default=10_000)
    parser.add_argument("--board-indices", type=Path, default=RUNTIME_ROOT / "experiments/hrm_mechanism_splits.json")
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--random-seed", type=int, default=20_260_828)
    parser.add_argument("--summary-only", action="store_true"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args(argv)
    if min(args.boards, args.batch_size, args.probes) <= 0 or min(args.board_offset, args.covariance_reference_boards, args.covariance_reference_offset) < 0:
        parser.error("boards, batch-size, probes must be positive; offset must be nonnegative")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite: parser.error(f"Output is non-empty: {output}")
    device, spec = torch.device(args.device), load_run_spec(args.seed_dir, args.epoch)
    if args.explicit_board_ids is not None:
        board_ids = tuple(args.explicit_board_ids)
        if not board_ids or len(board_ids) != len(set(board_ids)) or min(board_ids) < 0 or max(board_ids) >= 20_000:
            parser.error("explicit-board-ids must be a nonempty unique subset of [0, 20000)")
        split_hash = "explicit-test-hard-board-ids"; content_hash = "test-hard-20k-from-dataset"
    else:
        if args.full_test:
            population = tuple(range(20_000)); split_hash = hashlib.sha256(np.asarray(population, dtype=np.int64).tobytes()).hexdigest(); content_hash = "test-hard-20k-from-dataset"
        else:
            population, split_hash, content_hash = load_board_indices(args.board_indices, max_boards=2048, group="test")
        board_ids = population[args.board_offset:args.board_offset + args.boards]
        if len(board_ids) != args.boards: parser.error("Requested board range exceeds selected population")
    selected_hash = hashlib.sha256(np.asarray(board_ids, dtype=np.int64).tobytes()).hexdigest()
    dataset = build_fixed_dataset(spec, board_ids, batch_size=args.batch_size, expected_dataset_size=20_000, device=device)
    model = load_model(spec, dataset.metadata, device).float(); model.requires_grad_(False); model.embed.cast_to = torch.float32
    p_r_q, readout_checks = readout_basis(model); current_q, current_checks = centered_readout_basis(model)
    reference_updates = None; reference_ids: list[int] = []
    if args.covariance_reference_boards:
        reference_ids = list(range(args.covariance_reference_offset, args.covariance_reference_offset + args.covariance_reference_boards))
        if reference_ids[-1] >= 20_000: parser.error("Covariance reference range exceeds test_hard")
        reference_dataset = build_fixed_dataset(spec, reference_ids, batch_size=min(16, len(reference_ids)), expected_dataset_size=20_000, device=device)
        reference_updates = reference_update_pool(model, reference_dataset, device)
    rows, summary_rows, checks, consumed, probe_rows_count, started = [], [], [], 0, 0, time.time()
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for batch in dataset.loader:
            x, y = batch[0].to(device), batch[1].to(device); ids = board_ids[consumed:consumed + x.shape[0]]
            batch_rows, batch_checks = run_batch(model, x, y, ids, args.probes, args.random_seed, p_r_q, current_q, reference_updates)
            probe_rows_count += len(batch_rows)
            if args.summary_only: summary_rows.extend(summarize(batch_rows))
            else: rows.extend(batch_rows)
            checks.append(batch_checks); consumed += x.shape[0]
            print(json.dumps({"boards_done": consumed, "boards_total": len(board_ids), "elapsed_sec": round(time.time() - started, 3)}), flush=True)
    if consumed != len(board_ids): raise RuntimeError(f"Consumed {consumed}, expected {len(board_ids)}")
    if not args.summary_only: atomic_csv(output / "per_probe.csv", rows)
    atomic_csv(output / "per_board_triangle.csv", summary_rows if args.summary_only else summarize(rows))
    if reference_updates is not None:
        reference_rows = []
        for state, value in reference_updates.items():
            for source in range(OUTER_STEPS):
                viewed = value[:, source, 1:].float()
                reference_rows.append({"source_state": state, "source_step": source + 1,
                    "reference_boards": viewed.shape[0], "mean_coordinate_variance": float(viewed.square().mean()),
                    "mean_update_norm_sq": float(viewed.square().sum((1, 2)).mean())})
        atomic_csv(output / "covariance_reference.csv", reference_rows)
    atomic_json(output / "manifest.json", {"schema_version": SCHEMA_VERSION, "status": "complete",
        "checkpoint": str(spec.checkpoint), "checkpoint_sha256": spec.checkpoint_sha256, "seed": spec.seed,
        "epoch": args.epoch, "board_ids": list(board_ids), "board_ids_sha256": selected_hash,
        "board_split_hash": split_hash, "board_content_hash": content_hash, "elapsed_seconds": time.time() - started,
        "readout_checks": {**readout_checks, **current_checks},
        "checks": {"minimum_diagonal_h_gain_sq": min(v["minimum_diagonal_h_gain_sq"] for v in checks),
            "maximum_diagonal_l_gain_sq": max(v["maximum_diagonal_l_gain_sq"] for v in checks),
            "maximum_diagonal_h_future_only_gain_sq": max(v["maximum_diagonal_h_future_only_gain_sq"] for v in checks),
            "maximum_diagonal_h_p_r_null_gain_sq": max(v["maximum_diagonal_h_p_r_null_gain_sq"] for v in checks),
            "rows_expected": len(board_ids) * OUTER_STEPS * (OUTER_STEPS + 1) * args.probes,
            "rows_actual": probe_rows_count},
        "protocol": {"outer_steps": OUTER_STEPS, "h_cycles_per_outer": H_CYCLES_PER_OUTER,
            "l_steps_per_h_cycle": L_STEPS_PER_H,
            "source_nodes": "joint independent boundary coordinates after each outer step; y_i=head(H_i), so L_i to y_i is structurally zero",
            "endpoint_reuse": "one direct readout VJP plus a reverse recurrence through reusable joint-transition VJPs returns every earlier H/L source",
            "output": "cellwise logits with per-cell class mean removed; BOS excluded",
            "gain_sq": "Hutchinson VJP estimate of ||d centered_logits_j / d state_i||_F^2 / (81*10)",
            "h_components": "P_R is full row(lm_head.weight); current is row(class-centered lm_head.weight); future_only is its orthogonal complement; L is intentionally not split",
            "per_probe_saved": not args.summary_only,
            "covariance_weighting": {"reference_board_ids": reference_ids, "directions": "per-step H/L actual boundary updates, BOS-zeroed and centered across reference boards", "estimator": "mean_k (vjp dot centered_update_k)^2 / (81*10); normalized variant divides by mean coordinate variance so isotropic covariance would match raw gain"}}})
    return 0

if __name__ == "__main__": raise SystemExit(main())
