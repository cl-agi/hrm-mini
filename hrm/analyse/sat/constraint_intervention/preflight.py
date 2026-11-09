"""Explicit engineering preflight only; never collects or launches formal results."""

from __future__ import annotations
from hrm.paths import analysis_directory

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from . import runtime
from hrm.paths import repo_path, work_path


from . import study
ROOT = study.ROOT
CHECKS = study.CHECKS
SEED = 20261018
# Historical Sudoku engineering thresholds, not validated SAT tolerances.
REFERENCE_THRESHOLDS = {
    "relative_token_norm_error": 2e-6,
    "relative_chord_error": 2e-7,
    "control_probe_absolute_error": 2e-4,
    "control_probe_normalized_error": 2e-4,
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_gpu0_idle() -> dict:
    """Check the physical device before this process creates its CUDA context."""
    result = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    uuid, name, used, total, utilization = [part.strip() for part in result.split(",")]
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    active = [line for line in processes.splitlines() if line.startswith(uuid + ",")]
    if active or int(utilization) != 0 or int(used) > 128:
        raise RuntimeError(f"GPU0 is not idle: {result}; processes={active}")
    return {"physical_index": 0, "uuid": uuid, "name": name, "used_mib": int(used), "total_mib": int(total), "utilization_percent": int(utilization), "processes": active}


def _copied_common():
    """Import the immutable local common.py, suppressing all bytecode writes."""
    from hrm.tasks.sat_flow import SATTask
    return SATTask(runtime.load_config(),study.DIRECTORY)


def _equal_tensors(left: dict, right: dict, keys) -> dict:
    return {key: bool(torch.equal(left[key], right[key])) for key in keys}


def _rng_state() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()


def _rng_unchanged(before) -> bool:
    after = _rng_state()
    return all(torch.equal(a, b) for a, b in zip(before, after))


def _check_directory(forward_dtype: str) -> Path:
    if forward_dtype not in ("float32", "bfloat16"):
        raise ValueError("Unsupported forward_dtype")
    # Preserve the original BF16 reports and snapshots in checks/ forever.
    return CHECKS / ("fp32" if forward_dtype == "float32" else "bf16")


def _floating_tensors(value):
    if torch.is_tensor(value):
        if value.is_floating_point():
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floating_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _floating_tensors(item)


def _checkpoint_value_audit(model) -> dict:
    checkpoint = torch.load(runtime.verify_checkpoint()["path"], map_location="cpu", weights_only=True)["model"]
    equal = {name: bool(torch.equal(value.detach().cpu().double(), checkpoint[name].double())) for name, value in model.state_dict().items()}
    return {"all_checkpoint_values_exactly_preserved": all(equal.values()), "tensor_count": len(equal), "mismatched_tensors": [name for name, match in equal.items() if not match]}


def run_parity(forward_dtype="float32") -> dict:
    """One fixed test case, four outer steps, raw checkpoint, explicit precision."""
    checks = _check_directory(forward_dtype)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=0 to restrict this preflight to physical GPU0")
    idle = check_gpu0_idle()
    started = time.perf_counter()
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    model = runtime.load_model("cuda:0", forward_dtype=forward_dtype)
    value_audit = _checkpoint_value_audit(model)
    dtype_audit = {}

    def audit_dtype(name):
        def capture(_module, args, output):
            entry = dtype_audit.setdefault(name, {"input_dtypes": set(), "output_dtypes": set(), "calls": 0})
            entry["input_dtypes"].update(str(value.dtype) for value in _floating_tensors(args))
            entry["output_dtypes"].update(str(value.dtype) for value in _floating_tensors(output))
            entry["calls"] += 1
        return capture

    dtype_handles = [module.register_forward_hook(audit_dtype(name or "model")) for name, module in model.named_modules()]
    arrays = runtime.load_arrays(Path(study.protocol()['data']['root'])/'test_n1024.npz')
    shape = arrays["answers"].shape
    zero_arrays = {"clauses": arrays["clauses"], "answers": np.zeros(shape, dtype=np.int64)}
    one_arrays = {"clauses": arrays["clauses"], "answers": np.ones(shape, dtype=np.int64)}
    batch = runtime.inference_batch(zero_arrays, 0, "cuda:0")
    isolated_batch = runtime.inference_batch(one_arrays, 0, "cuda:0")
    common = _copied_common()
    copied_batch = common.batch(zero_arrays, [0], 801000, False)
    copied_isolated = common.batch(one_arrays, [0], 801000, False)
    batch_keys = ("x", "mask", "mix", "clauses", "p")
    batch_parity = _equal_tensors(batch, copied_batch, batch_keys)
    target_isolation = _equal_tensors(batch, isolated_batch, batch_keys)
    copied_target_isolation = _equal_tensors(copied_batch, copied_isolated, batch_keys)
    del arrays, zero_arrays, one_arrays, isolated_batch, copied_batch, copied_isolated
    runtime.bind_graph_mask(model, batch["mask"])
    native_carry = runtime.initial_carry(model, batch)
    observed_carry = runtime.initial_carry(model, batch)
    rows, events = [], []
    h4_carry = None

    def observe(event, states):
        pooled = runtime.pooled_features(states)
        event = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in event.items()}
        event["pooled_features_finite"] = bool(all(torch.isfinite(value).all().item() for value in pooled.values()))
        events.append(event)

    with torch.no_grad():
        for outer in range(1, 5):
            before = _rng_state()
            step_started = time.perf_counter()
            with torch.autocast("cuda", enabled=False):
                native_carry, native_logits = model(native_carry, batch["x"])
            torch.cuda.synchronize(0)
            native_seconds = time.perf_counter() - step_started
            native_rng = _rng_unchanged(before)
            before = _rng_state()
            step_started = time.perf_counter()
            observed_carry, observed_logits = runtime.forward_outer(
                model, observed_carry, batch["x"], observer=observe, outer=outer
            )
            torch.cuda.synchronize(0)
            observed_seconds = time.perf_counter() - step_started
            observed_rng = _rng_unchanged(before)
            equal = _equal_tensors(native_carry, observed_carry, ("z_H", "z_L", "mixture_weights"))
            equal["logits"] = bool(torch.equal(native_logits, observed_logits))
            row = {
                "outer": outer,
                "exact_equal": equal,
                "maximum_absolute_error": {
                    "z_H": float((native_carry["z_H"].float() - observed_carry["z_H"].float()).abs().max()),
                    "z_L": float((native_carry["z_L"].float() - observed_carry["z_L"].float()).abs().max()),
                    "logits": float((native_logits.float() - observed_logits.float()).abs().max()),
                },
                "native_rng_unchanged": native_rng,
                "observed_rng_unchanged": observed_rng,
                "native_seconds": native_seconds,
                "observed_seconds": observed_seconds,
            }
            rows.append(row)
            if outer == 2:
                # H_cycles=2: fourth H update is the end of outer 2.
                h4_carry = {key: value.detach().cpu().clone() for key, value in native_carry.items()}
            print(json.dumps({"parity_outer_completed": row}), flush=True)
    torch.cuda.synchronize(0)
    for handle in dtype_handles:
        handle.remove()
    dtype_audit = {name: {key: sorted(value) if isinstance(value, set) else value for key, value in audit.items()} for name, audit in dtype_audit.items()}
    dtype_passed = forward_dtype != "float32" or all(
        set(audit["input_dtypes"] + audit["output_dtypes"]) <= {"torch.float32"}
        for audit in dtype_audit.values()
    )
    checks.mkdir(parents=True, exist_ok=True)
    artifact = {
        "purpose": "engineering_preflight_not_formal_results",
        "case_id": 0,
        "seed": 801000,
        "experiment_seed": SEED,
        "outer": 2,
        "h_update": 4,
        "forward_dtype": forward_dtype,
        "backend_flags": model.runtime_precision,
        "checkpoint_sha256": runtime.verify_checkpoint()['sha256'],
        "carry": h4_carry,
        "batch": {key: batch[key].detach().cpu() if torch.is_tensor(batch[key]) else batch[key] for key in ("x", "mask", "mix", "clauses", "n")},
        "lm_head_direction": (model.lm_head.weight[2] - model.lm_head.weight[1]).detach().cpu(),
    }
    torch.save(artifact, checks / "preflight_h4.pt")
    source_files = ['hrm/tasks/sat_flow.py','hrm/analyse/sat/rt/encoding.py','hrm/models/hrm_flow_bias.py','hrm/models/layers.py']
    passed = all(batch_parity.values()) and all(target_isolation.values()) and all(copied_target_isolation.values())
    passed = passed and all(all(row["exact_equal"].values()) and row["native_rng_unchanged"] and row["observed_rng_unchanged"] for row in rows)
    passed = passed and len(events) == 56 and all(event["pooled_features_finite"] for event in events)
    passed = passed and dtype_passed and value_audit["all_checkpoint_values_exactly_preserved"]
    result = {
        "purpose": "engineering_preflight_not_formal_results",
        "stage": "native_vs_observed_parity",
        "passed": passed,
        "gpu_before_start": idle,
        "device": str(next(model.parameters()).device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mode": "eager",
        "forward_dtype": forward_dtype,
        "backend_flags": model.runtime_precision,
        "state_dtype": str(native_carry["z_H"].dtype),
        "state_shapes": {name: list(value.shape) for name, value in native_carry.items()},
        "logits_shape": list(native_logits.shape),
        "input_shape": list(batch["x"].shape),
        "all_forward_dtypes_passed": dtype_passed,
        "submodule_dtype_audit": dtype_audit,
        "checkpoint_value_audit": value_audit,
        "experiment_seed": SEED,
        "case_id": 0,
        "input_seed": 801000,
        "outer_count": 4,
        "checkpoint_sha256": runtime.verify_checkpoint()['sha256'],
        "source_sha256": {name: sha256(repo_path(name)) for name in source_files},
        "runtime_sha256": sha256(Path(__file__).resolve().parent / "runtime.py"),
        "preflight_sha256": sha256(__file__),
        "batch_native_exact_equal": batch_parity,
        "target_isolation_exact_equal": target_isolation,
        "copied_common_target_isolation_exact_equal": copied_target_isolation,
        "outer_comparisons": rows,
        "observed_events": events,
        "h4_artifact": str((checks / "preflight_h4.pt").relative_to(study.DIRECTORY)),
        "h4_artifact_sha256": sha256(checks / "preflight_h4.pt"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (checks / "preflight_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"parity_passed": passed, "elapsed_seconds": result["elapsed_seconds"], "result": str(checks / "preflight_parity.json")}), flush=True)
    return result


def _maximum_abs(value) -> float:
    return float(value.detach().double().abs().max())


def _arm_summary(arm: dict) -> dict:
    result = {"kind": arm["kind"], "sign": arm["sign"]}
    for precision in ("reference_check", "native_check"):
        check = arm[precision]
        result[precision] = {
            "passed": check["passed"],
            "failure_counts": {name: int(mask.sum()) for name, mask in check["failures"].items()},
            "maximum_token_norm_relative_error": _maximum_abs(check["token_norm_relative_error"]),
            "maximum_token_relative_chord_absolute_error": _maximum_abs(check["token_relative_displacement_error"]),
            "pooled_relative_displacement": check["pooled_relative_displacement"].detach().cpu().tolist(),
            "pooled_probe_delta": check["probe_delta"].detach().cpu().tolist(),
            "maximum_token_probe_absolute_delta": _maximum_abs(check["token_probe_delta"]),
            "maximum_token_probe_normalized_delta": _maximum_abs(check["token_probe_delta_normalized"]),
            "maximum_pooled_probe_absolute_delta": _maximum_abs(check["probe_delta"]),
            "maximum_pooled_probe_normalized_delta": _maximum_abs(check["probe_delta_normalized"]),
        }
    result["cast_errors"] = {
        key: _maximum_abs(arm[key])
        for key in ("cast_probe_delta_error", "cast_token_probe_delta_error", "cast_token_chord_error", "cast_token_norm_error")
    }
    return result


@torch.no_grad()
def run_precision(alphas, *, device="cuda:0", forward_dtype="float32") -> dict:
    """Measure a geometry obstruction using an explicit, unfitted test direction.

    This tests both H/L at the fourth H update (outer 2). It neither validates
    SAT scientific tolerances nor uses a trained critic. No correction after
    native-precision casting, tolerance adjustment, failed-arm removal or reroll occurs.
    """
    from . import perturbations

    checks = _check_directory(forward_dtype)
    if str(device) != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Precision preflight is authorized on physical GPU0 only")
    idle = check_gpu0_idle()
    started = time.perf_counter()
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    backend_flags = runtime.configure_precision()
    artifact_path = checks / "preflight_h4.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact["outer"] != 2 or artifact["h_update"] != 4:
        raise ValueError("Geometry requires fourth H update, at the end of outer 2")
    if artifact["checkpoint_sha256"] != runtime.verify_checkpoint()['sha256']:
        raise ValueError("H4 artifact has the wrong checkpoint identity")
    if artifact["forward_dtype"] != forward_dtype or any(artifact["carry"][key].dtype != getattr(torch, forward_dtype) for key in ("z_H", "z_L")):
        raise ValueError("H4 artifact has the wrong forward precision")
    w = artifact["lm_head_direction"].to(device=device, dtype=torch.float64)
    covariance = torch.eye(len(w), device=device, dtype=torch.float64)
    tolerances = perturbations.Tolerances(
        token_norm_relative=REFERENCE_THRESHOLDS["relative_token_norm_error"],
        relative_chord_absolute=REFERENCE_THRESHOLDS["relative_chord_error"],
        control_probe_absolute=REFERENCE_THRESHOLDS["control_probe_absolute_error"],
        control_probe_normalized=REFERENCE_THRESHOLDS["control_probe_normalized_error"],
    )
    rows = []
    for module in ("H", "L"):
        state = artifact["carry"][f"z_{module}"].to(device)
        for alpha in alphas:
            generated = perturbations.generate_arms(state, w, covariance, float(alpha), SEED, tolerances=tolerances)
            row = {
                "module": module,
                "alpha": float(alpha),
                "passed": generated["passed"],
                "arms": {name: _arm_summary(arm) for name, arm in generated["arms"].items()},
                "matching": {
                    precision: {
                        "passed": audit["passed"],
                        "maximum_token_relative_chord_pairwise_span": _maximum_abs(audit["token_chord_pairwise_span"]),
                        "failed_token_count": int(audit["failed_tokens"].sum()),
                    }
                    for precision, audit in generated["matching"].items()
                },
            }
            # Retain raw per-token measurements and failure masks, but no full
            # generated state tensors or model rollout beyond the parity stage.
            measurements = {}
            for name, arm in generated["arms"].items():
                for precision in ("reference_check", "native_check"):
                    for key, value in arm[precision].items():
                        if torch.is_tensor(value):
                            measurements[f"{name}/{precision}/{key}"] = value.detach().cpu().numpy()
                        elif key == "failures":
                            for failure, mask in value.items():
                                measurements[f"{name}/{precision}/failure/{failure}"] = mask.detach().cpu().numpy()
            for precision, audit in generated["matching"].items():
                measurements[f"matching/{precision}/token_span"] = audit["token_chord_pairwise_span"].detach().cpu().numpy()
                measurements[f"matching/{precision}/failed_tokens"] = audit["failed_tokens"].detach().cpu().numpy()
            measurement_name = f"precision_geometry_{module}_alpha{float(alpha):g}.npz"
            np.savez_compressed(checks / measurement_name, **measurements)
            row["raw_measurements"] = str((checks / measurement_name).relative_to(study.DIRECTORY))
            row["raw_measurements_sha256"] = sha256(checks / measurement_name)
            rows.append(row)
            native = [arm["native_check"] for arm in row["arms"].values()]
            control = [arm["native_check"] for arm in row["arms"].values() if arm["kind"] == "control"]
            print(json.dumps({
                "geometry_test_not_fitted_critic": True,
                "module": module,
                "alpha": alpha,
                "passed": row["passed"],
                "maximum_token_norm_relative_error": max(arm["maximum_token_norm_relative_error"] for arm in native),
                "maximum_token_chord_absolute_error": max(arm["maximum_token_relative_chord_absolute_error"] for arm in native),
                "maximum_control_token_probe_error": max(arm["maximum_token_probe_absolute_delta"] for arm in control),
                "maximum_control_pooled_probe_error": max(arm["maximum_pooled_probe_absolute_delta"] for arm in control),
            }), flush=True)
            del generated, measurements
    torch.cuda.synchronize(0)
    result = {
        "purpose": "geometry_test_not_fitted_critic",
        "stage": "precision_obstruction_diagnostic",
        "formal_results": False,
        "passed": all(row["passed"] for row in rows),
        "precision_policy_decision": "await_user_if_failed",
        "threshold_status": "conservative_historical_Sudoku_engineering_gates_not_validated_SAT_tolerances",
        "control_gate_scope": "both_per_token_projection_and_pooled_readout; per_token_extension_is_stricter_than_historical_pooled_check",
        "thresholds": REFERENCE_THRESHOLDS,
        "gpu_before_start": idle,
        "device": str(device),
        "torch_version": torch.__version__,
        "forward_dtype": forward_dtype,
        "backend_flags": backend_flags,
        "state_dtype": forward_dtype,
        "state_shapes": {key: list(artifact["carry"][key].shape) for key in ("z_H", "z_L")},
        "geometry_dtype": "float64",
        "direction": "frozen_lm_head_weight_row2_minus_row1",
        "covariance": "identity; not a fitted training covariance",
        "case_id": artifact["case_id"],
        "outer": artifact["outer"],
        "h_update": artifact["h_update"],
        "experiment_seed": SEED,
        "control_seeds": [SEED + index for index in range(4)],
        "alphas": [float(alpha) for alpha in alphas],
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "artifact_sha256": sha256(artifact_path),
        "runtime_sha256": sha256(Path(__file__).resolve().parent / "runtime.py"),
        "perturbations_sha256": sha256(Path(__file__).resolve().parent / "perturbations.py"),
        "preflight_sha256": sha256(__file__),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (checks / "preflight_precision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"precision_passed": result["passed"], "elapsed_seconds": result["elapsed_seconds"], "result": str(checks / "preflight_precision.json")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--parity", action="store_true", help="run only the explicit GPU0 parity preflight")
    stages.add_argument("--precision", action="store_true", help="measure geometry precision; no fitted critic or formal run")
    parser.add_argument("--alphas", nargs="+", type=float, help="explicit positive alpha magnitudes for the engineering geometry test")
    parser.add_argument("--forward-dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.precision and not args.alphas:
        parser.error("--precision requires explicit --alphas")
    result = run_parity(args.forward_dtype) if args.parity else run_precision(args.alphas, forward_dtype=args.forward_dtype)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
