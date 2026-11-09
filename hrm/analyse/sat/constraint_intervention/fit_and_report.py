"""Fit frozen SAT probes and report complete FP32 intervention cohorts.

Usage: python fit_and_report.py {fit,report}
Neither command starts a model or chooses settings from intervention outcomes.
All paths and analysis settings come from the frozen study protocol.
"""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

import hrm.analyse.sat.constraint_intervention.study as study
from hrm.analyse.sat.constraint_intervention.probe_analysis import cluster_bootstrap_metrics, fit_ridge_probe, holm_adjust, paired_puzzle_effect, predict_probe


EVENT_MODULES = np.array(list("LLLLLLHLLLLLLH"))
EVENT_METRICS = ("delta_l2", "delta_rms", "source_l2", "source_rms",
                 "destination_l2", "destination_rms")
CONTROL_ARMS = tuple(f"random{k}_{sign}" for k in range(4) for sign in ("plus", "minus"))
ARMS = ("critic_plus", "critic_minus") + CONTROL_ARMS
ENDPOINTS = ("first_post_intervention_L_delta_l2", "final_violated_clauses",
             "success_any_continuation", "success_any_full_trajectory", "success_final_outer")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _scalar(data, key):
    _require(key in data and np.asarray(data[key]).shape == (), f"Missing scalar {key}")
    return data[key].item()


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _need_files(paths):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete cohort: {len(missing)} missing files; first: {missing[:8]}")


def _write_csv(path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(buffer.getvalue())
    temporary.replace(path)


def _settings(protocol):
    _require(protocol["execution"]["forward_dtype"] == "float32", "This analysis requires the frozen FP32 protocol")
    statistics = protocol["statistics"]
    return {"n_bootstrap": statistics["bootstrap_replicates"],
            "seed": statistics["bootstrap_seed"], "confidence": statistics["confidence"]}


def _validate_trajectory(data, outer, protocol):
    n, variables = len(outer), protocol["data"]["n_vars"]
    _require(np.array_equal(data["outer"], outer), "Wrong or incomplete outer-step sequence")
    _require(_scalar(data, "forward_dtype") == "float32", "Trajectory is not FP32")
    prediction = data["predictions"]
    _require(prediction.shape == (n, variables), "Wrong prediction shape")
    _require(np.issubdtype(prediction.dtype, np.integer) and np.isin(prediction, [0, 1, 2, 3]).all(),
             "Predictions must contain integer token codes 0..3")
    for key in ("violated", "legal", "valid"):
        _require(data[key].shape == (n,), f"Wrong {key} shape")
    violated = data["violated"]
    _require(np.issubdtype(violated.dtype, np.integer) and (violated >= 0).all()
             and (violated <= protocol["data"]["n_clauses"]).all(), "Invalid violated-clause counts")
    _require(data["legal"].dtype == np.bool_ and data["valid"].dtype == np.bool_, "Legality/validity must be boolean")
    _require(np.array_equal(data["legal"], np.isin(prediction, [1, 2]).all(axis=1)), "Legality disagrees with predictions")
    _require(np.array_equal(data["valid"], data["legal"] & (violated == 0)), "Success disagrees with legality/violations")
    _require(np.array_equal(data["event_module"], EVENT_MODULES), "Wrong update-event order")
    for metric in EVENT_METRICS:
        value = data["event_" + metric]
        _require(value.shape == (n, 14) and np.isfinite(value).all() and (value >= 0).all(),
                 f"Invalid event_{metric}")


def load_native_split(run, protocol, split):
    """Require every expected case and verify its complete manifest and hashes."""
    run = Path(run)
    count = protocol["data"][split]
    directory = run / "native" / split
    paths = [directory / f"case_{case:04d}.npz" for case in range(count)]
    manifests = [path.with_suffix(".complete.json") for path in paths]
    _need_files([run / "run_manifest.json"] + paths + manifests)
    manifest_hash = study.sha256(run / "run_manifest.json")
    outer = np.arange(1, protocol["trajectory"]["outer_cycles"] + 1)
    cases = []
    for case, path, complete_path in zip(range(count), paths, manifests):
        metadata = json.loads(complete_path.read_text())
        seed = protocol["trajectory"]["input_seed_base"] + case
        _require(metadata["status"] == "COMPLETE", f"Incomplete native case: {complete_path}")
        _require(metadata["case_id"] == case and metadata["split"] == split and metadata["input_seed"] == seed,
                 f"Native manifest identity mismatch: {complete_path}")
        _require(metadata["forward_dtype"] == "float32" and metadata["run_manifest_sha256"] == manifest_hash,
                 f"Native manifest precision/provenance mismatch: {complete_path}")
        required_artifacts = [path.name] + ([f"case_{case:04d}_h4.pt"] if split == "test" else [])
        for name in required_artifacts:
            artifact = directory / name
            _need_files([artifact])
            _require(metadata["sha256"].get(name) == study.sha256(artifact), f"Native hash mismatch: {artifact}")
        data = _read_npz(path)
        _require(_scalar(data, "case_id") == case and _scalar(data, "split") == split
                 and _scalar(data, "input_seed") == seed, f"Native NPZ identity mismatch: {path}")
        _validate_trajectory(data, outer, protocol)
        for module in protocol["probe"]["modules"]:
            features = data[module]
            _require(features.shape == (len(outer), protocol["trajectory"]["feature_dim"])
                     and features.dtype == np.float32 and np.isfinite(features).all(),
                     f"Invalid {module} features: {path}")
        cases.append(data)
    return cases


def _native_success(cases):
    return {"n_puzzles": len(cases),
            "success_any_outer_rate": float(np.mean([case["valid"].any() for case in cases])),
            "success_final_outer_rate": float(np.mean([case["valid"][-1] for case in cases]))}


def fit_run(run, protocol):
    """CPU fitting; caller supplies a frozen protocol (CLI verifies the freeze)."""
    run, bootstrap = Path(run), _settings(protocol)
    # Validate the entire collection before computing or writing any probe.
    splits = {split: load_native_split(run, protocol, split) for split in ("train", "valid", "test")}
    target = {split: np.concatenate([case["violated"] for case in cases]) for split, cases in splits.items()}
    case_ids = np.concatenate([np.full(len(case["outer"]), int(case["case_id"])) for case in splits["test"]])
    outer = np.concatenate([case["outer"] for case in splits["test"]])
    fitted, reports, predictions = {}, {}, {}
    for module in protocol["probe"]["modules"]:
        features = {split: np.concatenate([case[module] for case in cases]) for split, cases in splits.items()}
        probe = fit_ridge_probe(features["train"], target["train"], features["valid"], target["valid"],
                                alphas=protocol["probe"]["alphas"])
        prediction = predict_probe(probe, features["test"])
        fitted[module], predictions[module] = probe, prediction
        reports[module] = {key: value for key, value in probe.items()
                           if key not in ("weight", "mean", "covariance")}
        reports[module]["test"] = cluster_bootstrap_metrics(target["test"], prediction, case_ids, **bootstrap)
    summary = {
        "status": "COMPLETE", "forward_dtype": "float32", "target": protocol["probe"]["target"],
        "independent_unit": "puzzle", "validation_selection_only": True, "refit_on_validation": False,
        "bootstrap_scope": "fixed trained probes; evaluation puzzles resampled, no probe refitting",
        "native_success": {split: _native_success(cases) for split, cases in splits.items()},
        "probes": reports, "run_manifest_sha256": study.sha256(run / "run_manifest.json"),
    }
    directory = run / "probes"
    for module, probe in fitted.items():
        study.write_npz(directory / f"{module}.npz", **{key: probe[key] for key in
                        ("weight", "intercept", "mean", "covariance", "alpha", "lambda_primal")})
        study.write_json(directory / f"{module}.json", reports[module])
    study.write_npz(directory / "test_predictions.npz", case_id=case_ids, outer=outer,
                    violated=target["test"], **predictions)
    study.write_json(directory / "fit_summary.json", summary)
    return summary


def _arm_path(run, case, module, alpha, arm):
    return Path(run) / "interventions" / "test" / f"case_{case:04d}" / module / f"alpha_{alpha:g}" / f"{arm}.npz"


def _load_arm(path, protocol, *, case, module, alpha, arm, provenance):
    metadata = json.loads(path.with_suffix(".json").read_text())
    _require(metadata["status"] == "COMPLETE" and metadata["sha256"] == study.sha256(path),
             f"Incomplete intervention or hash mismatch: {path}")
    _require(all(metadata["provenance"].get(key) == value for key, value in provenance.items()),
             f"Intervention frozen-run/probe/native provenance mismatch: {path}")
    data = _read_npz(path)
    expected = {"case_id": case, "module": module, "alpha": alpha, "arm": arm,
                "sign": 1 if arm.endswith("plus") else -1,
                "input_seed": protocol["trajectory"]["input_seed_base"] + case,
                "geometry_passed": True}
    for key, value in expected.items():
        _require(_scalar(data, key) == value, f"Intervention {key} mismatch: {path}")
    outer = np.arange(protocol["intervention"]["fork_after_outer"] + 1,
                      protocol["intervention"]["continue_through_outer"] + 1)
    _validate_trajectory(data, outer, protocol)
    return data


def _endpoints(data, prefix_success):
    continuation = bool(data["valid"].any())
    return np.array([data["event_delta_l2"][0, 0], data["violated"][-1], continuation,
                     bool(prefix_success) or continuation, data["valid"][-1]], dtype=np.float64)


def _descriptive_effect(control, treatment, case_ids, bootstrap):
    effect = paired_puzzle_effect(control, treatment, case_ids, **bootstrap)
    return {key: value for key, value in effect.items() if not key.startswith("ttest_")}


def report_run(run, protocol):
    """Report only a complete intervention cohort; never omit failed cases/arms."""
    run, bootstrap = Path(run), _settings(protocol)
    native = load_native_split(run, protocol, "test")
    case_ids = np.arange(len(native))
    modules = protocol["intervention"]["modules"]
    doses = protocol["intervention"]["signed_relative_displacements"]
    _require(len(doses) == 6 and len(set(doses)) == 6
             and len(set(abs(dose) for dose in doses)) == 3, "Expected six signed doses per module")
    magnitudes = sorted(set(abs(dose) for dose in doses))
    required = [_arm_path(run, case, module, alpha, arm)
                for case in case_ids for module in modules for alpha in magnitudes for arm in ARMS]
    probe_paths = {module: run / "probes" / f"{module}.npz" for module in modules}
    _need_files(required + [path.with_suffix(".json") for path in required] + list(probe_paths.values()))
    probe_hashes = {module: study.sha256(path) for module, path in probe_paths.items()}
    run_hash = study.sha256(run / "run_manifest.json")
    native_hashes = [json.loads((run / "native" / "test" / f"case_{case:04d}.complete.json").read_text())["sha256"]
                     for case in case_ids]
    fork = protocol["intervention"]["fork_after_outer"]
    time_metrics = ("violated", "legal", "valid") + tuple("event_" + metric for metric in EVENT_METRICS)
    prefix = [bool(case["valid"][:fork].any()) for case in native]
    native_tail = [{key: case[key][fork:] for key in time_metrics} for case in native]
    native_values = np.stack([_endpoints(case, before) for case, before in zip(native_tail, prefix)])
    # Everything below is held in memory until every arm passes completeness,
    # provenance and schema validation. A late failure writes no partial report.
    groups = []
    for module in modules:
        for alpha in magnitudes:
            endpoints = {arm: [] for arm in ARMS}
            timecourses = {arm: {key: [] for key in time_metrics} for arm in ARMS}
            for case in case_ids:
                provenance = {"run_manifest_sha256": run_hash, "probe_sha256": probe_hashes[module],
                              "native_npz_sha256": native_hashes[case][f"case_{case:04d}.npz"],
                              "h4_sha256": native_hashes[case][f"case_{case:04d}_h4.pt"]}
                for arm in ARMS:
                    path = _arm_path(run, case, module, alpha, arm)
                    data = _load_arm(path, protocol, case=int(case), module=module, alpha=alpha,
                                     arm=arm, provenance=provenance)
                    endpoints[arm].append(_endpoints(data, prefix[case]))
                    for key in time_metrics:
                        timecourses[arm][key].append(data[key])
            values = {arm: np.stack(rows) for arm, rows in endpoints.items()}
            controls = np.mean([values[arm] for arm in CONTROL_ARMS], axis=0)
            control_time = {key: np.mean([timecourses[arm][key] for arm in CONTROL_ARMS], axis=0)
                            for key in time_metrics}
            for sign, arm in ((1, "critic_plus"), (-1, "critic_minus")):
                treatment = values[arm]
                primary = paired_puzzle_effect(controls[:, 0], treatment[:, 0], case_ids, **bootstrap)
                secondary = {key: _descriptive_effect(controls[:, j], treatment[:, j], case_ids, bootstrap)
                             for j, key in enumerate(ENDPOINTS[1:], 1)}
                native_contrast = {key: _descriptive_effect(native_values[:, j], treatment[:, j], case_ids, bootstrap)
                                   for j, key in enumerate(ENDPOINTS)}
                groups.append({"module": module, "signed_dose": sign * alpha,
                               "summary": {"primary": primary, "secondary": secondary,
                                           "native_contrast_descriptive": native_contrast},
                               "treatment": treatment, "control": controls,
                               "time_treatment": {key: np.asarray(timecourses[arm][key], dtype=np.float64) for key in time_metrics},
                               "time_control": control_time})
    groups.sort(key=lambda group: (modules.index(group["module"]), doses.index(group["signed_dose"])))
    for module in modules:
        family = [group for group in groups if group["module"] == module]
        adjusted = holm_adjust([group["summary"]["primary"]["ttest_pvalue"] for group in family])
        for group, pvalue in zip(family, adjusted):
            group["summary"]["primary"]["holm_pvalue"] = pvalue
            group["summary"]["primary"]["holm_family_size"] = 6
    summary = {
        "status": "COMPLETE", "forward_dtype": "float32", "n_puzzles": len(native),
        "independent_unit": "puzzle", "primary_endpoint": ENDPOINTS[0],
        "primary_comparison": "critic arm minus mean of all eight matched random controls within each puzzle",
        "multiplicity": "Holm adjustment of six signed-dose primary t-tests separately for H and L",
        "secondary_inference": "descriptive paired effect and puzzle-bootstrap interval; no secondary p-values",
        "success_any_continuation": f"any legal satisfying assignment after outer {fork}, through continuation end",
        "success_any_full_trajectory": f"native success in outer 1..{fork} OR success in the arm continuation",
        "native_success": _native_success(native), "bootstrap": bootstrap,
        "timecourse_inference": "descriptive means only; full puzzle-level arrays retained, no timepoint p-values",
        "run_manifest_sha256": run_hash,
        "comparisons": [{"module": group["module"], "signed_dose": group["signed_dose"], **group["summary"]}
                        for group in groups],
    }
    directory = run / "reports"
    endpoint_rows, time_rows = [], []
    continuation_outer = np.arange(fork + 1, protocol["intervention"]["continue_through_outer"] + 1)
    for group in groups:
        module, dose = group["module"], group["signed_dose"]
        label = f"{module}_{dose:+g}"
        arrays = {"case_id": case_ids, "outer": continuation_outer}
        for key in time_metrics:
            treatment = group["time_treatment"][key]
            control = group["time_control"][key]
            baseline = np.asarray([case[key] for case in native_tail])
            arrays.update({f"{key}_critic": treatment, f"{key}_control_mean": control,
                           f"{key}_native": baseline, f"{key}_effect": treatment - control,
                           f"{key}_native_effect": treatment - baseline})
            for index, outer in enumerate(continuation_outer):
                events = range(14) if key.startswith("event_") else [None]
                for event in events:
                    position = (slice(None), index, event) if event is not None else (slice(None), index)
                    t, c, b = treatment[position], control[position], baseline[position]
                    time_rows.append({"module": module, "signed_dose": dose, "outer": int(outer),
                                      "event_index": event, "event_module": EVENT_MODULES[event] if event is not None else "",
                                      "metric": key, "critic_mean": float(t.mean()), "control_mean": float(c.mean()),
                                      "native_mean": float(b.mean()), "effect_mean": float((t - c).mean()),
                                      "native_effect_mean": float((t - b).mean())})
        study.write_npz(directory / "timecourse" / f"{label}.npz", **arrays)
        for case in case_ids:
            for j, endpoint in enumerate(ENDPOINTS):
                t, c, b = group["treatment"][case, j], group["control"][case, j], native_values[case, j]
                endpoint_rows.append({"case_id": int(case), "module": module, "signed_dose": dose,
                                      "endpoint": endpoint, "critic": float(t), "control_mean": float(c),
                                      "native": float(b), "effect": float(t - c), "native_effect": float(t - b)})
    study.write_npz(directory / "per_puzzle_effects.npz", case_id=case_ids, endpoints=np.array(ENDPOINTS),
                    module=np.array([group["module"] for group in groups]),
                    signed_dose=np.array([group["signed_dose"] for group in groups]),
                    critic=np.stack([group["treatment"] for group in groups]),
                    control_mean=np.stack([group["control"] for group in groups]), native=native_values,
                    effect=np.stack([group["treatment"] - group["control"] for group in groups]))
    _write_csv(directory / "per_puzzle_effects.csv", endpoint_rows, list(endpoint_rows[0]))
    _write_csv(directory / "timecourse_summary.csv", time_rows, list(time_rows[0]))
    study.write_json(directory / "intervention_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "report"))
    arguments = parser.parse_args()
    protocol = study.assert_frozen()
    summary = fit_run(study.RUN, protocol) if arguments.command == "fit" else report_run(study.RUN, protocol)
    print(json.dumps({"status": summary["status"], "command": arguments.command, "run": str(study.RUN)}))


if __name__ == "__main__":
    main()
