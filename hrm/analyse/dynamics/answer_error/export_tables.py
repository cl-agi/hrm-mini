"""Export the archived critic summaries without running a model or refitting probes."""
from hrm.paths import analysis_directory
from pathlib import Path
import csv
import json

ROOT = analysis_directory(__file__)
TARGETS = ("original_excess", "ce", "mse")


def read(path):
    return json.loads(path.read_text())


def build_tables(root=ROOT):
    source = root / "sources/answer_error"
    l_results = read(source / "fixed_state/RESULTS.json")["results"]
    h_results = read(source / "h_and_error_correlations_20261012/fixed_state_H/H_INTERVENTION_RESULTS.json")["targets"]
    correlations = read(source / "h_and_error_correlations_20261012/results.json")["cohorts"]["test"]
    tables = {name: [] for name in (
        "probe_test_metrics", "error_correlations", "next_L_effects",
        "first_H_effects", "initial_error_changes")}
    for target in TARGETS:
        probe_path = (root / "sources/original_excess/results.json" if target == "original_excess"
                      else source / "probes" / target / "results.json")
        for row in read(probe_path)["metrics"]:
            if row["split"] == "test":
                tables["probe_test_metrics"].append(dict(
                    target=target, method=row["method"], puzzles=128, events_per_puzzle=192,
                    **{k: row.get(k) for k in ("pearson_error", "r2_error", "rmse_error",
                        "pearson_next_L", "pearson_error_within_time", "pearson_next_L_within_time")}))
        tests = {row["dose"]: row for row in l_results[target]["paired_tests"]["rows"]}
        for row in l_results[target]["intervention_statistics"]["rows"]:
            dose = row["intervention_percent"] / 100
            if dose == 0:
                continue
            test = tests[dose]
            tables["next_L_effects"].append(dict(
                module="L", target=target, dose=dose, n_puzzles=81, event_index=0,
                mean_difference=row["paired_control_corrected_mean"],
                ci95_low=row["paired_control_corrected_ci95_low"],
                ci95_high=row["paired_control_corrected_ci95_high"],
                p_t=test["p_t"], p_t_holm=test["p_t_holm"],
                p_sign_flip_holm=test["p_sign_flip_holm"]))
        for endpoint, table in (("primary_next_L", "next_L_effects"),
                                ("secondary_first_H", "first_H_effects")):
            block = h_results[target]["endpoints"][endpoint]
            for row in block["rows"]:
                effect = row["paired_control_corrected"]
                tables[table].append(dict(
                    module="H", target=target, dose=row["dose"], n_puzzles=81,
                    event_index=block["event_index_zero_based"], mean_difference=effect["mean"],
                    ci95_low=effect["ci95"][0], ci95_high=effect["ci95"][1],
                    p_t=row["p_t"], p_t_holm=row["p_t_holm"],
                    p_sign_flip_holm=row["p_sign_flip_holm"]))
        for row in l_results[target]["actual_target_diagnostics"]:
            tables["initial_error_changes"].append(dict(
                module="L", target=target, dose=row["signed_state_displacement"], n_puzzles=81,
                probe_change_vs_native=row["mean_probe_change"],
                actual_change_vs_native=row["mean_actual_target_change"],
                actual_change_vs_matched_controls=row["mean_actual_target_control_corrected_change"],
                actual_corrected_ci95_low=None, actual_corrected_ci95_high=None,
                inference_scope="descriptive; no multiplicity-adjusted test"))
        readouts = h_results[target]["initial_readouts"]
        probe_rows = {row["dose"]: row for row in readouts["critic"]["rows"]}
        for row in readouts[h_results[target]["actual_target_field"]]["rows"]:
            effect = row["paired_control_corrected"]
            tables["initial_error_changes"].append(dict(
                module="H", target=target, dose=row["dose"], n_puzzles=81,
                probe_change_vs_native=probe_rows[row["dose"]]["change_from_native"]["mean"],
                actual_change_vs_native=row["change_from_native"]["mean"],
                actual_change_vs_matched_controls=effect["mean"],
                actual_corrected_ci95_low=effect["ci95"][0],
                actual_corrected_ci95_high=effect["ci95"][1],
                inference_scope="descriptive pointwise CI; no multiplicity-adjusted test"))
    for row in correlations["error_pairs"]:
        tables["error_correlations"].append(dict(
            x=row["x"], y=row["y"], puzzles=128, events_per_puzzle=192,
            pearson=row["pearson"], ci95_low=row["pearson_ci95"][0], ci95_high=row["pearson_ci95"][1],
            spearman=row["spearman"], time_centered_pearson=row["pearson_time_centered"],
            time_centered_ci95_low=row["pearson_time_centered_ci95"][0],
            time_centered_ci95_high=row["pearson_time_centered_ci95"][1]))
    return tables


def export(root=ROOT):
    destination = root / "summary_tables"
    destination.mkdir(exist_ok=True)
    tables = build_tables(root)
    for name, rows in tables.items():
        with (destination / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {name: len(rows) for name, rows in tables.items()}


if __name__ == "__main__":
    print(json.dumps(export()))
