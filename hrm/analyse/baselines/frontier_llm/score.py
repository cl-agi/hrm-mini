#!/usr/bin/env python3
"""Reparse and score the frozen 3,200 saved final responses, entirely offline."""
from hrm.paths import analysis_directory
import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import hrm.analyse.baselines.frontier_llm.frozen_code as frozen_code
import hrm.analyse.baselines.llm_sat.frozen_grading as frozen_grading

HERE = analysis_directory(__file__)
DATA = HERE / "frozen_data"
FLAGS = ("correct", "json_parse_success", "strict_json_object",
         "answer_schema_valid", "strict_output_format")
MODELS = {"openai/gpt-5.2", "google/gemini-3.1-pro-preview",
          "anthropic/claude-opus-4.5", "deepseek/deepseek-v3.2"}
SCALES = {"sat": {"8", "16", "32", "64", "128", "256"},
          "sudoku": {"9x9"}, "maze": {"30x30"}}
STATUSES = {"completed": 2647, "length_limited": 552, "no_final_content": 1}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    result = {row["job_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate job_id in " + path.name)
    return result


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def function_ast(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError("Expected one function " + name + " in " + path.name)
    return ast.dump(functions[0], include_attributes=False)


def source_provenance():
    """Record source hashes and check extracted definitions without importing audits."""
    paths = ["frozen_code.py", "frozen_grading.py",
             "frozen_sources/original/openrouter.py",
             "frozen_sources/original/dataset/sat3.py",
             "frozen_sources/original/dataset/sudoku.py",
             "frozen_sources/protocol/maze_shortest_v003.py",
             "frozen_sources/runtime/transport.py",
             "frozen_sources/runtime/audit_full.py"]
    files = [{"path": name, "sha256": sha((HERE / name).read_bytes())}
             for name in paths]
    identities = []
    for name, source in [
            ("robust_json_parse", "frozen_sources/original/openrouter.py"),
            ("independent_validate", "frozen_sources/runtime/audit_full.py"),
            ("regrade", "frozen_sources/runtime/audit_full.py")]:
        extracted = function_ast(HERE / "frozen_grading.py", name)
        original = function_ast(HERE / source, name)
        identities.append({"function": name, "source": source,
                           "ast_sha256": sha(extracted.encode()),
                           "ast_identical": extracted == original})
    return {"files": files, "extracted_function_identity": identities}


def score():
    answers = read_jsonl(DATA / "final_answers.jsonl.gz")
    questions = read_jsonl(DATA / "questions.jsonl.gz")
    expected_settings = read_csv(DATA / "current_settings.csv")
    if len(answers) != 3200 or set(questions) != set(answers):
        raise ValueError("Expected exactly the same 3,200 unique answer/question job IDs")
    expected_keys = {(task, model, scale) for task, scales in SCALES.items()
                     for model in MODELS for scale in scales}
    archived_keys = [(row["task"], row["model"], row["scale"])
                     for row in expected_settings]
    if len(archived_keys) != 32 or set(archived_keys) != expected_keys:
        raise ValueError("Archived settings do not match the frozen four-model, 32-setting cohort")

    errors = []

    def check(condition, code, identity):
        if not condition:
            errors.append({"code": code, "identity": identity})

    provenance = source_provenance()
    for entry in provenance["extracted_function_identity"]:
        check(entry["ast_identical"], "extracted_function_changed", entry["function"])

    cases, groups = [], defaultdict(list)
    for job, answer in answers.items():
        question = questions[job]
        task, model, scale = answer["task"], answer["model"], str(answer["scale"])
        if (task, model, scale) not in expected_keys:
            raise ValueError("Unexpected task/model/scale for " + job)
        for field in ("task", "model", "scale", "origin", "source_stage", "problem_sha256"):
            check(answer[field] == question[field], "question_answer_" + field, job)
        problem_hash = sha(canonical(question["problem"]).encode())
        check(problem_hash == question["problem_sha256"], "problem_sha256", job)
        final = answer["final_content"]
        if not isinstance(final, str):
            raise ValueError("Saved final_content must be a string for " + job)
        # Both scorers receive saved raw final text. Archived parsed_answer is
        # used only as an expectation after parsing has independently finished.
        original = frozen_code.original_grade(task, question["problem"], final)
        independent = frozen_grading.regrade(
            task, question["problem"], final, frozen_grading.robust_json_parse)
        for field in FLAGS:
            check(original[field] == independent[field], "original_vs_audit_" + field, job)
        check(canonical(original.get("parsed_answer")) == canonical(independent.get("parsed_answer")),
              "original_vs_audit_parsed_answer", job)
        for field in ("correct", "strict_output_format"):
            check(type(answer[field]) is bool and original[field] == answer[field],
                  "original_vs_archive_" + field, job)
        check(canonical(original.get("parsed_answer")) == canonical(answer.get("parsed_answer")),
              "original_vs_archive_parsed_answer", job)
        seconds = answer["elapsed_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid elapsed_seconds for " + job)
        check(math.isclose(answer["elapsed_minutes"], seconds / 60, abs_tol=1e-12),
              "elapsed_minutes", job)
        status = answer["selected_model_status"]
        check(status in STATUSES, "unexpected_terminal_status", job)
        check(answer["terminal_finish_reason"] == ("length" if status == "length_limited" else "stop"),
              "terminal_finish_reason", job)
        row = {"job_id": job, "task": task, "model": model, "scale": scale,
               "source_stage": answer["source_stage"], "origin": answer["origin"],
               "selected_model_status": status,
               **{field: original[field] for field in FLAGS},
               "independent_correct": independent["correct"],
               "archived_correct": answer["correct"],
               "elapsed_seconds": seconds, "elapsed_minutes": seconds / 60,
               "problem_sha256": problem_hash,
               "final_content_sha256": sha(final.encode()),
               "parsed_answer_sha256": sha(canonical(original.get("parsed_answer")).encode()),
               "selected_result_sha256": answer["selected_result_sha256"]}
        cases.append(row)
        groups[(task, model, scale)].append((row, answer))

    settings = []
    for expected in expected_settings:
        key = (expected["task"], expected["model"], expected["scale"])
        identity = "/".join(key)
        group = groups[key]
        if len(group) != 100:
            raise ValueError("Expected 100 trials for " + identity)
        rows = [row for row, _ in group]
        durations = [row["elapsed_minutes"] for row in rows]
        n_correct = sum(row["correct"] for row in rows)
        providers = Counter(canonical(json.loads(answer["provider_configuration_json"]))
                            for _, answer in group)
        stages = Counter(row["source_stage"] for row in rows)
        result = {"task": key[0], "model": key[1], "scale": key[2],
                  "n_trials": len(rows),
                  "n_unique_problems": len({row["problem_sha256"] for row in rows}),
                  "n_correct": n_correct, "accuracy": n_correct / len(rows),
                  "accuracy_percent": 100 * n_correct / len(rows),
                  "retained_trials": sum(row["origin"] == "retained" for row in rows),
                  "new_trials": sum(row["origin"] == "new" for row in rows),
                  **{status + "_trials": sum(row["selected_model_status"] == status for row in rows)
                     for status in STATUSES},
                  "selected_duration_count": len(durations),
                  "mean_minutes": mean(durations), "sample_sd_minutes": stdev(durations),
                  "strict_output_format_trials": sum(row["strict_output_format"] for row in rows),
                  "provider_groups_json": canonical([
                      {"configuration": json.loads(config), "n_trials": count}
                      for config, count in sorted(providers.items())]),
                  "source_stages_json": canonical(dict(stages))}
        for field in ("n_trials", "n_unique_problems", "n_correct", "retained_trials", "new_trials",
                      "completed_trials", "length_limited_trials", "no_final_content_trials",
                      "selected_duration_count"):
            check(result[field] == int(expected[field]), "setting_" + field, identity)
        for field in ("accuracy", "mean_minutes", "sample_sd_minutes"):
            check(math.isclose(result[field], float(expected[field]), rel_tol=1e-12, abs_tol=1e-12),
                  "setting_" + field, identity)
        expected_providers = {canonical(entry["configuration"]): entry["n_trials"]
                              for entry in json.loads(expected["provider_groups_json"])}
        check(providers == expected_providers, "setting_providers", identity)
        check(dict(stages) == json.loads(expected["source_stages_json"]), "setting_source_stages", identity)
        settings.append(result)

    statuses = dict(Counter(row["selected_model_status"] for row in cases))
    check(statuses == STATUSES, "terminal_status_counts", "cohort")
    total_correct = sum(row["correct"] for row in cases)
    check(total_correct == 885, "total_correct", "cohort")
    report = {"status": "verified" if not errors else "failed", "api_called": False,
              "scoring_input": "saved final_content, reparsed independently by both scorers",
              "selected_trials": len(cases), "settings": len(settings),
              "sat_settings": sum(row["task"] == "sat" for row in settings),
              "reparsed_trials": len(cases), "original_scored_trials": len(cases),
              "independently_scored_trials": len(cases), "n_correct": total_correct,
              "terminal_status_counts": statuses,
              "task_counts": {task: {"n_trials": sum(row["task"] == task for row in cases),
                                     "n_correct": sum(row["correct"] for row in cases if row["task"] == task)}
                              for task in SCALES},
              "timing": {"unit": "minutes", "sample_sd_ddof": 1,
                         "source": "selected request elapsed_seconds / 60"},
              "input_files": [{"path": "frozen_data/" + name,
                               "sha256": sha((DATA / name).read_bytes())}
                              for name in ("questions.jsonl.gz", "final_answers.jsonl.gz", "current_settings.csv")],
              "source_provenance": provenance, "mismatch_count": len(errors), "mismatches": errors}
    return cases, settings, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs/offline_regrade",
                        help="Derived CSV/JSON directory (default: package outputs/offline_regrade)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacement of derived files in an existing output directory")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    protected = (DATA.resolve(), (HERE / "frozen_sources").resolve())
    if output == HERE or output in HERE.parents or any(output == path or path in output.parents for path in protected):
        parser.error("Output must not replace the package or write inside frozen inputs/sources")
    if output.exists() and (not output.is_dir() or not args.overwrite):
        parser.error("Output exists; choose a new directory or explicitly pass --overwrite")
    try:
        cases, settings, report = score()
        output.mkdir(parents=True, exist_ok=args.overwrite)
        write_csv(output / "per_case.csv", cases)
        write_csv(output / "settings.csv", settings)
        (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("Offline scoring failed: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("status", "api_called", "selected_trials", "settings",
                     "sat_settings", "n_correct", "terminal_status_counts", "mismatch_count")}, indent=2))
    print("Derived files: " + str(output))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
