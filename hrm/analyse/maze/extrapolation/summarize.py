"""CPU-only audit of the latest D three-seed 45--80 Maze extension.

The parent 15--40 results remain immutable. New final outputs are published
only when all three seeds and all four larger test sets pass complete replay.
"""
from __future__ import annotations
from hrm.paths import analysis_directory

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = analysis_directory(__file__)
from . import maze_evaluation_main_results_retest_20261011_summarize as parent
PARENT_SUMMARIZER = Path(parent.__file__)
require, sha256, write_json = parent.require, parent.sha256, parent.write_json
Pending, score_predictions = parent.Pending, parent.score_predictions
METRIC_NAME, METRIC_VERSION = parent.METRIC_NAME, parent.METRIC_VERSION
LARGE_NAMES = ("maze_45", "maze_50", "maze_60", "maze_80")
ALL_NAMES = tuple(parent.DATASETS) + LARGE_NAMES


def frozen_parent(manifest):
    declaration = manifest["parent_results"]
    path = Path(declaration["path"]).resolve()
    require(sha256(path) == declaration["sha256"], "Parent results changed since this extension was frozen")
    result = json.loads(path.read_text())
    require(result.get("status") == "completed" and result.get("metric_version") == METRIC_VERSION,
            "Parent results are not a completed shortest-path evaluation")
    require(result.get("T") == 16 and result.get("weights") == "EMA"
            and result.get("checkpoint_epoch_zero_based") == 19, "Parent inference protocol differs")
    require(set(result["D_eight_datasets"]) == set(parent.DATASETS), "Parent D dataset inventory differs")
    d_jobs = {job["seed"]: job for job in result["verified_jobs"] if job["variant"] == "D_2D"}
    require(set(d_jobs) == {1, 2, 3}, "Parent must contain exactly the latest D three seeds")
    for job in manifest["jobs"]:
        prior = d_jobs[job["seed"]]
        require(Path(prior["checkpoint"]).resolve() == Path(job["checkpoint"]).resolve()
                and prior["checkpoint_sha256"] == job["checkpoint_sha256"],
                f"Larger-size run uses a different checkpoint from the parent: {job['id']}")
    return result


def load_large_sources(manifest):
    declared = manifest["datasets"]
    require(set(declared) == set(LARGE_NAMES), "Expected exactly 45, 50, 60, 80 test datasets")
    arrays, sources = {}, {}
    bundle_path = Path(manifest["large_data_root"]) / "manifest.json"
    require(sha256(bundle_path) == manifest["large_data_manifest_sha256"], "Frozen large-data bundle manifest changed")
    bundle = {entry["path"]: entry for entry in json.loads(bundle_path.read_text())["datasets"]}
    for name in LARGE_NAMES:
        entry = declared[name]
        path = Path(entry["path"]).resolve()
        require(entry["sha256"] == bundle[entry["relative_path"]]["sha256"]
                and path == (Path(manifest["large_data_root"]) / entry["relative_path"]).resolve(),
                f"{name}: dataset differs from frozen large-data bundle")
        require(sha256(path) == entry["sha256"], f"Frozen large dataset changed: {name}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        size = int(name.split("_")[1])
        require(len(rows) == 1000, f"{name}: expected 1000 rows")
        require(len({row['question'] for row in rows}) == 1000, f"{name}: duplicate source questions")
        require(all(len(row["question"]) == len(row["answer"]) == size * size for row in rows),
                f"{name}: wrong board dimensions")
        inputs = np.asarray([[0] + [parent.VOCAB[c] for c in row["question"]] for row in rows], dtype=np.int32)
        targets = np.asarray([[0] + [parent.VOCAB[c] for c in row["answer"]] for row in rows], dtype=np.int32)
        require(score_predictions(inputs, targets, targets, size=size)["shortest_path_correct"].all(),
                f"{name}: source includes a non-shortest reference answer")
        arrays[name] = inputs, targets
        sources[name] = {"path":str(path), "sha256":entry["sha256"], "examples":1000, "size":size,
                         "seq_len":size*size+1, "unique_questions":1000}
    return sources, arrays


def verify_large_dataset(job, name, metrics, paths, expected):
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            require(int(shard["world_size"]) == 8, f"{path}: expected the original eight-rank evaluation layout")
    arrays, stored_flags, indices, hashes = parent.load_prediction_parts(paths, require_indices=True)
    inputs, targets, predictions = (arrays[key] for key in ("inputs", "targets", "predictions"))
    expected_inputs, expected_targets = expected
    require(inputs.shape == targets.shape == predictions.shape == expected_inputs.shape,
            f"{job['id']}/{name}: prediction shape or total differs from 1000 frozen examples")
    require(indices is not None and np.array_equal(np.sort(indices), np.arange(1000)),
            f"{job['id']}/{name}: missing/duplicate original row indices")
    # Original rows are unique; exact indexed equality also proves unique coverage.
    require(np.array_equal(inputs, expected_inputs[indices]), f"{job['id']}/{name}: input/index identity mismatch")
    require(np.array_equal(targets, expected_targets[indices]), f"{job['id']}/{name}: reference labels changed")
    scores = score_predictions(inputs, predictions, targets)
    for key, flags in stored_flags.items():
        require(np.array_equal(flags, scores[key]), f"{job['id']}/{name}: saved {key} disagrees with CPU replay")
    shortest, valid, canonical, format_correct = (scores[key] for key in parent.FLAGS)
    same_board = np.all(predictions[:,1:] == targets[:,1:], axis=1)
    alternatives = shortest & ~same_board
    bos_only = shortest & same_board & ~canonical
    require(not np.any(canonical & ~shortest), "Exact shortest references must imply shortest-path correctness")
    require(int(canonical.sum()) + int(alternatives.sum()) + int(bos_only.sum()) == int(shortest.sum()),
            "Shortest path partition failed")
    require(metrics.get("metric_version") == METRIC_VERSION, f"Wrong metric version: {job['id']}/{name}")
    parent.assert_metric(metrics.get("total"), 1000, "total")
    parent.assert_metric(metrics.get("correct"), int(shortest.sum()), "correct")
    parent.assert_metric(metrics.get(METRIC_NAME), float(shortest.mean()), METRIC_NAME)
    parent.assert_metric(metrics.get("exact_match"), float(canonical.mean()), "reference exact-match")
    for key, expected_value in (("accuracy",shortest.mean()), ("valid_path_accuracy",valid.mean()),
                                ("valid_path_rate",valid.mean()), ("format_accuracy",format_correct.mean()),
                                ("format_correct_rate",format_correct.mean())):
        if key in metrics:
            parent.assert_metric(metrics[key], float(expected_value), key)
    # These new sizes have no historical result for these three checkpoints.
    declared_old = job.get("old_exact_match")
    require(declared_old is None or (isinstance(declared_old,dict) and all(value is None for value in declared_old.values())),
            "New large sizes must not import old-model or guessed historical scores")
    return {"dataset":name, "seed":job["seed"], "job_id":job["id"], "variant":"D_2D",
            "metric_name":METRIC_NAME, "metric_version":METRIC_VERSION, "total":1000,
            "shortest_path_correct":int(shortest.sum()), "shortest_path_accuracy":float(shortest.mean()),
            "valid_path_correct":int(valid.sum()), "valid_nonshortest_count":int((valid & ~shortest).sum()),
            "invalid_path_count":int((~valid).sum()), "canonical_correct":int(canonical.sum()),
            "canonical_accuracy":float(canonical.mean()), "alternative_shortest_count":int(alternatives.sum()),
            "reference_route_wrong_bos_count":int(bos_only.sum()), "bos_only_reference_mismatch":int(bos_only.sum()),
            "format_correct_count":int(format_correct.sum()), "old_canonical_correct":None,
            "canonical_delta_vs_old_count":None, "canonical_replay_matches_old":None,
            "coverage":{"unique_questions":1000,"matches_frozen_source":True,"D_indices_complete":True},
            "prediction_files":hashes}


def verify_job(root, job, expected_arrays):
    directory = root / "results" / job["id"]
    path = directory / "metrics.json"
    if not path.is_file():
        raise Pending(f"{job['id']}: metrics.json pending")
    result = json.loads(path.read_text())
    metadata_path = directory / "run_metadata.json"
    require(metadata_path.is_file(), f"Missing committed run provenance: {job['id']}")
    metadata = json.loads(metadata_path.read_text())
    require(metadata.get("manifest_sha256") == sha256(root / "manifest.json"),
            f"{job['id']}: metadata belongs to a different/archived attempt manifest")
    original_metadata = json.loads((Path(job["run_dir"]) / "run_metadata.json").read_text())
    require(metadata.get("training_identity") == original_metadata["identity"],
            f"{job['id']}: original training identity was not preserved")
    evaluation_identity = metadata.get("evaluation_identity", {})
    require(evaluation_identity.get("experiment") == "D" and evaluation_identity.get("seed") == job["seed"],
            f"{job['id']}: missing or mismatched evaluation identity")
    original_architecture = json.loads((Path(job["run_dir"]) / "resolved_config.json").read_text())["architecture"]
    require(metadata.get("original_architecture") == original_architecture,
            f"{job['id']}: original model architecture changed")
    allowed_architecture = dict(original_architecture)
    allowed_architecture["allowed_sizes"] = sorted(set(allowed_architecture["allowed_sizes"]) | {45,50,60,80})
    require(metadata.get("architecture") == allowed_architecture and metadata.get("world_size") == 8,
            f"{job['id']}: unauthorized model configuration/evaluation layout change")
    checkpoint = Path(job["checkpoint"]).resolve()
    require(sha256(checkpoint) == job["checkpoint_sha256"], f"Checkpoint changed: {job['id']}")
    require(Path(result["checkpoint"]).resolve() == checkpoint, f"Wrong result checkpoint: {job['id']}")
    require(result.get("epoch") == 19 and result.get("weights") == "EMA" and result.get("seed") == job["seed"] and result.get("T") == 16,
            f"Wrong final epoch/weights/seed: {job['id']}")
    require(result.get("checkpoint_sha256") == job["checkpoint_sha256"], f"Wrong checkpoint digest in result: {job['id']}")
    require(set(result["metrics"]) == set(LARGE_NAMES), f"Wrong large-size result inventory: {job['id']}")
    rows=[]
    for name in LARGE_NAMES:
        paths=sorted((directory / "metrics_predictions" / name).glob("rank_*.npz"))
        if not paths:
            raise Pending(f"{job['id']}/{name}: prediction shards pending")
        rows.append(verify_large_dataset(job,name,result["metrics"][name],paths,expected_arrays[name]))
    return {"id":job["id"],"variant":"D_2D","seed":job["seed"],"checkpoint":str(checkpoint),
            "checkpoint_sha256":job["checkpoint_sha256"],"metrics_file":str(path.resolve()),
            "metrics_sha256":sha256(path),"run_metadata_file":str(metadata_path.resolve()),
            "run_metadata_sha256":sha256(metadata_path),"datasets":rows}


def render_report(result):
    lines=["Latest three-seed D evaluation: complete Maze tests up to 80×80",f"Metric version: {METRIC_VERSION}",
           "Fixed existing epoch-20 EMA checkpoints, T16, seeds 1/2/3; no retraining.",
           "Training sizes 15/20/25/30, with 250 instances each and 1,000 total; sizes 32 and above are outside the training range.",
           "All 1,000 instances per test set and seed; ± denotes sample SD across three seeds.",
           "The original eight sets come from the verified parent report. Sizes 45/50/60/80 are first-time evaluations of these three checkpoints, without historical comparison values.",
           "This table excludes the earlier LR1e-4 D, other models and cloud timing results.","",
           "Twelve test sets (%)", "Dataset             Shortest-path mean±SD    seed1 / seed2 / seed3    Same-prediction reference exact-match mean±SD"]
    for name in ALL_NAMES:
        row=result["D_twelve_datasets"][name]
        semantic,canonical=row[METRIC_NAME],row["canonical_accuracy"]
        seeds=" / ".join(f"{100*v:.1f}" for v in semantic["per_seed"])
        lines.append(f"{name:<20} {100*semantic['mean']:.2f}±{100*semantic['sample_std']:.2f}    {seeds}    "
                     f"{100*canonical['mean']:.2f}±{100*canonical['sample_std']:.2f}")
    lines += ["", "Additional large-size counts by seed (1,000 instances per row)", "run / dataset: Shortest path | Reference exact match | Alternative shortest path | BOS error only | Valid non-shortest | Invalid"]
    for job in result["verified_large_jobs"]:
        for row in job["datasets"]:
            lines.append(f"{job['id']} / {row['dataset']}: {row['shortest_path_correct']} | {row['canonical_correct']} | "
                         f"{row['alternative_shortest_count']} | {row['bos_only_reference_mismatch']} | "
                         f"{row['valid_nonshortest_count']} | {row['invalid_path_count']}")
    lines += ["", "Any valid shortest path is accepted. BFS verifies the original shortest distance without repairing or replacing predictions.",
              "Alternative shortest paths must differ on the board; BOS-only differences are counted separately, as are valid non-shortest and invalid predictions.",
              "All additional NPZ outputs are rescored on CPU, checking 1,000 complete rows, original indices, unique puzzles, original CSV labels, and data/checkpoint SHA256.",
              "Historical exact match and historical differences are null for new sizes; results from another earlier D model cannot substitute for them.",
              f"Parent results: {result['parent_results']['path']}",
              "Earlier checkpoints, original eight-set results and reports are not overwritten."]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args(argv)
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    status=root/"summary_status.json"
    try:
        manifest_path=root/"manifest.json"
        if not manifest_path.is_file():raise Pending("Frozen manifest.json pending")
        manifest=json.loads(manifest_path.read_text());jobs=manifest["jobs"]
        require(len(jobs)==3 and len({job['id'] for job in jobs})==3
                and {(job['variant'],job['seed']) for job in jobs}=={('D_2D',seed) for seed in (1,2,3)},
                "Expected precisely the latest D three seeds")
        require(manifest.get("metric_version")==METRIC_VERSION and manifest.get("cycles")==16
                and manifest.get("checkpoint_epoch_zero_based")==19 and manifest.get("weights")=="EMA",
                "Frozen evaluation protocol mismatch")
        scorer_hash=sha256(ROOT.parent/"scoring.py")
        require(manifest["scorer_sha256"]==scorer_hash,"Frozen scorer changed")
        source_root=Path(manifest["project"]).resolve()
        for relative,expected in manifest["source_sha256"].items():
            require(sha256(source_root/relative)==expected,f"Frozen inference source changed: {relative}")
        for relative,expected in manifest["extension_source_sha256"].items():
            require(sha256(root/relative)==expected,f"Frozen extension source changed: {relative}")
        require(sha256(root/"checks.json")==manifest["checks_sha256"],"Frozen CPU check receipt changed")
        prior=frozen_parent(manifest)
        sources,arrays=load_large_sources(manifest)
        verified,pending=[],[]
        for job in jobs:
            try:verified.append(verify_job(root,job,arrays))
            except Pending as error:pending.append(str(error))
        if pending:
            write_json(status,{"status":"pending","verified_jobs":[job['id'] for job in verified],
                               "pending":pending,"final_results_published":False})
            print(json.dumps({"status":"pending","verified":len(verified),"total":3,"pending":pending}))
            return 2
        rows=[row for job in verified for row in job["datasets"]]
        large={name:parent.aggregate([row for row in rows if row['dataset']==name]) for name in LARGE_NAMES}
        all_datasets=dict(prior["D_eight_datasets"],**large)
        result={"status":"completed","metric_name":METRIC_NAME,"metric_version":METRIC_VERSION,
                "T":16,"weights":"EMA","checkpoint_epoch_zero_based":19,"verification_device":"CPU",
                "manifest_sha256":sha256(manifest_path),"scorer_sha256":scorer_hash,
                "summarizer_sha256":sha256(Path(__file__)),
                "source_sha256":manifest["source_sha256"],"extension_source_sha256":manifest["extension_source_sha256"],
                "checks_sha256":manifest["checks_sha256"],"large_data_manifest_sha256":manifest["large_data_manifest_sha256"],
                "parent_summarizer_sha256":sha256(PARENT_SUMMARIZER),"parent_results":manifest["parent_results"],
                "new_datasets":sources,"verified_large_jobs":verified,
                "D_four_large_datasets":large,"D_twelve_datasets":all_datasets}
        output=root/"final_results.json"
        if output.exists():require(json.loads(output.read_text())==result,"Published final results differ; refusing overwrite")
        else:write_json(output,result)
        report=render_report(result);report_path=root/"report.txt"
        if report_path.exists():require(report_path.read_text()==report,"Published report differs; refusing overwrite")
        else:parent.atomic_write(report_path,report)
        write_json(status,{"status":"completed","verified_jobs":3,"verified_new_dataset_evaluations":12,
                           "combined_D_datasets":12,"final_results_published":True,"final_results":str(output)})
        print(report);return 0
    except Pending as error:
        write_json(status,{"status":"pending","reason":str(error),"final_results_published":False})
        print(f"PENDING: {error}");return 2
    except (OSError,ValueError,KeyError,TypeError) as error:
        write_json(status,{"status":"verification_failed","error":f"{type(error).__name__}: {error}",
                           "final_results_published":False})
        print(f"VERIFICATION FAILED: {error}",file=sys.stderr);return 1


if __name__=="__main__":
    raise SystemExit(main())
