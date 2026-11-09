"""Verify frozen requests offline, or explicitly execute a new independent call group.

The network transport, parsing, grading, retry classification and attempt selection
are the unchanged historical implementation. This adapter replaces site-local
input paths and launch authorization with packaged inputs and an explicit CLI.
"""
from hrm.paths import analysis_directory

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import signal
import statistics
import sys
import threading

import hrm.analyse.baselines.frontier_llm.frozen_code as frozen_code


ROOT = Path(__file__).resolve().parent
MODELS = ("openai/gpt-5.2", "google/gemini-3.1-pro-preview",
          "anthropic/claude-opus-4.5", "deepseek/deepseek-v3.2")
BACKOFF_SECONDS = (5, 15, 45, 120, 300)
SOCKET_TIMEOUT_SECONDS = 180


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index(rows):
    result = {row["job_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate archived job identity")
    return result


def prepare(data_dir=None):
    """Rebuild all 3,200 requests from the original formatters and model profiles."""
    data_dir = Path(data_dir) if data_dir is not None else analysis_directory(__file__) / "frozen_data"
    source_hashes = frozen_code.verify_sources()
    original = frozen_code.load_original()
    digest = original.transport.problem_hash
    questions = _index(read_rows(data_dir / "questions.jsonl.gz"))
    requests = _index(read_rows(data_dir / "requests.jsonl.gz"))
    answers = _index(read_rows(data_dir / "final_answers.jsonl.gz"))
    profiles = json.loads((data_dir / "profiles.json").read_text())
    if len(questions) != 3200 or set(questions) != set(requests) or set(questions) != set(answers):
        raise ValueError("Expected the complete 3,200-trial frozen cohort")
    jobs = []
    for job_id in sorted(questions):
        question, request, answer = questions[job_id], requests[job_id], answers[job_id]
        path = PurePosixPath(job_id)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe archived job identity")
        task, model, problem = question["task"], question["model"], question["problem"]
        if task not in {"sudoku", "maze", "sat"} or model not in MODELS:
            raise ValueError("Unexpected task or model outside the formal four-model cohort")
        problem_sha = digest(problem)
        for row in (question, request, answer):
            if (row["task"] != task or row["model"] != model
                    or row["problem_sha256"] != problem_sha):
                raise ValueError("Archived problem identity mismatch: " + job_id)
        profile = profiles[request["profile_key"]]
        dimensions = {"height": 30, "width": 30} if task == "maze" else {}
        formatter = original.formatters[task]
        prompt = formatter.format_prompt(**problem, **dimensions)
        admission = original.engine.context_admission(prompt, profile)
        if not admission["admission_passed"]:
            raise ValueError("Frozen request fails original context admission: " + job_id)
        payload = original.engine.build_payload(prompt, model, profile)
        payload_sha = digest(payload)
        if (payload != request["payload"] or payload_sha != request["payload_sha256"]
                or payload_sha != answer["payload_sha256"]):
            raise ValueError("Reconstructed payload differs from archived request: " + job_id)
        grader = ("protocol/maze_shortest_v003.py" if task == "maze"
                  else "original/dataset/" + {"sudoku": "sudoku.py", "sat": "sat3.py"}[task])
        template_sha = hashlib.sha256(formatter.get_prompt().encode()).hexdigest()
        frozen = {"model": model, "provider": payload["provider"], "payload": payload,
                  "template_sha256": template_sha, "problem_sha256": problem_sha,
                  "grader_sha256": source_hashes[grader], "repeat_index": 1,
                  "parser_sha256": source_hashes["original/openrouter.py"],
                  "reference_job_id": job_id, "profile_sha256": digest(profile)}
        jobs.append({"job_id": job_id, "task": task, "variant": "005_output_contract",
                     "model": model, "case_id": request["case_id"], "scale": question["scale"], "problem": problem,
                     "problem_sha256": problem_sha, "payload": payload,
                     "payload_sha256": payload_sha, "call_settings": profile["call_settings"],
                     "model_profile": profile,
                     "expected_provider_names": profile["expected_provider_names"],
                     "validator": original.validators[task], "maze_dimensions": dimensions or None,
                     "maze_diagnose": original.maze_diagnose if task == "maze" else None,
                     "frozen_job": frozen, "context_admission": admission})
    counts = Counter(job["task"] for job in jobs)
    if counts != {"sudoku": 400, "maze": 400, "sat": 2400}:
        raise ValueError("Frozen task counts differ from the formal cohort")
    files = ("questions.jsonl.gz", "requests.jsonl.gz", "final_answers.jsonl.gz", "profiles.json")
    manifest = {"adapter_version": 1, "reference_cohort_trials": len(jobs),
                "data_sha256": {name: original.transport.file_hash(data_dir / name) for name in files},
                "source_sha256": source_hashes,
                "adapter_sha256": {name: original.transport.file_hash(ROOT / name)
                                   for name in ("run.py", "frozen_code.py")},
                "schedule_seed": 42, "socket_timeout_seconds": SOCKET_TIMEOUT_SECONDS,
                "retry_policy": original.retry.RetryPolicy(None, BACKOFF_SECONDS).as_dict()}
    return jobs, manifest


def select_jobs(jobs, *, model=None, task=None, job_ids=None, limit=None):
    requested = set(job_ids or [])
    known = {job["job_id"] for job in jobs}
    if requested - known:
        raise ValueError("Unknown --job-id")
    selected = [job for job in jobs if (model is None or job["model"] == model)
                and (task is None or job["task"] == task)
                and (not requested or job["job_id"] in requested)]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("Filters selected no frozen requests")
    random.Random(42).shuffle(selected)
    return selected


def execution_manifest(jobs, manifest, workers):
    if type(workers) is not int or workers < 1 or workers > 300:
        raise ValueError("--workers must be between 1 and the original global limit 300")
    return {**manifest, "workers": workers,
            "group_semantics": manifest.get("group_semantics", "New independent calls; job IDs reference the archived seed001 trials"),
            "schedule": [{key: job[key] for key in ("job_id", "task", "model", "case_id", "scale", "payload_sha256")}
                         for job in jobs]}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


@contextmanager
def execution_lock(output):
    with (output / "execution.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another coordinator is using this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def setting_results(jobs, selected):
    """Descriptive success and timing for the selected, explicitly launched scope."""
    grouped = {}
    for job in jobs:
        key = (job["task"], job["model"], job["scale"])
        grouped.setdefault(key, []).append(job)
    rows = []
    for (task, model, scale), group in sorted(grouped.items()):
        records = [selected[job["job_id"]] for job in group if job["job_id"] in selected]
        elapsed = [record["elapsed_seconds"] / 60 for record in records
                   if type(record.get("elapsed_seconds")) in (int, float)
                   and math.isfinite(record["elapsed_seconds"]) and record["elapsed_seconds"] >= 0]
        correct = sum(record.get("correct") is True for record in records)
        complete = len(records) == len(group)
        statuses = Counter(record["status"] for record in records)
        rows.append({"task": task, "model": model, "scale": scale,
                     "planned_trials": len(group), "finalized_trials": len(records),
                     "unresolved_trials": len(group) - len(records), "complete": complete,
                     "n_unique_problems": len({job["problem_sha256"] for job in group}),
                     "n_correct": correct, "accuracy": correct / len(group) if complete else None,
                     "completed_trials": statuses["completed"],
                     "length_limited_trials": statuses["length_limited"],
                     "no_final_content_trials": statuses["no_final_content"],
                     "content_filtered_trials": statuses["content_filtered"],
                     "selected_duration_count": len(elapsed),
                     "mean_minutes": statistics.mean(elapsed) if elapsed else None,
                     "sample_sd_minutes": statistics.stdev(elapsed) if len(elapsed) >= 2 else None})
    return rows


def execute(jobs, manifest, output_dir, *, workers=300, resume=False, max_attempts=None):
    """Execute only when called by the explicit CLI branch; all attempts are retained."""
    original = frozen_code.load_original()
    delays = BACKOFF_SECONDS if max_attempts is None else tuple(BACKOFF_SECONDS[min(i, len(BACKOFF_SECONDS)-1)] for i in range(max_attempts-1))
    policy = original.retry.RetryPolicy(max_attempts, delays)
    frozen = execution_manifest(jobs, manifest, workers)
    frozen["retry_policy"] = policy.as_dict()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required for --execute")
    output = Path(output_dir).expanduser().resolve()
    protected = [ROOT / name for name in ("frozen_sources", "frozen_data", "provenance", "prompts", "verification")]
    if (output == ROOT or output in ROOT.parents
            or any(output == path or path in output.parents for path in protected)):
        raise ValueError("Choose an output directory outside protected release files and their ancestors")
    if resume:
        if not (output / "execution_plan.json").is_file():
            raise ValueError("--resume requires an existing execution plan")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with execution_lock(output):
        plan_path = output / "execution_plan.json"
        if resume:
            if json.loads(plan_path.read_text()) != frozen:
                raise ValueError("Resume parameters or frozen sources differ from the original launch")
        else:
            original.transport.write_new_json(plan_path, frozen)
        pause = threading.Event()
        pause_reasons = []
        lock = threading.Lock()
        previous_handlers = {}

        def stop(reason):
            with lock:
                pause_reasons.append(reason)
                pause.set()

        def should_pause():
            if not pause.is_set() and (output / "STOP").exists():
                stop({"reason": "STOP_file"})
            return pause.is_set()

        def handle_signal(number, frame):
            # Do not acquire the coordinator lock from an asynchronous handler.
            pause_reasons.append({"reason": "signal", "signal": number})
            pause.set()

        def attempt(job, attempt_dir):
            result = original.transport.run_one({**job, "output_dir": str(attempt_dir)},
                                                original.client, key, SOCKET_TIMEOUT_SECONDS)
            classification = original.retry.classify_attempt(result)
            if classification["kind"] in {"quota_pause", "blocked_engineering"}:
                stop({"job_id": job["job_id"], **classification})
            return result


        def run_case(job):
            try:
                result = original.retry.run_case(
                    job, output / "cases" / job["job_id"], attempt, policy=policy,
                    frozen_job=job["frozen_job"], should_pause=should_pause,
                    resume_quota=resume, recover_interrupted=resume)
                if not result.get("finalized") and not pause.is_set():
                    stop({"job_id": job["job_id"], "reason": result.get("status")})
                return result
            except Exception:
                stop({"job_id": job["job_id"], "reason": "case_execution_exception"})
                raise

        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[number] = signal.signal(number, handle_signal)
        results = []
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_case, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"finalized": False, "status": "case_execution_exception",
                                  "exception_type": type(exc).__name__}
                    results.append((job, result))
                    progress = {"observed": len(results), "planned": len(jobs),
                                "finalized": sum(bool(value.get("finalized")) for _, value in results),
                                "paused": pause.is_set()}
                    atomic_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
        finalized = sum(bool(result.get("finalized")) for _, result in results)
        selected_records = {
            job["job_id"]: json.loads(Path(result["selected_result_path"]).read_text())
            for job, result in results if result.get("finalized")}
        settings = setting_results(jobs, selected_records)
        summary = {"status": "completed" if finalized == len(jobs) else "paused_or_needs_review",
                   "planned": len(jobs), "finalized": finalized, "unresolved": len(jobs) - finalized,
                   "paused": pause.is_set(), "pause_reasons": pause_reasons,
                   "settings": settings,
                   "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "results": [{"job_id": job["job_id"], "finalized": result.get("finalized"),
                                "status": result.get("status"),
                                "selected_result_path": result.get("selected_result_path"),
                                "accounting": result.get("accounting")}
                               for job, result in sorted(results, key=lambda item: item[0]["job_id"])]}
        atomic_json(output / "summary.json", summary)
        with (output / "settings.csv.tmp").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(settings[0]))
            writer.writeheader()
            writer.writerows(settings)
        os.replace(output / "settings.csv.tmp", output / "settings.csv")
        with (output / "final_answers.jsonl.tmp").open("w") as handle:
            for job, result in sorted(results, key=lambda item: item[0]["job_id"]):
                if not result.get("finalized"):
                    continue
                selected = selected_records[job["job_id"]]
                fields = ("task", "model", "case_id", "problem_sha256", "payload_sha256", "final_content",
                          "parsed_answer", "correct", "strict_output_format", "status", "usage", "elapsed_seconds")
                row = {name: selected.get(name) for name in fields}
                row.update(job_id=job["job_id"], scale=job["scale"], selected_attempt=result["selected_attempt"],
                           selected_result_path=result["selected_result_path"])
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(output / "final_answers.jsonl.tmp", output / "final_answers.jsonl")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Make paid API calls for the selected frozen requests")
    mode.add_argument("--dry-run", action="store_true", help="Verify requests offline (the default)")
    parser.add_argument("--output-dir", type=Path, help="New directory for an independent call group")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume the same output directory and frozen plan")
    parser.add_argument("--workers", type=int, default=300)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--task", choices=("sudoku", "maze", "sat"))
    parser.add_argument("--job-id", action="append", help="Exact archived job ID; may be repeated")
    parser.add_argument("--limit", type=int, help="Take the first N filtered job IDs before deterministic shuffling")
    args = parser.parse_args(argv)
    if args.execute and args.output_dir is None:
        parser.error("--execute requires --output-dir")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    jobs, manifest = prepare()
    jobs = select_jobs(jobs, model=args.model, task=args.task, job_ids=args.job_id, limit=args.limit)
    execution_manifest(jobs, manifest, args.workers)
    if not args.execute:
        print(json.dumps({"dry_run": True, "network_calls": 0, "reference_cohort_trials": 3200,
                          "verified_payloads": 3200, "selected_requests": len(jobs),
                          "tasks": dict(Counter(job["task"] for job in jobs)),
                          "models": sorted({job["model"] for job in jobs}),
                          "workers": args.workers, "all_context_admission_passed": True,
                          "source_hashes": manifest["source_sha256"]}, indent=2))
        return 0
    summary = execute(jobs, manifest, args.output_dir, workers=args.workers, resume=args.resume)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)
    return 0 if summary["unresolved"] == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Credentials and provider responses never appear in exception messages.
        print(json.dumps({"fatal_error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1)
