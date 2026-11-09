#!/usr/bin/env python3
"""Export the selected 3,200-trial archive into a new, empty release directory.

Maintenance only: requires the source archive and the raw request/problem paths
recorded in it. Normal release execution never needs these historical paths.
No network calls are made and no environment or credential files are read.
"""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


MODELS = {
    "anthropic/claude-opus-4.5", "deepseek/deepseek-v3.2",
    "google/gemini-3.1-pro-preview", "openai/gpt-5.2",
}
REQUEST_KEYS = {
    "model", "messages", "stream", "max_completion_tokens",
    "reasoning", "provider", "debug",
}
SECRET_KEYS = {
    "authorization", "api_key", "apikey", "api-key", "access_token",
    "password", "client_secret", "api_secret",
}
COPY_DATA = (
    "questions.jsonl.gz", "final_answers.jsonl.gz", "current_settings.csv",
    "selected_cases.csv",
)
COPY_PROVENANCE = {
    "source_artifact_index.jsonl.gz": "source_artifact_index.jsonl.gz",
    "source_manifest.json": "archive_source_manifest.json",
    "package_manifest.json": "archive_package_manifest.json",
    "sources/input_source_manifest.json": "input_source_manifest.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def rows_by_job(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    indexed = {row["job_id"]: row for row in rows}
    require(len(rows) == len(indexed), f"Duplicate job_id in {path.name}")
    return indexed


def check_no_secrets(value, context):
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.lower() not in SECRET_KEYS,
                    f"Credential field is not exportable: {context}/{key}")
            check_no_secrets(child, f"{context}/{key}")
    elif isinstance(value, list):
        for child in value:
            check_no_secrets(child, context)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export(source, output):
    source = source.resolve(strict=True)
    output = output.resolve()
    require(source.is_dir(), "--source-archive must be a directory")
    require(output != source and source not in output.parents,
            "Output must be outside the source archive")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Refusing nonempty output; frozen release data must never be overwritten")

    questions = rows_by_job(source / "questions.jsonl.gz")
    answers = rows_by_job(source / "final_answers.jsonl.gz")
    artifacts = rows_by_job(source / "source_artifact_index.jsonl.gz")
    require(questions.keys() == answers.keys() == artifacts.keys(),
            "Question, answer, and artifact job sets differ")
    require(len(answers) == 3200, "This release requires exactly 3,200 selected trials")
    counts = collections.Counter(row["model"] for row in answers.values())
    require(set(counts) == MODELS and set(counts.values()) == {800},
            "This release requires exactly 800 trials per model")
    with (source / "selected_cases.csv").open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    require(len(cases) == len(answers) and {row["job_id"] for row in cases} == answers.keys(),
            "Selected-case job set differs from the frozen answers")

    snapshots = {}
    profile_documents = {}
    profiles = {}
    profile_sources = {}
    requests = []
    source_experiments = collections.Counter()
    for job_id, answer in answers.items():
        question, artifact = questions[job_id], artifacts[job_id]
        problem_hash = sha256(canonical(question["problem"]))
        require(problem_hash == question["problem_sha256"] == answer["problem_sha256"]
                == artifact["problem_sha256"], f"Problem hash mismatch: {job_id}")
        require(question["model"] == answer["model"] and question["task"] == answer["task"],
                f"Question/answer identity mismatch: {job_id}")
        expected_problem_keys = {"clauses", "n"} if answer["task"] == "sat" else {"question"}
        require(set(question["problem"]) == expected_problem_keys,
                f"Unexpected problem fields (references are never inputs): {job_id}")

        request_path = Path(answer["selected_request_path"])
        require(request_path.name == "request.json", f"Unexpected request filename: {job_id}")
        require(str(request_path) == artifact["selected_request_path"],
                f"Request path mismatch: {job_id}")
        raw = request_path.read_bytes()
        require(sha256(raw) == answer["selected_request_sha256"]
                == artifact["selected_request_sha256"], f"Request file hash mismatch: {job_id}")
        payload = json.loads(raw)
        require(set(payload) == REQUEST_KEYS, f"Unexpected request keys: {job_id}")
        check_no_secrets(payload, job_id)
        require(payload["model"] == answer["model"] and payload["stream"] is True,
                f"Request model/stream mismatch: {job_id}")
        require(len(payload["messages"]) == 1
                and set(payload["messages"][0]) == {"role", "content"}
                and payload["messages"][0]["role"] == "user",
                f"Unexpected message contract: {job_id}")
        require(sha256(canonical(payload)) == answer["payload_sha256"],
                f"Payload hash mismatch: {job_id}")

        problem_path = Path(artifact["selected_problem_path"])
        require(problem_path.name == "problem.json", f"Unexpected problem filename: {job_id}")
        raw_problem = problem_path.read_bytes()
        require(sha256(raw_problem) == artifact["selected_problem_file_sha256"],
                f"Problem file hash mismatch: {job_id}")
        require(json.loads(raw_problem) == question["problem"],
                f"Raw/frozen problem mismatch: {job_id}")

        experiment = next((parent for parent in request_path.parents
                           if parent.parent.name == "llm-solver-experiments"), None)
        require(experiment is not None, f"Cannot identify request experiment: {job_id}")
        require(experiment.name[:3] in {"009", "010", "011", "012", "015"},
                f"Unexpected source experiment: {experiment.name}")
        source_experiments[experiment.name] += 1
        if experiment not in profile_documents:
            for name in ("MODEL_PROFILES.json", "MODEL_SETTINGS.json", "FULL_PLAN.json",
                         "PROVIDER_CHANGES.json"):
                path = experiment / name
                if name == "MODEL_PROFILES.json" or path.is_file():
                    data = path.read_bytes()
                    document = json.loads(data)
                    check_no_secrets(document, f"{experiment.name}/{name}")
                    snapshots[path] = data
                    if name == "MODEL_PROFILES.json":
                        profile_documents[experiment] = document
        profile = profile_documents[experiment]["models"][answer["model"]]
        settings = profile["call_settings"]
        require(settings == answer["request_call_settings"]
                and settings == {key: payload[key] for key in settings},
                f"Original profile differs from request settings: {job_id}")
        profile_key = f"{experiment.name}/{answer['model']}"
        profiles[profile_key] = profile
        profile_sources[profile_key] = {
            "source_path": str(experiment / "MODEL_PROFILES.json"),
            "source_sha256": sha256(snapshots[experiment / "MODEL_PROFILES.json"]),
            "json_pointer": "/models/" + answer["model"].replace("~", "~0").replace("/", "~1"),
        }
        requests.append({
            "job_id": job_id, "task": answer["task"], "model": answer["model"],
            "case_id": answer["case_id"], "problem_sha256": problem_hash,
            "payload_sha256": answer["payload_sha256"], "payload": payload,
            "profile_key": profile_key,
        })

    # Only the named artifacts above are read. In particular, events, complete
    # reasoning traces, reference-answer files, and .env files are not opened.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".llm-freeze-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "frozen_data").mkdir()
        (stage / "provenance").mkdir()
        copied = []

        def copy_bytes(path, relative, data=None):
            data = path.read_bytes() if data is None else data
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            copied.append({"source_path": str(path), "source_sha256": sha256(data),
                           "release_path": relative, "bytes": len(data),
                           "encoding": "byte-identical"})

        for name in COPY_DATA:
            copy_bytes(source / name, f"frozen_data/{name}")
        for name, target in COPY_PROVENANCE.items():
            copy_bytes(source / name, f"provenance/{target}")
        for path, data in sorted(snapshots.items()):
            copy_bytes(path, f"provenance/model_config/{path.parent.name}/{path.name}", data)
        with (stage / "frozen_data/requests.jsonl.gz").open("wb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as compressed:
                for row in requests:
                    compressed.write(canonical(row) + b"\n")
        write_json(stage / "frozen_data/profiles.json", profiles)
        manifest = {
            "schema_version": 1, "source_archive_path": str(source),
            "scope": "3200 selected trials; four historical models; 005 output contract",
            "trials": len(requests), "model_counts": dict(sorted(counts.items())),
            "source_experiment_counts": dict(sorted(source_experiments.items())),
            "profile_sources": profile_sources, "copied_files": copied,
            "verification": {
                "request_file_sha256": len(requests), "payload_sha256": len(requests),
                "problem_file_sha256": len(requests), "problem_sha256": len(requests),
                "original_profile_call_settings": len(requests),
                "request_allowed_keys": sorted(REQUEST_KEYS),
                "reference_answers_used_as_input": False, "network_calls": 0,
                "raw_events_read_or_copied": False,
                "raw_event_hash_policy": "Preserved from original independent audit; not rehashed",
            },
            "files": [{"path": str(path.relative_to(stage)),
                       "sha256": sha256(path.read_bytes()), "bytes": path.stat().st_size}
                      for path in sorted(stage.rglob("*")) if path.is_file()],
            "excluded_from_files": ["provenance/freeze_manifest.json"],
        }
        write_json(stage / "provenance/freeze_manifest.json", manifest)
        output.mkdir(exist_ok=True)
        require(not any(output.iterdir()), "Output became nonempty during validation; refusing overwrite")
        for name in ("frozen_data", "provenance"):
            os.rename(stage / name, output / name)
    return {"output": str(output), "trials": len(requests), "profiles": len(profiles),
            "request_and_problem_hashes_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty directory; populated releases are never overwritten")
    args = parser.parse_args()
    try:
        result = export(args.source_archive, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"freeze_data: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
