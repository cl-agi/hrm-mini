"""Durable retries for transport failures, with bounded or continuing policies.

No network implementation is included. An injected callback executes one attempt
and returns its full result, optionally writing its own result.json. Answers and
grades never determine retry eligibility. Explicit policy and job identity are
frozen before the first attempt.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid


NORMAL_STATUSES = {"completed", "length_limited", "content_filtered", "no_final_content"}
TRANSIENT_HTTP = {408, 429, *range(500, 600)}
SSE_ERROR_MARKERS = {"finish_reason_error"}
MALFORMED_CODES = {"invalid_sse_json", "invalid_sse_object", "invalid_choice",
                   "unexpected_multiple_choices", "invalid_delta"}
STOP_REASONS = {"stop"}
NATIVE_STOP_REASONS = {"stop", "stop_sequence", "end_turn", "completed"}
MODEL_TERMINAL_REASONS = STOP_REASONS | {"length", "content_filter"}
NATIVE_TERMINAL_REASONS = NATIVE_STOP_REASONS | {"length", "max_output_tokens", "max_tokens", "content_filter"}
IDENTITY_KEYS = {"model", "provider", "payload", "template_sha256", "problem_sha256",
                 "grader_sha256", "repeat_index"}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int | None
    backoff_seconds: tuple

    def __post_init__(self):
        if self.max_attempts is not None and (type(self.max_attempts) is not int or self.max_attempts < 1):
            raise ValueError("max_attempts must be a positive integer or explicit null")
        delays = tuple(self.backoff_seconds)
        if self.max_attempts is not None and len(delays) != self.max_attempts - 1:
            raise ValueError("Supply one explicit backoff for each possible retry")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in delays):
            raise ValueError("Backoff durations must be finite nonnegative numbers")
        if self.max_attempts is None and (not delays or any(x <= 0 for x in delays)):
            raise ValueError("Continuing retries require a nonempty positive backoff schedule")
        object.__setattr__(self, "backoff_seconds", delays)

    def can_retry(self, completed_attempt):
        return self.max_attempts is None or completed_attempt < self.max_attempts

    def backoff_for(self, completed_attempt):
        if type(completed_attempt) is not int or completed_attempt < 1:
            raise ValueError("Retry backoff requires a positive attempt number")
        if not self.can_retry(completed_attempt):
            return None
        return self.backoff_seconds[min(completed_attempt - 1, len(self.backoff_seconds) - 1)]

    def as_dict(self):
        result = {"max_attempts": self.max_attempts, "backoff_seconds": list(self.backoff_seconds),
                  "retry_after_rule": "max(configured_backoff, valid_retry_after)",
                  "selection_rule": "first_model_terminal_in_attempt_order"}
        if self.max_attempts is None:
            result["backoff_rule"] = "repeat_last_configured_delay"
        return result


def _code(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return None


def _unknown_error_model_is_missing(result):
    config = result.get("configuration_check") or {}
    providers, expected = result.get("observed_providers"), config.get("expected_provider_names")
    models, allowed = result.get("observed_models"), config.get("expected_response_models")
    issues = config.get("issues")
    errors = result.get("error_codes")
    return (result.get("status") == "sse_error" and _code(result.get("http_status")) == 200 and
            isinstance(models, list) and "unknown" in models and isinstance(allowed, list) and
            all(model == "unknown" or model in allowed for model in models) and result.get("response_model") in models and
            isinstance(providers, list) and bool(providers) and isinstance(expected, list) and
            all(provider in expected for provider in providers) and
            result.get("model_output_present") is False and result.get("final_content") == "" and
            not result.get("reasoning") and not any(result.get("reasoning_channels", {}).values()) and
            isinstance(errors, list) and bool(errors) and any(_code(value) in TRANSIENT_HTTP for value in errors) and
            all(_code(value) in TRANSIENT_HTTP or value == "finish_reason_error" for value in errors) and
            all(isinstance(result.get(key), list) and all(value == "error" for value in result[key])
                for key in ("finish_reasons", "native_finish_reasons")) and
            isinstance(issues, list) and "response_model_missing" in issues and
            all(isinstance(issue, str) and issue.endswith("_missing") for issue in issues))


def classify_attempt(result):
    """Classify explicit transport evidence; correctness is deliberately unused."""
    def decision(kind, reason):
        return {"kind": kind, "reason": reason}

    if not isinstance(result, dict):
        return decision("blocked_engineering", "invalid_result_object")
    terminal_seen = any(isinstance(value, str) and value in reasons
                        for field, reasons in (("finish_reasons", MODEL_TERMINAL_REASONS),
                                               ("native_finish_reasons", NATIVE_TERMINAL_REASONS))
                        for value in (result.get(field) if isinstance(result.get(field), list) else []))
    if terminal_seen and (result.get("status") not in NORMAL_STATUSES or result.get("error_codes")
                          or (_code(result.get("http_status")) or 0) >= 400):
        # Preserve a terminal sample even when contradictory tail evidence also
        # reports a quota/auth/service error. Such records require review.
        return decision("blocked_engineering", "terminal_evidence_with_unresolved_transport_status")
    # A 402 pauses the queue regardless of missing streaming/debug metadata.
    if _code(result.get("http_status")) == 402:
        return decision("quota_pause", "quota_exhausted_pause_required")
    status = result.get("status")
    errors = result.get("error_codes") or []
    finishes = result.get("finish_reasons") or []
    native = result.get("native_finish_reasons") or []
    if not all(isinstance(v, list) for v in (errors, finishes, native)):
        return decision("blocked_engineering", "invalid_result_evidence_fields")
    if any(not isinstance(v, (str, int)) or isinstance(v, bool) for v in errors):
        return decision("blocked_engineering", "unrecognized_error_code")
    if any(not isinstance(v, str) for v in finishes + native):
        return decision("blocked_engineering", "invalid_finish_reason")
    codes = {_code(v) for v in errors if _code(v) is not None}
    http = _code(result.get("http_status"))
    if 402 in codes:
        return decision("quota_pause", "quota_exhausted_pause_required")
    if any(code in {400, 401, 403, 404, 405, 422} for code in codes | {http}):
        return decision("blocked_engineering", "http_auth_quota_or_request_error")
    if MALFORMED_CODES.intersection(errors):
        return decision("blocked_engineering", "malformed_stream_requires_investigation")
    if ("unknown" in (result.get("observed_models") or []) or result.get("response_model") == "unknown") and not _unknown_error_model_is_missing(result):
        return decision("blocked_engineering", "configuration_audit_failed")
    config = result.get("configuration_check")
    config_failed = isinstance(config, dict) and config.get("status") == "issues"
    if config_failed:
        missing_issues = {"response_provider_missing", "response_model_missing", "debug_provider_missing", "upstream_debug_missing",
                          "upstream_output_limit_missing", "upstream_reasoning_effort_missing"}
        config_issues = config.get("issues")
        missing_only = isinstance(config_issues, list) and bool(config_issues)
        for issue in config_issues if isinstance(config_issues, list) else []:
            if issue == "debug_provider_missing_or_mismatch":
                expected = config.get("expected_provider_names") or []
                events = result.get("upstream_debug") or []
                if any(event.get("provider") and event["provider"] not in expected for event in events):
                    missing_only = False
            elif not (isinstance(issue, str) and (issue in missing_issues or
                      (issue.startswith("upstream_") and issue.endswith("_missing")))):
                missing_only = False
        if not missing_only:
            return decision("blocked_engineering", "configuration_audit_failed")
    if status in {"network_error", "socket_timeout"}:
        if errors or "error" in finishes:
            return decision("blocked_engineering", "mixed_network_and_unclassified_error_evidence")
        return decision("retryable_transport", status)
    if status == "http_error":
        if http in TRANSIENT_HTTP and all(code in TRANSIENT_HTTP for code in codes) and all(_code(v) is not None for v in errors):
            return decision("retryable_transport", f"transient_http_{http}")
        return decision("blocked_engineering", "unclassified_http_error")
    if status == "sse_error":
        unknown = [v for v in errors if _code(v) is None and v not in SSE_ERROR_MARKERS]
        if codes and all(code in TRANSIENT_HTTP for code in codes) and not unknown:
            return decision("retryable_transport", "explicit_transient_sse_error")
        return decision("blocked_engineering", "unclassified_sse_error")
    if status not in NORMAL_STATUSES:
        return decision("blocked_engineering", "stream_or_client_execution_unresolved")
    if config_failed:
        return decision("blocked_engineering", "terminal_configuration_audit_incomplete")
    if errors or "error" in finishes or (http is not None and http >= 400):
        return decision("blocked_engineering", "terminal_status_with_engineering_error")
    if status == "length_limited":
        terminal = "length" in finishes or bool({"length", "max_output_tokens", "max_tokens"}.intersection(native))
    elif status == "content_filtered":
        terminal = "content_filter" in finishes or "content_filter" in native
    else:
        terminal = bool(STOP_REASONS.intersection(finishes) or NATIVE_STOP_REASONS.intersection(native))
    if not terminal:
        return decision("blocked_engineering", "explicit_model_terminal_evidence_missing")
    return decision("model_terminal", status)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path, value):
    # The case lock serializes writers; x mode also rejects accidental replacement.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_summary(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    _write_new(temporary, value)
    os.replace(temporary, path)


def _retry_after(result, now):
    value = result.get("retry_after_seconds", result.get("retry_after"))
    if value is None and isinstance(result.get("response_headers"), dict):
        value = next((v for k, v in result["response_headers"].items()
                      if str(k).lower() == "retry-after"), None)
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            seconds = max(0.0, target.timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def executor_identity(pid=None):
    """Identify a Linux process without mistaking a reused PID for its owner."""
    pid = os.getpid() if pid is None else pid
    if type(pid) is not int or pid < 1:
        raise ValueError("Executor PID must be a positive integer")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "state": fields[0]}


def _valid_executor(identity):
    return (isinstance(identity, dict) and type(identity.get("pid")) is int and identity["pid"] > 0
            and isinstance(identity.get("start_ticks"), str) and identity["start_ticks"].isdigit()
            and isinstance(identity.get("boot_id"), str) and bool(identity["boot_id"]))


def _executor_exited(prior, observed, checked_boot_id):
    if not _valid_executor(prior) or not isinstance(checked_boot_id, str) or not checked_boot_id:
        raise ValueError("Interrupted claim lacks verifiable executor identity")
    if prior["boot_id"] != checked_boot_id or observed is None:
        return True
    if not _valid_executor(observed) or observed["pid"] != prior["pid"] or observed["boot_id"] != checked_boot_id:
        raise ValueError("Invalid observed executor identity")
    return observed["start_ticks"] != prior["start_ticks"] or observed.get("state") in {"Z", "X"}


def _interrupted_sources(attempt_dir):
    sources = {}
    for path in sorted(attempt_dir.iterdir()):
        if path.name == "interruption_recovery.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("Interrupted attempt contains an unsupported file type")
        if path.name in {"result.json", "attempt_outcome.json", "execution_job_mutated.json", "callback_result_mismatch.json"}:
            raise ValueError("Interrupted attempt has result or conflicting execution evidence")
        sources[path.name] = {"sha256": _sha_file(path), "bytes": path.stat().st_size}
    if "attempt_claim.json" not in sources:
        raise ValueError("Interrupted attempt has no claim")
    return sources


def _interrupted_events(attempt_dir, job=None):
    """Read preserved evidence; ambiguous or terminal streams cannot be resampled."""
    events_path = attempt_dir / "events.jsonl"
    count, observed_usage = 0, {}
    profile = (job or {}).get("model_profile") or {}
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or event.get("done") or "unparsed_event_sha256" in event:
                raise ValueError("Interrupted stream has DONE or ambiguous event evidence")
            count += 1
            if "chunk" not in event:
                if "http_error_status" in event:
                    if _code(event["http_error_status"]) not in TRANSIENT_HTTP:
                        raise ValueError("Interrupted stream contains a nontransient HTTP error")
                elif "transport_error" in event:
                    error = event["transport_error"]
                    if (not isinstance(error, dict) or error.get("status") not in {"network_error", "socket_timeout"}
                            or event.get("after_terminal") is not False):
                        raise ValueError("Interrupted stream contains an unclassified transport error")
                else:
                    raise ValueError("Interrupted stream contains an unknown event")
                continue
            chunk = event["chunk"]
            if not isinstance(chunk, dict):
                raise ValueError("Interrupted stream contains an invalid chunk")
            for field, expected_field in (("provider", "expected_provider_names"), ("model", "expected_response_models")):
                expected = profile.get(expected_field)
                if expected and chunk.get(field) and chunk[field] not in expected:
                    raise ValueError("Interrupted stream configuration identity mismatch")
            if isinstance(chunk.get("usage"), dict):
                observed_usage.update(chunk["usage"])
            containers = [chunk]
            choices = chunk.get("choices") or []
            if not isinstance(choices, list):
                raise ValueError("Interrupted stream contains invalid choices")
            error_finish = False
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    raise ValueError("Interrupted stream contains invalid choice")
                finish, native = choice.get("finish_reason"), choice.get("native_finish_reason")
                if finish not in (None, "error") or native not in (None, "error"):
                    raise ValueError("Interrupted stream already contains model terminal evidence")
                error_finish = error_finish or finish == "error" or native == "error"
                delta = choice.get("delta") or choice.get("message") or {}
                if not isinstance(delta, dict):
                    raise ValueError("Interrupted stream contains invalid delta")
                containers.extend((choice, delta))
            transient_error = False
            for container in containers:
                error = container.get("error")
                if error and (not isinstance(error, dict) or _code(error.get("code")) not in TRANSIENT_HTTP):
                    raise ValueError("Interrupted stream contains unclassified error evidence")
                transient_error = transient_error or bool(error)
            if error_finish and not transient_error:
                raise ValueError("Interrupted stream has unclassified error finish")
            debug = chunk.get("debug")
            if isinstance(debug, dict):
                expected = profile.get("expected_provider_names")
                if expected and debug.get("provider") and debug["provider"] not in expected:
                    raise ValueError("Interrupted upstream provider mismatch")
                parameters = debug.get("upstream_parameters") or {}
                for requirement in profile.get("upstream_requirements", []):
                    for path in requirement["paths"]:
                        value = parameters
                        for field in path.split("."):
                            value = value.get(field) if isinstance(value, dict) else None
                        if value is not None and value not in requirement["accepted_values"]:
                            raise ValueError("Interrupted upstream parameter mismatch")
    return {"event_count": count, "terminal_seen": False, "done_seen": False}, observed_usage


def interruption_recovery_record(attempt_dir, index, job_sha256, policy, job=None):
    """Verify an append-only recovery record against the original attempt files."""
    attempt_dir = Path(attempt_dir)
    path = attempt_dir / "interruption_recovery.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        claim = json.loads((attempt_dir / "attempt_claim.json").read_text())
        timestamp = record["recovered_at_epoch"]
        if (record.get("kind") != "explicit_interrupted_attempt_recovery" or record.get("attempt") != index
                or record.get("job_sha256") != job_sha256 or claim.get("attempt") != index
                or claim.get("job_sha256") != job_sha256 or record.get("executor") != claim.get("executor")
                or type(timestamp) not in (float, int) or not math.isfinite(timestamp)
                or timestamp < claim["claimed_epoch"]
                or record.get("recovered_at_utc") != datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                or record.get("source_files") != _interrupted_sources(attempt_dir)):
            raise ValueError("Interrupted recovery identity or evidence mismatch")
        check = record["executor_exit_check"]
        if (check.get("checked_epoch") != timestamp or check.get("prior_executor_exited") is not True
                or not _executor_exited(record["executor"], check.get("observed_executor"), check["checked_boot_id"])):
            raise ValueError("Interrupted recovery executor exit was not established")
        event_check, usage = _interrupted_events(attempt_dir, job)
        if (record.get("events_check") != event_check or record.get("observed_partial_usage") != usage
                or record.get("usage") != {} or record.get("cost_status") != "unknown_interrupted_attempt"):
            raise ValueError("Interrupted recovery stream evidence mismatch")
        expected_delay = policy.backoff_for(index)
        expected_deadline = timestamp + expected_delay if expected_delay is not None else None
        if (record.get("configured_backoff_seconds") != expected_delay
                or record.get("next_attempt_not_before_epoch") != expected_deadline):
            raise ValueError("Interrupted recovery retry delay mismatch")
        return record
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid interrupted recovery record") from error


def _recover_interrupted(attempt_dir, index, job_sha256, policy, job, now_fn):
    sources = _interrupted_sources(attempt_dir)
    claim = json.loads((attempt_dir / "attempt_claim.json").read_text())
    prior = claim.get("executor")
    if claim.get("attempt") != index or claim.get("job_sha256") != job_sha256 or not _valid_executor(prior):
        raise ValueError("Interrupted claim identity is not verifiable")
    observed = executor_identity(prior["pid"])
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not _executor_exited(prior, observed, boot_id):
        raise ValueError("Interrupted attempt executor is still alive")
    request_path = attempt_dir / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != job["payload"]:
        raise ValueError("Interrupted request differs from frozen payload")
    problem_path = attempt_dir / "problem.json"
    if problem_path.exists() and json.loads(problem_path.read_text()) != job.get("problem"):
        raise ValueError("Interrupted problem differs from frozen problem")
    event_check, usage = _interrupted_events(attempt_dir, job)
    now = now_fn()
    if type(claim.get("claimed_epoch")) not in (int, float) or now < claim["claimed_epoch"]:
        raise ValueError("Interrupted claim timestamp is invalid")
    delay = policy.backoff_for(index)
    record = {"kind": "explicit_interrupted_attempt_recovery", "attempt": index, "job_sha256": job_sha256,
              "executor": prior, "executor_exit_check": {"checked_epoch": now, "checked_boot_id": boot_id,
                  "observed_executor": observed, "prior_executor_exited": True},
              "source_files": sources, "events_check": event_check, "observed_partial_usage": usage,
              "usage": {}, "cost_status": "unknown_interrupted_attempt", "recovered_at_epoch": now,
              "recovered_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
              "configured_backoff_seconds": delay,
              "next_attempt_not_before_epoch": now + delay if delay is not None else None}
    _write_new(attempt_dir / "interruption_recovery.json", record)


def _outcome(attempt_dir, index, policy, now_fn, job_sha256=None, job=None):
    outcome_path, result_path = attempt_dir / "attempt_outcome.json", attempt_dir / "result.json"

    def unresolved(reason, usage=None):
        return {"attempt": index, "attempt_dir": str(attempt_dir), "status": "persisted_attempt_requires_review",
                "classification": {"kind": "blocked_engineering", "reason": reason},
                "usage": usage or {}, "result_path": str(result_path) if result_path.exists() else None}

    if (attempt_dir / "interruption_recovery.json").exists():
        try:
            record = interruption_recovery_record(attempt_dir, index, job_sha256, policy, job)
        except ValueError:
            return unresolved("invalid_interrupted_recovery_evidence")
        return {"attempt": index, "attempt_dir": str(attempt_dir), "status": "interrupted_recovered",
                "classification": {"kind": "retryable_transport", "reason": "executor_exited_before_result"},
                "usage": {}, "result_path": None, "interruption_recovery_sha256": _sha_file(attempt_dir / "interruption_recovery.json"),
                "configured_backoff_seconds": record["configured_backoff_seconds"],
                "next_attempt_not_before_epoch": record["next_attempt_not_before_epoch"]}
    if outcome_path.exists():
        try:
            outcome = json.loads(outcome_path.read_text())
            expected_sha = outcome["result_sha256"]
        except (ValueError, KeyError, TypeError):
            return unresolved("invalid_persisted_outcome")
        if not result_path.exists() or _sha_file(result_path) != expected_sha:
            return unresolved("persisted_result_changed", outcome.get("usage"))
        return outcome
    if not result_path.exists():
        return {"attempt": index, "attempt_dir": str(attempt_dir), "status": "unresolved_claim",
                "classification": {"kind": "blocked_engineering", "reason": "claimed_attempt_has_no_result"},
                "usage": {}, "result_path": None}
    try:
        result = json.loads(result_path.read_text())
    except ValueError:
        return unresolved("invalid_persisted_result")
    if not isinstance(result, dict):
        return unresolved("invalid_persisted_result_object")
    if (attempt_dir / "execution_job_mutated.json").exists():
        return unresolved("execution_job_mutated", result.get("usage"))
    if (attempt_dir / "callback_result_mismatch.json").exists():
        return unresolved("callback_result_mismatch", result.get("usage"))
    now = now_fn()
    outcome = {"attempt": index, "attempt_dir": str(attempt_dir), "status": result.get("status"),
               "classification": classify_attempt(result), "usage": result.get("usage") or {},
               "elapsed_seconds": result.get("elapsed_seconds"), "correct": result.get("correct"),
               "result_path": str(result_path), "result_sha256": _sha_file(result_path),
               "recorded_at_epoch": now}
    if policy.can_retry(index) and outcome["classification"]["kind"] == "retryable_transport":
        header_delay = _retry_after(result, now)
        delay = max(policy.backoff_for(index), header_delay or 0)
        outcome.update({"configured_backoff_seconds": policy.backoff_for(index),
                        "retry_after_seconds": header_delay, "next_attempt_not_before_epoch": now + delay})
    _write_new(outcome_path, outcome)
    return outcome


def _accounting(outcomes):
    totals = {}
    for key in ("cost", "prompt_tokens", "completion_tokens", "total_tokens"):
        values, missing = [], 0
        for outcome in outcomes:
            usage = outcome.get("usage")
            value = usage.get(key) if isinstance(usage, dict) else None
            try:
                number = Decimal(str(value))
                if not number.is_finite() or number < 0 or isinstance(value, bool):
                    raise InvalidOperation
                values.append(number)
            except InvalidOperation:
                missing += 1
        total = sum(values, Decimal(0))
        totals[key] = {"sum_reported": str(total) if key == "cost" else float(total),
                       "reported_attempts": len(values), "missing_attempts": missing,
                       "complete": missing == 0}
    elapsed = [o["elapsed_seconds"] for o in outcomes if type(o.get("elapsed_seconds")) in (int, float)]
    return {"attempt_count": len(outcomes), "additional_attempt_count": max(0, len(outcomes) - 1),
            "usage": totals, "sum_reported_attempt_elapsed_seconds": sum(elapsed),
            "missing_elapsed_attempts": len(outcomes) - len(elapsed)}


def _resume_path(case_dir, prior_attempt):
    return case_dir / "quota_resumes" / (
        f"quota_resume_after_attempt_{prior_attempt:03d}_before_attempt_{prior_attempt + 1:03d}.json")


def quota_resume_record(case_dir, outcome, job_sha256):
    """Read and verify the durable authorization that crossed a prior 402.

    None means no authorization exists; invalid evidence raises ValueError.
    The public audit helper validates the same append-only record as scheduling.
    """
    case_dir = Path(case_dir)
    prior = outcome["attempt"]
    path = _resume_path(case_dir, prior)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        expected = {"kind": "explicit_quota_resume", "prior_attempt": prior,
                    "next_attempt": prior + 1, "job_sha256": job_sha256,
                    "prior_result_sha256": outcome["result_sha256"],
                    "prior_outcome_sha256": _sha_file(Path(outcome["attempt_dir"]) / "attempt_outcome.json")}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Quota resume identity mismatch")
        if outcome["classification"]["kind"] != "quota_pause":
            raise ValueError("Quota resume requires a quota outcome")
        timestamp = record["resumed_at_epoch"]
        if type(timestamp) not in (float, int) or not math.isfinite(timestamp):
            raise ValueError("Invalid quota resume timestamp")
        if record["resumed_at_utc"] != datetime.fromtimestamp(timestamp, timezone.utc).isoformat():
            raise ValueError("Quota resume timestamp mismatch")
        claim_path = case_dir / "attempts" / f"attempt_{prior + 1:03d}" / "attempt_claim.json"
        claim = json.loads(claim_path.read_text())
        if claim.get("attempt") != prior + 1 or claim.get("job_sha256") != job_sha256:
            raise ValueError("Quota resume next claim mismatch")
        snapshot_name = record.get("prior_summary_snapshot_file")
        if snapshot_name is not None:
            if not isinstance(snapshot_name, str) or Path(snapshot_name).name != snapshot_name:
                raise ValueError("Invalid quota summary snapshot name")
            snapshot = path.parent / snapshot_name
            if _sha_file(snapshot) != record.get("prior_summary_snapshot_sha256"):
                raise ValueError("Quota summary snapshot changed")
        elif record.get("prior_summary_snapshot_sha256") is not None:
            raise ValueError("Unexpected quota summary snapshot hash")
        return record
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid quota resume record") from error


def _record_quota_resume(case_dir, outcome, job_sha256, now_fn):
    path = _resume_path(case_dir, outcome["attempt"])
    path.parent.mkdir(exist_ok=True)
    now = now_fn()
    record = {"kind": "explicit_quota_resume", "prior_attempt": outcome["attempt"],
              "next_attempt": outcome["attempt"] + 1, "job_sha256": job_sha256,
              "prior_result_sha256": outcome["result_sha256"],
              "prior_outcome_sha256": _sha_file(Path(outcome["attempt_dir"]) / "attempt_outcome.json"),
              "resumed_at_epoch": now, "resumed_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
              "prior_summary_snapshot_file": None, "prior_summary_snapshot_sha256": None}
    summary = case_dir / "case_summary.json"
    if summary.exists():
        snapshot = path.with_name(path.stem + "_prior_summary.json")
        with snapshot.open("xb") as handle:
            handle.write(summary.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        record.update(prior_summary_snapshot_file=snapshot.name,
                      prior_summary_snapshot_sha256=_sha_file(snapshot))
    _write_new(path, record)


def _attempt_directories(attempts_dir):
    directories = list(Path(attempts_dir).glob("attempt_*"))
    try:
        directories.sort(key=lambda path: int(path.name.removeprefix("attempt_")))
    except ValueError:
        raise ValueError("Invalid attempt directory name") from None
    if [path.name for path in directories] != [f"attempt_{index:03d}" for index in range(1, len(directories) + 1)]:
        raise ValueError("Attempt directory sequence has a gap")
    return directories


def run_case(job, case_dir, attempt_execute, *, policy, frozen_job,
             sleep_fn=time.sleep, now_fn=time.time, should_pause=None, resume_quota=False,
             recover_interrupted=False):
    """Run/resume one case; return a terminal selection or an explicit pending state.

    ``attempt_execute(job, attempt_dir)`` must perform at most one model call and
    return its full result. The caller controls provider concurrency around that
    callback. The callback may create result.json; all other scheduler names are
    reserved. An existing claimed directory is immutable. Explicit recovery of
    an interrupted claim archives its evidence and allocates the next attempt
    only after verifying that its identified executor has exited. A persisted
    terminal event, DONE, or ambiguous stream blocks recovery and resampling.
    ``should_pause()`` prevents each new claim while the shared queue is paused.
    ``resume_quota=True`` authorizes crossing a 402 already present on entry;
    a new 402 always returns pending_quota and requires another explicit resume.
    """
    if not isinstance(policy, RetryPolicy):
        raise TypeError("An explicit RetryPolicy is required")
    if should_pause is not None and not callable(should_pause):
        raise TypeError("should_pause must be callable or None")
    if type(resume_quota) is not bool:
        raise TypeError("resume_quota must be a boolean")
    if type(recover_interrupted) is not bool:
        raise TypeError("recover_interrupted must be a boolean")
    if not isinstance(frozen_job, dict) or not IDENTITY_KEYS.issubset(frozen_job):
        raise ValueError("frozen_job must explicitly cover payload, template, grading, problem and repeat identity")
    frozen_job = json.loads(_canonical(frozen_job))
    if any(not isinstance(frozen_job[k], str) or not frozen_job[k]
           for k in ("model", "template_sha256", "problem_sha256", "grader_sha256")):
        raise ValueError("Explicit model, template, problem and grader fingerprints are required")
    if type(frozen_job["repeat_index"]) is not int or frozen_job["repeat_index"] < 0:
        raise ValueError("repeat_index must be an explicit nonnegative integer")
    if frozen_job["payload"] != job.get("payload") or frozen_job["model"] != job.get("model"):
        raise ValueError("Frozen payload/model differs from execution job")
    if not isinstance(job["payload"], dict) or job["payload"].get("model") != job["model"]:
        raise ValueError("Execution model differs from payload model")
    if frozen_job["problem_sha256"] != job.get("problem_sha256"):
        raise ValueError("Frozen problem differs from execution job")
    if frozen_job["provider"] != job["payload"].get("provider"):
        raise ValueError("Frozen provider differs from payload")
    fingerprint = hashlib.sha256(_canonical(frozen_job).encode()).hexdigest()
    manifest = {"scheduler_version": "transport_retry_v003_interruption_recovery", "job_sha256": fingerprint,
                "frozen_job": frozen_job, "retry_policy": policy.as_dict()}
    case_dir = Path(case_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "case.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "pending_case_locked", "finalized": False,
                    "selected_attempt": None, "correct": None, "case_dir": str(case_dir)}
        manifest_path = case_dir / "case_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Frozen job or retry policy differs from existing case")
        else:
            _write_new(manifest_path, manifest)
        attempts_dir = case_dir / "attempts"
        attempts_dir.mkdir(exist_ok=True)
        attempts_on_entry = len(list(attempts_dir.glob("attempt_*")))

        def summarize(status, outcomes, selected=None):
            summary = {"status": status, "finalized": selected is not None,
                       "case_dir": str(case_dir), "job_sha256": fingerprint,
                       "retry_policy": policy.as_dict(), "attempts": outcomes,
                       "selected_attempt": selected["attempt"] if selected else None,
                       "selected_result_path": selected["result_path"] if selected else None,
                       "selected_model_status": selected["status"] if selected else None,
                       "correct": selected.get("correct") if selected else None,
                       "accounting": _accounting(outcomes), "updated_epoch": now_fn()}
            _write_summary(case_dir / "case_summary.json", summary)
            return summary

        while True:
            dirs = _attempt_directories(attempts_dir)
            if policy.max_attempts is not None and len(dirs) > policy.max_attempts:
                raise ValueError("Persisted attempts exceed the frozen retry limit")
            outcomes = [_outcome(path, index, policy, now_fn, fingerprint, job) for index, path in enumerate(dirs, 1)]
            quota_to_resume = None
            recovered_claim = False
            for outcome in outcomes:
                if outcome["status"] == "unresolved_claim":
                    if not recover_interrupted or outcome["attempt"] > attempts_on_entry:
                        return summarize("pending_inflight_or_interrupted", outcomes)
                    if outcome["attempt"] != len(outcomes):
                        return summarize("pending_interrupted_recovery_review", outcomes)
                    if should_pause is not None and should_pause():
                        return summarize("pending_global_pause", outcomes)
                    try:
                        _recover_interrupted(Path(outcome["attempt_dir"]), outcome["attempt"],
                                             fingerprint, policy, job, now_fn)
                    except (OSError, KeyError, TypeError, ValueError):
                        return summarize("pending_interrupted_recovery_review", outcomes)
                    # Re-read every saved record before scheduling a successor.
                    recovered_claim = True
                    break
                if outcome["classification"]["kind"] == "blocked_engineering":
                    return summarize("pending_engineering_review", outcomes)
                if outcome["classification"]["kind"] == "model_terminal":
                    return summarize("completed", outcomes, outcome)
                if outcome["classification"]["kind"] == "quota_pause":
                    try:
                        authorization = quota_resume_record(case_dir, outcome, fingerprint)
                    except ValueError:
                        return summarize("pending_quota_resume_review", outcomes)
                    if authorization is not None:
                        continue
                    if outcome["attempt"] < len(outcomes):
                        return summarize("pending_quota_resume_review", outcomes)
                    if not resume_quota or outcome["attempt"] > attempts_on_entry:
                        return summarize("pending_quota", outcomes)
                    quota_to_resume = outcome
            if recovered_claim:
                continue
            if not policy.can_retry(len(outcomes)):
                return summarize("pending_retry_exhausted", outcomes)
            if should_pause is not None and should_pause():
                return summarize("pending_global_pause", outcomes)
            if outcomes and outcomes[-1]["classification"]["kind"] == "retryable_transport":
                deadline = outcomes[-1]["next_attempt_not_before_epoch"]
                delay = max(0.0, deadline - now_fn())
                if delay:
                    summarize("pending_retry_delay", outcomes)
                    while delay > 0:
                        if should_pause is not None and should_pause():
                            return summarize("pending_global_pause", outcomes)
                        before = now_fn()
                        sleep_fn(min(delay, 1.0) if should_pause is not None else delay)
                        after = now_fn()
                        if after <= before:
                            return summarize("pending_retry_delay", outcomes)
                        delay = max(0.0, deadline - after)
            if should_pause is not None and should_pause():
                return summarize("pending_global_pause", outcomes)
            index = len(outcomes) + 1
            if (job.get("payload") != frozen_job["payload"] or job.get("model") != frozen_job["model"]
                    or job.get("problem_sha256") != frozen_job["problem_sha256"]):
                return summarize("pending_execution_identity_changed", outcomes)
            attempt_dir = attempts_dir / f"attempt_{index:03d}"
            attempt_dir.mkdir()  # Atomic exclusive claim; an existing attempt is immutable.
            _write_new(attempt_dir / "attempt_claim.json", {
                "attempt": index, "job_sha256": fingerprint, "claimed_epoch": now_fn(),
                "executor": executor_identity()})
            if quota_to_resume is not None:
                _record_quota_resume(case_dir, quota_to_resume, fingerprint, now_fn)
            try:
                result = attempt_execute(job, attempt_dir)
                if not isinstance(result, dict):
                    raise TypeError("Attempt callback must return a full result object")
            except Exception as exc:
                # Exception messages may contain credentials or request bodies.
                result = {"status": "client_error", "exception_type": type(exc).__name__,
                          "error_codes": ["attempt_callback_exception"], "usage": {}}
            if (job.get("payload") != frozen_job["payload"] or job.get("model") != frozen_job["model"]
                    or job.get("problem_sha256") != frozen_job["problem_sha256"]):
                _write_new(attempt_dir / "execution_job_mutated.json", {"requires_review": True})
            result_path = attempt_dir / "result.json"
            if result_path.exists():
                try:
                    saved = json.loads(result_path.read_text())
                except ValueError:
                    # The next loop reports the persisted partial record as unresolved.
                    continue
                if _canonical(saved) != _canonical(result):
                    # Retain the transport's persisted record, and require inspection.
                    _write_new(attempt_dir / "callback_result_mismatch.json", {"requires_review": True})
                    return summarize("pending_callback_result_mismatch", [*outcomes, {
                        "attempt": index, "status": "callback_result_mismatch",
                        "classification": {"kind": "blocked_engineering", "reason": "callback_result_mismatch"},
                        "usage": (saved.get("usage") or {}) if isinstance(saved, dict) else {},
                        "result_path": str(result_path)}])
            else:
                _write_new(result_path, result)
