"""Single-attempt transport; transport_retry_v001 selects per-case attempts."""
import argparse


from functools import partial


import ast


from concurrent.futures import ThreadPoolExecutor, as_completed


from datetime import datetime, timezone


import hashlib


import json


import os


from pathlib import Path


import random


import re


import socket


import ssl


import sys


import time


import types


from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
from email.utils import parsedate_to_datetime


from urllib.request import HTTPRedirectHandler, Request, build_opener


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


TASK_FILES = {"sudoku": "sudoku.py", "maze": "maze.py", "sat": "sat3.py"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def problem_hash(problem):
    canonical = json.dumps(problem, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_task_functions(path):
    """Skip dataset imports, dataset loading and module-level examples."""
    module = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    names = {"get_prompt", "format_prompt", "validate"}
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if not {"get_prompt", "validate"}.issubset({node.name for node in functions}):
        raise ValueError("Missing prompt or validation function")
    namespace = {"json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_client(path):
    # Load definitions without creating bytecode in frozen experiment folders.
    module = types.ModuleType("pilot_v004_module")
    module.__file__ = str(path)
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def grade(client, task, problem, content, validator, maze_dimensions=None):
    outcome = {"correct": False, "json_parse_success": False, "strict_json_object": False,
               "answer_schema_valid": False, "strict_output_format": False}
    try:
        parsed = client.robust_json_parse(content)
    except (ValueError, TypeError):
        return outcome
    outcome["json_parse_success"] = True
    try:
        strict = json.loads(content.strip())
        outcome["strict_json_object"] = isinstance(strict, dict) and set(strict) == {"answer"}
    except (ValueError, TypeError):
        pass
    if not isinstance(parsed, dict) or "answer" not in parsed:
        return outcome
    answer = parsed["answer"]
    if task == "sat":
        outcome["answer_schema_valid"] = (isinstance(answer, list) and len(answer) == problem["n"]
                                           and all(type(value) is int and value in (0, 1) for value in answer))
    else:
        dims = maze_dimensions or {"height": 30, "width": 30}
        required_length = 81 if task == "sudoku" else dims["height"] * dims["width"]
        alphabet = "123456789" if task == "sudoku" else "# SGo"
        outcome["answer_schema_valid"] = (isinstance(answer, str) and len(answer) == required_length
                                           and all(char in alphabet for char in answer))
    outcome["strict_output_format"] = outcome["strict_json_object"] and outcome["answer_schema_valid"]
    outcome["parsed_answer"] = answer
    try:
        outcome["correct"] = bool(validator(problem["clauses"], answer) if task == "sat" else validator(problem["question"], answer))
    except Exception as exc:
        outcome["validator_exception_type"] = type(exc).__name__
    return outcome


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def sanitize_debug(debug, provider):
    """Persist only configuration fields, never the echoed request body."""
    safe = {"provider": provider if isinstance(provider, str) else None,
            "upstream_keys": [], "upstream_parameters": {}}
    upstream = debug.get("echo_upstream_body") if isinstance(debug, dict) else None
    if isinstance(upstream, str):
        try:
            upstream = json.loads(upstream)
        except ValueError:
            upstream = None
    if not isinstance(upstream, dict):
        safe["diagnostic"] = "missing_or_invalid_upstream_body"
        return safe
    # Field names help explain provider translations; values are whitelisted.
    safe["upstream_keys"] = sorted(upstream)
    allowed = {"model", "max_tokens", "max_output_tokens", "max_completion_tokens",
               "reasoning_effort", "stream", "temperature", "top_p", "seed",
               "enable_thinking", "include_reasoning", "store", "truncation"}
    for key in allowed:
        if key in upstream and isinstance(upstream[key], (str, int, float, bool, type(None))):
            safe["upstream_parameters"][key] = upstream[key]
    for key, fields in {"reasoning": {"effort", "enabled", "max_tokens", "summary", "exclude"},
                        "thinking": {"type", "budget_tokens"},
                        "chat_template_kwargs": {"enable_thinking", "thinking", "reasoning_effort"}}.items():
        if isinstance(upstream.get(key), dict):
            safe["upstream_parameters"][key] = {
                field: value for field, value in upstream[key].items()
                if field in fields and isinstance(value, (str, int, float, bool, type(None)))}
    for name in ("generationConfig", "generation_config"):
        config = upstream.get(name)
        if isinstance(config, dict):
            selected = {k: config[k] for k in ("maxOutputTokens", "max_output_tokens", "temperature", "topP", "top_p", "seed")
                        if k in config and isinstance(config[k], (str, int, float, bool, type(None)))}
            for thinking_name in ("thinkingConfig", "thinking_config"):
                thinking = config.get(thinking_name)
                if isinstance(thinking, dict):
                    selected[thinking_name] = {k: thinking[k] for k in ("thinkingLevel", "thinking_level", "thinkingBudget", "thinking_budget", "includeThoughts", "include_thoughts")
                                              if k in thinking and isinstance(thinking[k], (str, int, float, bool, type(None)))}
            safe["upstream_parameters"][name] = selected
    return safe


def parse_retry_after(value):
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            seconds = (target - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if 0 <= seconds < float("inf") else None


def nested_value(obj, path):
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def unknown_error_model_is_missing(result, expected_providers, expected_models, issues):
    """Recognize only an output-free provider error's literal model sentinel."""
    def code(value):
        return value if type(value) is int else int(value) if isinstance(value, str) and value.isascii() and value.isdigit() else None
    errors = result.get("error_codes")
    transient = {408, 429, *range(500, 600)}
    if (result.get("status") != "sse_error" or result.get("http_status") != 200 or
            not isinstance(errors, list) or not errors or
            not any(code(value) in transient for value in errors) or
            any(code(value) not in transient and value != "finish_reason_error" for value in errors)):
        return False
    providers, models = result.get("observed_providers"), result.get("observed_models")
    return (isinstance(expected_models, list) and isinstance(models, list) and "unknown" in models and
            all(model == "unknown" or model in expected_models for model in models) and result.get("response_model") in models and
            isinstance(providers, list) and bool(providers) and all(value in expected_providers for value in providers) and
            result.get("model_output_present") is False and result.get("final_content") == "" and
            not result.get("reasoning") and not any(result.get("reasoning_channels", {}).values()) and
            all(isinstance(result.get(key), list) and all(value == "error" for value in result[key])
                for key in ("finish_reasons", "native_finish_reasons")) and
            all(issue == "response_model_mismatch" or issue.endswith("_missing") for issue in issues))


def check_configuration(job, result):
    profile = job["model_profile"]
    expected = profile["expected_provider_names"]
    providers = result.get("observed_providers", [])
    events = result.get("upstream_debug", [])
    issues = []
    expected_models = profile.get("expected_response_models")
    if expected_models:
        observed_models = result.get("observed_models") or ([result["response_model"]] if result.get("response_model") else [])
        if not observed_models:
            issues.append("response_model_missing")
        elif any(model not in expected_models for model in observed_models):
            issues.append("response_model_mismatch")
    if not providers:
        issues.append("response_provider_missing")
    elif any(provider not in expected for provider in providers):
        issues.append("response_provider_mismatch")
    if not events:
        issues.append("upstream_debug_missing")
    for event in events:
        if event.get("provider") is None:
            issues.append("debug_provider_missing")
        elif event.get("provider") not in expected:
            issues.append("debug_provider_mismatch")
        params = event.get("upstream_parameters", {})
        for requirement in profile["upstream_requirements"]:
            observed = [nested_value(params, path) for path in requirement["paths"]]
            observed = [value for value in observed if value is not None]
            if not observed:
                issues.append("upstream_" + requirement["name"] + "_missing")
            elif any(value not in requirement["accepted_values"] for value in observed):
                issues.append("upstream_" + requirement["name"] + "_mismatch")
    if "response_model_mismatch" in issues and unknown_error_model_is_missing(result, expected, expected_models, issues):
        issues.remove("response_model_mismatch")
        issues.append("response_model_missing")
    return {"status": "issues" if issues else "verified", "issues": sorted(set(issues)),
            "expected_provider_names": expected, "expected_response_models": expected_models, "observed_providers": providers,
            "expected_max_completion_tokens": job["call_settings"]["max_completion_tokens"],
            "upstream_debug_count": len(events)}


def raw_text_fragments(value):
    """Extract published text without stripping, inserting or deduplicating."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in raw_text_fragments(item)]
    if isinstance(value, dict):
        # Encrypted reasoning and IDs are not text. The full SSE record remains
        # available separately, including details metadata and original ordering.
        return [text for key in ("text", "content", "summary", "reasoning", "thinking")
                if key in value for text in raw_text_fragments(value[key])]
    return []


def collect_reasoning_channels(delta, channels):
    # Providers can expose the same text through more than one channel. Keep
    # each channel independently so redundant channels are not concatenated.
    for key in ("reasoning", "reasoning_text", "thinking", "reasoning_details", "reasoning_content", "summary"):
        if key in delta:
            channels.setdefault(key, []).extend(raw_text_fragments(delta[key]))
    if isinstance(delta.get("content"), list):
        for item in delta["content"]:
            if isinstance(item, dict) and item.get("type") in {"reasoning", "thinking", "reasoning_text", "reasoning.text"}:
                channels.setdefault("content_reasoning", []).extend(raw_text_fragments(item))


def run_one(job, client, api_key, timeout):
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    identity = {key: job[key] for key in ("task", "variant", "model", "case_id", "problem_sha256")}
    try:
        write_new_json(output / "started.json", {**identity, "started_utc": utc_now()})
    except FileExistsError:
        if (output / "result.json").exists():
            try:
                existing = json.loads((output / "result.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {**identity, "status": "incomplete_existing_not_retried", "correct": False, "skipped_existing": True}
            if any(existing.get(key) != value for key, value in identity.items()):
                raise ValueError("Existing case identity mismatch")
            return {**existing, "skipped_existing": True}
        return {**identity, "status": "incomplete_existing_not_retried", "correct": False, "skipped_existing": True}
    write_new_json(output / "request.json", job["payload"])
    write_new_json(output / "problem.json", job["problem"])
    request = Request(ENDPOINT, data=json.dumps(job["payload"]).encode("utf-8"),
                      headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    result = {**identity, "started_utc": utc_now(), "status": "incomplete_stream",
              "http_status": None, "saw_done": False, "finish_reasons": [], "native_finish_reasons": [], "usage": {},
              "error_codes": [], "response_model": None, "response_id": None, "provider": None,
              "request_call_settings": job["call_settings"], "upstream_debug": [], "observed_providers": [], "observed_models": [],
              "terminal_finish_reason": None, "terminal_event_index": None, "stream_termination": None,
              "model_output_present": False}
    content_parts, reasoning_channels = [], {}
    first_event = None
    saved_event_count = 0
    with (output / "events.jsonl").open("x", encoding="utf-8") as events:
        def save_event(value):
            nonlocal saved_event_count
            serialized = json.dumps({"received_utc": utc_now(), "elapsed_seconds": time.monotonic() - started, **value}, ensure_ascii=False)
            events.write(serialized.replace(api_key, "[REDACTED]") + "\n")
            events.flush()
            saved_event_count += 1

        def record_transport_error(status, exception_type):
            error = {"status": status, "exception_type": exception_type}
            after_terminal = result["terminal_finish_reason"] is not None
            save_event({"transport_error": error, "after_terminal": after_terminal})
            result["stream_termination"] = "transport_error"
            if after_terminal:
                result["tail_transport_error"] = error
            else:
                result["status"] = status
                result["exception_type"] = exception_type

        def record_client_error(exception_type):
            result["status"] = "client_error"
            result["exception_type"] = exception_type
            result["stream_termination"] = "client_error"
            save_event({"client_error": {"exception_type": exception_type},
                        "after_terminal": result["terminal_finish_reason"] is not None})
        try:
            with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
                result["http_status"] = response.status
                result["retry_after_seconds"] = parse_retry_after(response.headers.get("Retry-After"))
                for event in client._iter_sse_events(response):
                    if first_event is None:
                        first_event = time.monotonic() - started
                    if event == "[DONE]":
                        result["saw_done"] = True
                        result["stream_termination"] = "done"
                        save_event({"done": True})
                        break
                    if not event:
                        continue
                    try:
                        chunk = json.loads(event)
                    except ValueError:
                        # A malformed event cannot be safely checked for echoed
                        # debug credentials; persist its identity, not its body.
                        save_event({"unparsed_event_sha256": hashlib.sha256(event.encode()).hexdigest(),
                                    "unparsed_event_characters": len(event)})
                        result["error_codes"].append("invalid_sse_json")
                        continue
                    if isinstance(chunk, dict) and "debug" in chunk:
                        safe_debug = sanitize_debug(chunk["debug"], chunk.get("provider"))
                        chunk["debug"] = safe_debug
                        result["upstream_debug"].append(safe_debug)
                    save_event({"chunk": chunk})
                    if not isinstance(chunk, dict):
                        result["error_codes"].append("invalid_sse_object")
                        continue
                    for key in ("model", "id", "provider"):
                        if chunk.get(key):
                            result[{"model": "response_model", "id": "response_id", "provider": "provider"}[key]] = chunk[key]
                    if isinstance(chunk.get("provider"), str) and chunk["provider"] not in result["observed_providers"]:
                        result["observed_providers"].append(chunk["provider"])
                    if isinstance(chunk.get("model"), str) and chunk["model"] not in result["observed_models"]:
                        result["observed_models"].append(chunk["model"])
                    if isinstance(chunk.get("usage"), dict):
                        result["usage"].update(chunk["usage"])
                    if chunk.get("error"):
                        error = chunk["error"]
                        result["error_codes"].append(error.get("code", "top_level_error") if isinstance(error, dict) else "top_level_error")
                    choices = chunk.get("choices") or []
                    if not isinstance(choices, list):
                        result["error_codes"].append("invalid_choices_container")
                        continue
                    for choice in choices:
                        if not isinstance(choice, dict):
                            result["error_codes"].append("invalid_choice")
                            continue
                        if choice.get("index", 0) != 0:
                            result["error_codes"].append("unexpected_multiple_choices")
                            continue
                        delta = choice.get("delta") or choice.get("message") or {}
                        for field in ("delta", "message"):
                            choice_output = choice.get(field)
                            if (isinstance(choice_output, dict) and any(value for key, value in choice_output.items() if key not in {"role", "error"})
                                    or choice_output and not isinstance(choice_output, dict)):
                                result["model_output_present"] = True
                        finish = choice.get("finish_reason")
                        if result["terminal_finish_reason"] is not None:
                            if not isinstance(delta, dict):
                                result["error_codes"].append("invalid_delta")
                                continue
                            post_reasoning = {}
                            collect_reasoning_channels(delta, post_reasoning)
                            if (client._extract_text_from_content(delta.get("content"))
                                    or any("".join(parts) for parts in post_reasoning.values())
                                    or finish not in (None, result["terminal_finish_reason"])
                                    or any(container.get("error") for container in (choice, delta))):
                                result["error_codes"].append("unexpected_post_terminal_choice")
                            continue
                        if finish is not None:
                            result["finish_reasons"].append(finish)
                        native_finish = choice.get("native_finish_reason")
                        if native_finish is not None:
                            result["native_finish_reasons"].append(native_finish)
                        if finish == "error":
                            result["error_codes"].append("finish_reason_error")
                        for container in (choice, delta):
                            if isinstance(container, dict) and container.get("error"):
                                error = container["error"]
                                result["error_codes"].append(error.get("code", "choice_error") if isinstance(error, dict) else "choice_error")
                        if not isinstance(delta, dict):
                            result["error_codes"].append("invalid_delta")
                            continue
                        text = client._extract_text_from_content(delta.get("content"))
                        if text:
                            content_parts.append(text)
                        collect_reasoning_channels(delta, reasoning_channels)
                        if finish in {"stop", "length", "content_filter"}:
                            result["terminal_finish_reason"] = finish
                            result["terminal_event_index"] = saved_event_count
                if result["stream_termination"] is None:
                    result["stream_termination"] = "clean_eof"
                    if result["terminal_finish_reason"] is None:
                        record_transport_error("network_error", "UnexpectedStreamEOF")
        except HTTPError as exc:
            result["http_status"] = exc.code
            result["retry_after_seconds"] = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            result["status"] = "http_error"
            result["stream_termination"] = "http_error"
            result["error_codes"].append(exc.code)
            error_event = {"http_error_status": exc.code}
            try:
                body = exc.read(65537)
                error_event["body_truncated"] = len(body) > 65536
                error_event["body_sha256"] = hashlib.sha256(body[:65536]).hexdigest()
                error_event["body_recorded_bytes"] = min(len(body), 65536)
            except Exception as read_exc:
                error_event["body_read_exception_type"] = type(read_exc).__name__
            save_event(error_event)
        except (TimeoutError, socket.timeout) as exc:
            record_transport_error("socket_timeout", type(exc).__name__)
        except URLError as exc:
            if isinstance(exc.reason, ssl.SSLError) and not isinstance(exc.reason, (ssl.SSLEOFError, ssl.SSLZeroReturnError)):
                record_client_error(type(exc.reason).__name__)
            else:
                status = "socket_timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "network_error"
                record_transport_error(status, type(exc.reason).__name__)
        except (ConnectionError, EOFError, IncompleteRead, ssl.SSLEOFError, ssl.SSLZeroReturnError) as exc:
            record_transport_error("network_error", type(exc).__name__)
        except Exception as exc:
            record_client_error(type(exc).__name__)
    # A model terminal fixes the selected answer even when the usage/DONE tail
    # is lost. Protocol or configuration problems still require investigation.
    if result["error_codes"] and result["status"] not in {"http_error", "client_error"}:
        result["status"] = "sse_error"
    elif result["status"] not in {"http_error", "client_error"}:
        finish = result["terminal_finish_reason"]
        if finish == "length":
            result["status"] = "length_limited"
        elif finish == "content_filter":
            result["status"] = "content_filtered"
        elif finish == "stop":
            result["status"] = "completed" if content_parts else "no_final_content"
    result["elapsed_seconds"] = time.monotonic() - started
    result["finished_utc"] = utc_now()
    result["time_to_first_event_seconds"] = first_event
    result["final_content"] = "".join(content_parts)
    result["reasoning_channels"] = {key: "".join(parts) for key, parts in reasoning_channels.items()}
    priority = ("reasoning", "reasoning_text", "thinking", "reasoning_details", "reasoning_content", "summary", "content_reasoning")
    result["reasoning_source"] = next((key for key in priority if result["reasoning_channels"].get(key)), None)
    result["reasoning"] = result["reasoning_channels"].get(result["reasoning_source"], "")
    result["configuration_check"] = check_configuration(job, result)
    result["payload_sha256"] = problem_hash(job["payload"])
    if result["status"] in {"completed", "length_limited", "content_filtered", "no_final_content"} and result["configuration_check"]["status"] != "verified":
        result["model_terminal_status_before_configuration_check"] = result["status"]
        result["status"] = "configuration_error"
    result["transport_success"] = result["status"] in {"completed", "length_limited", "content_filtered", "no_final_content"}
    result.update(grade(client, job["task"], job["problem"], result["final_content"], job["validator"],
                        maze_dimensions=job.get("maze_dimensions")))
    if job.get("maze_diagnose") and "parsed_answer" in result:
        result["maze_diagnostics"] = job["maze_diagnose"](job["problem"]["question"], result["parsed_answer"])
    result = json.loads(json.dumps(result, ensure_ascii=False).replace(api_key, "[REDACTED]"))
    write_new_json(output / "result.json", result)
    return result


def safe_progress(result):
    usage = result.get("usage") or {}
    return {key: result.get(key) for key in ("task", "variant", "case_id", "status", "correct", "elapsed_seconds", "skipped_existing")} | {
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "cost": usage.get("cost"),
        "configuration_status": result.get("configuration_check", {}).get("status")}
