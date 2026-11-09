import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _build_payload(prompt: str, model: str, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {
            "effort": "high",
            "enabled": True,
        },
        "stream": stream,
    }


def _build_request(payload: dict) -> Request:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

    return Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _request_llm(prompt: str, model: str):
    request = _build_request(_build_payload(prompt, model, stream=False))

    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed with status {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc


def _iter_sse_events(response):
    data_lines = []

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace")

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue

        if not line.strip() and data_lines:
            yield "".join(data_lines).strip()
            data_lines = []

    if data_lines:
        yield "".join(data_lines).strip()


def _collect_text_values(value, preferred_keys=None):
    texts = []

    def _walk(node):
        if isinstance(node, str):
            stripped = node.strip()
            if stripped:
                texts.append(stripped)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)
            return

        if isinstance(node, dict):
            if preferred_keys:
                for key in preferred_keys:
                    if key in node:
                        _walk(node[key])
                return
            for item in node.values():
                _walk(item)

    _walk(value)
    return texts


def _extract_text_from_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_parts:
            return "".join(text_parts)

    if isinstance(content, dict):
        text_parts = _collect_text_values(content, preferred_keys=("text", "content"))
        if text_parts:
            return "".join(text_parts)

    return None


def _extract_reasoning_parts(value):
    parts = []

    def _walk(node):
        if isinstance(node, str):
            stripped = node.strip()
            if stripped:
                parts.append(stripped)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)
            return

        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type in {"reasoning", "thinking", "reasoning_text", "reasoning.text"}:
                extracted = _collect_text_values(node, preferred_keys=("text", "content", "summary", "reasoning"))
                if extracted:
                    parts.extend(extracted)
                    return
            for key in ("reasoning", "thinking", "reasoning_text", "reasoning_details", "summary"):
                if key in node:
                    _walk(node[key])
            if node_type in {"reasoning", "thinking", "reasoning_text", "reasoning.text"}:
                return

    _walk(value)
    return parts


def _dedupe_preserve_order(parts):
    seen = set()
    deduped = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            deduped.append(part)
    return deduped


def _extract_reasoning_text(choice: dict, message: dict, content):
    reasoning_parts = []

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"reasoning", "thinking", "reasoning_text", "reasoning.text"}:
                reasoning_parts.extend(_extract_reasoning_parts(item))

    for container in (message, choice, choice.get("delta") if isinstance(choice, dict) else None):
        if isinstance(container, dict):
            reasoning_parts.extend(_extract_reasoning_parts(container))

    return "\n\n".join(_dedupe_preserve_order(reasoning_parts))


def _extract_error(choice: dict, message: dict):
    for container in (choice, message):
        if isinstance(container, dict) and container.get("error"):
            return container["error"]
    return None


def _stream_llm_trace(prompt: str, model: str, on_reasoning=None, on_content=None):
    request = _build_request(_build_payload(prompt, model, stream=True))
    chunks = []
    reasoning_parts = []
    content_parts = []

    try:
        with urlopen(request) as response:
            for event in _iter_sse_events(response):
                if not event or event == "[DONE]":
                    continue

                chunk = json.loads(event)
                chunks.append(chunk)
                choices = chunk.get("choices") or []
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta") or {}
                chunk_reasoning_parts = _extract_reasoning_parts(delta)
                if not chunk_reasoning_parts and isinstance(choice.get("message"), dict):
                    chunk_reasoning_parts = _extract_reasoning_parts(choice["message"])

                if chunk_reasoning_parts:
                    for part in chunk_reasoning_parts:
                        reasoning_parts.append(part)
                        if on_reasoning is not None:
                            on_reasoning(part)

                content_text = _extract_text_from_content(delta.get("content"))
                if content_text:
                    content_parts.append(content_text)
                    if on_content is not None:
                        on_content(content_text)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed with status {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

    return {
        "content": "".join(content_parts),
        "reasoning": "\n\n".join(_dedupe_preserve_order(reasoning_parts)),
        "raw_result": chunks,
        "error": None,
    }


def run_llm_trace(prompt: str, model: str, stream: bool = False, on_reasoning=None, on_content=None):
    if stream:
        return _stream_llm_trace(
            prompt,
            model,
            on_reasoning=on_reasoning,
            on_content=on_content,
        )

    result = _request_llm(prompt, model)
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {result}")

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    answer_text = _extract_text_from_content(content)
    reasoning_text = _extract_reasoning_text(choice, message, content)
    error = _extract_error(choice, message)

    return {
        "content": answer_text or "",
        "reasoning": reasoning_text,
        "raw_result": result,
        "error": error,
    }


def run_llm(prompt, model):
    return run_llm_trace(prompt, model)["content"]


def robust_json_parse(body: str):
    """Robustly extract and parse JSON from an LLM response body.

    LLMs often wrap JSON in markdown fences, explanatory text, or other
    decorations.  This helper tries increasingly aggressive strategies to
    locate and parse a valid JSON value (object or array).

    Returns the parsed Python object (dict / list), or raises ValueError
    if no valid JSON can be recovered.
    """

    def _try_parse(s: str):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None

    text = body.strip()

    result = _try_parse(text)
    if result is not None:
        return result

    fence_pattern = re.compile(
        r"```(?:json|JSON|js|javascript)?\s*\n?(.*?)\n?\s*```",
        re.DOTALL,
    )
    for match in fence_pattern.finditer(text):
        candidate = match.group(1).strip()
        result = _try_parse(candidate)
        if result is not None:
            return result

    def _find_balanced(s: str, open_ch: str, close_ch: str):
        for i, ch in enumerate(s):
            if ch == open_ch:
                depth = 0
                in_string = False
                escape = False
                for j in range(i, len(s)):
                    c = s[j]
                    if escape:
                        escape = False
                        continue
                    if c == "\\":
                        if in_string:
                            escape = True
                        continue
                    if c == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if c == open_ch:
                        depth += 1
                    elif c == close_ch:
                        depth -= 1
                        if depth == 0:
                            yield s[i : j + 1]
                            break

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        for candidate in _find_balanced(text, open_ch, close_ch):
            result = _try_parse(candidate)
            if result is not None:
                return result

    sanitized = text
    sanitized = re.sub(r"(?<=[\[{,])\s*'([^']+?)'\s*:", r' "\1":', sanitized)
    sanitized = re.sub(r":\s*'([^']*?)'\s*(?=[,\]}])", r': "\1"', sanitized)
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        for candidate in _find_balanced(sanitized, open_ch, close_ch):
            result = _try_parse(candidate)
            if result is not None:
                return result

    raise ValueError(f"Could not extract valid JSON from the response body:\n{body[:500]}")


def run_llm_json(prompt: str, model: str):
    result = run_llm(prompt, model)
    return robust_json_parse(result)
