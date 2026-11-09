"""Pure definitions copied from the frozen parser and independent audit."""
import json
import re
from collections import deque
FLAGS = ("correct", "json_parse_success", "strict_json_object", "answer_schema_valid", "strict_output_format")

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

def independent_validate(task, problem, answer):
    if task == 'sat':
        for clause in problem['clauses']:
            satisfied = False
            for literal in clause:
                index = abs(literal) - 1
                if index < 0 or index >= len(answer):
                    return False
                value = answer[index]
                if value == (1 if literal > 0 else 0):
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True
    question = problem['question']
    if task == 'sudoku':
        if len(answer) != 81 or any(x not in '123456789' for x in answer):
            return False
        if any(q != '.' and q != a for q, a in zip(question, answer)):
            return False
        groups = [[r * 9 + c for c in range(9)] for r in range(9)]
        groups += [[r * 9 + c for r in range(9)] for c in range(9)]
        groups += [[(br + r) * 9 + bc + c for r in range(3) for c in range(3)]
                   for br in (0, 3, 6) for bc in (0, 3, 6)]
        return all(len({answer[i] for i in group}) == 9 for group in groups)
    height = width = 30
    if (not isinstance(question, str) or len(question) != 900 or any(c not in '# SG' for c in question)
            or question.count('S') != 1 or question.count('G') != 1
            or not isinstance(answer, str) or len(answer) != 900 or any(c not in '# SGo' for c in answer)):
        return False
    if any(a not in ' o' if q == ' ' else a != q for q, a in zip(question, answer)):
        return False
    start, goal = question.index('S'), question.index('G')
    marked = {i for i, c in enumerate(answer) if c in 'SGo'}
    def neighbors(i):
        r, c = divmod(i, width)
        return [j for allowed, j in ((r > 0, i - width), (r + 1 < height, i + width),
                                    (c > 0, i - 1), (c + 1 < width, i + 1)) if allowed]
    if any(sum(j in marked for j in neighbors(i)) != (1 if i in (start, goal) else 2) for i in marked):
        return False
    reached, queue = {start}, deque([start])
    while queue:
        for j in neighbors(queue.popleft()):
            if j in marked and j not in reached:
                reached.add(j)
                queue.append(j)
    if reached != marked:
        return False
    # Distance only, solely for offline answer grading; no uniqueness investigation.
    distances, queue = {start: 0}, deque([start])
    while queue:
        i = queue.popleft()
        if i == goal:
            return distances[i] == len(marked) - 1
        for j in neighbors(i):
            if question[j] != '#' and j not in distances:
                distances[j] = distances[i] + 1
                queue.append(j)
    return False

def regrade(task, problem, final, parser):
    flags = dict.fromkeys(FLAGS, False)
    try:
        parsed = parser(final)
    except (ValueError, TypeError):
        return flags
    flags['json_parse_success'] = True
    try:
        direct = json.loads(final.strip())
        flags['strict_json_object'] = isinstance(direct, dict) and set(direct) == {'answer'}
    except (ValueError, TypeError):
        pass
    if not isinstance(parsed, dict) or 'answer' not in parsed:
        return flags
    answer = parsed['answer']
    if task == 'sat':
        valid = isinstance(answer, list) and len(answer) == problem['n'] and all(type(x) is int and x in (0, 1) for x in answer)
    else:
        length, alphabet = (81, '123456789') if task == 'sudoku' else (900, '# SGo')
        valid = isinstance(answer, str) and len(answer) == length and all(c in alphabet for c in answer)
    flags.update(answer_schema_valid=valid, strict_output_format=flags['strict_json_object'] and valid,
                 parsed_answer=answer)
    try:
        flags['correct'] = bool(independent_validate(task, problem, answer))
    except (ValueError, TypeError, KeyError, IndexError):
        flags['correct'] = False
    return flags
