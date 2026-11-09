"""Offline scoring for a marked shortest S-to-G path, shared by both v003 arms.

The breadth-first search returns only a distance. It is used after inference for
scoring, and neither reconstructs a reference path nor counts shortest paths.
"""

from collections import deque


def diagnose(question, answer, height=30, width=30):
    """Return correctness, a reason, and available path lengths (in moves)."""
    result = {
        "correct": False,
        "reason": None,
        "path_length": None,
        "shortest_length": None,
    }

    def reject(reason):
        return {**result, "reason": reason}

    if (type(height) is not int or type(width) is not int
            or height <= 0 or width <= 0):
        return reject("invalid_dimensions")
    size = height * width
    if not isinstance(question, str) or len(question) != size:
        return reject("invalid_question_length")
    if any(cell not in "# SG" for cell in question):
        return reject("invalid_question_characters")
    if question.count("S") != 1 or question.count("G") != 1:
        return reject("invalid_question_endpoints")
    if not isinstance(answer, str) or len(answer) != size:
        return reject("invalid_answer_length")
    if any(cell not in "# SGo" for cell in answer):
        return reject("invalid_answer_characters")
    for original, returned in zip(question, answer):
        if original == " ":
            if returned not in " o":
                return reject("map_changed")
        elif returned != original:
            return reject("map_changed")

    def neighbors(index):
        row, col = divmod(index, width)
        if row > 0:
            yield index - width
        if row + 1 < height:
            yield index + width
        if col > 0:
            yield index - 1
        if col + 1 < width:
            yield index + 1

    start, goal = question.index("S"), question.index("G")
    selected = {index for index, cell in enumerate(answer) if cell in "SGo"}
    result["path_length"] = len(selected) - 1
    for index in selected:
        degree = sum(adjacent in selected for adjacent in neighbors(index))
        if degree != (1 if index in (start, goal) else 2):
            return reject("invalid_path_degree")

    reached = {start}
    pending = deque([start])
    while pending:
        for adjacent in neighbors(pending.popleft()):
            if adjacent in selected and adjacent not in reached:
                reached.add(adjacent)
                pending.append(adjacent)
    if reached != selected:
        return reject("disconnected_marked_cells")

    # Compute only the original maze's distance, after checking the answer path.
    pending = deque([(start, 0)])
    reached = {start}
    while pending:
        index, distance = pending.popleft()
        if index == goal:
            result["shortest_length"] = distance
            break
        for adjacent in neighbors(index):
            if question[adjacent] != "#" and adjacent not in reached:
                reached.add(adjacent)
                pending.append((adjacent, distance + 1))
    if result["shortest_length"] is None:
        return reject("no_path_in_question")
    if result["path_length"] != result["shortest_length"]:
        return reject("path_is_not_shortest")
    return {**result, "correct": True, "reason": "valid_shortest_path"}


def validate(question, answer, height=30, width=30):
    """Accept any valid shortest path with the input map preserved."""
    return diagnose(question, answer, height=height, width=width)["correct"]
