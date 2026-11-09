"""Original teacher/student prompt shared by the teacher-CoT training entry."""

GRID = 81

INSTRUCTION_ROWS = (
    "Solve this Sudoku puzzle.\n\n"
    "The grid is given as nine rows of nine digits, where 0 marks an empty cell:\n"
    "{puzzle}\n\n"
    "Work it out, then give your final answer as the completed grid in the same layout: "
    "nine lines of nine digits, with no spaces or separators."
)

def as_rows(question: str) -> str:
    return "\n".join(question[i : i + 9] for i in range(0, GRID, 9))

def make_prompt(question: str, layout: str = "rows") -> list[dict]:
    """Chat-format prompt; the trainer / vLLM apply the model's own template (thinking on)."""
    if layout == "rows":
        content = INSTRUCTION_ROWS.format(puzzle=as_rows(question))
    elif layout == "line":
        from hrm_qwen.sft.prompt_eval import INSTRUCTION
        content = INSTRUCTION.format(puzzle=question)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return [{"role": "user", "content": content}]
