# sudoku
from datasets import load_dataset, Features, Value

sample = {
    'question': '7.....8....2..5....3..6..9..7.6....1...1.3.7.....7.9.....4.2....9..3..1..2..1643.', 
    'answer': '769341852142895367538267194273689541954123678681574923316452789495738216827916435'
}

def get_dataset():
    return load_dataset("sapientinc/sudoku-extreme-1k", cache_dir="/dev/shm/datasets", features=Features({"question": Value("string"), "answer": Value("string")}), split="test_hard")

def get_prompt():
    return """You are a Sudoku solver. 
You are given a 9x9 Sudoku puzzle represented as a single string of 81 characters, read left-to-right. 
Each character is either a digit '1'-'9' (a given clue) or '.' (an empty cell to fill).

Rules:
- Every row must contain the digits 1-9 exactly once.
- Every column must contain the digits 1-9 exactly once.
- Every 3x3 box must contain the digits 1-9 exactly once.
- All given clue digits must remain unchanged in the solution.

The puzzle is:
{question}

Respond with a JSON object in this format:
{{"answer": "<81-character string with all cells filled with digits 1-9>"}}
"""

def validate(question: str, answer: str):
    # Check length
    if len(answer) != 81:
        return False
    # Check all characters are digits 1-9
    if not all(c in '123456789' for c in answer):
        return False
    # Check answer matches question clues
    for i in range(81):
        if question[i] != '.' and question[i] != answer[i]:
            return False
    # Check rows
    for r in range(9):
        row = answer[r * 9:(r + 1) * 9]
        if len(set(row)) != 9:
            return False
    # Check columns
    for c in range(9):
        col = answer[c::9]
        if len(set(col)) != 9:
            return False
    # Check 3x3 boxes
    for br in range(3):
        for bc in range(3):
            box = []
            for r in range(3):
                for c in range(3):
                    box.append(answer[(br * 3 + r) * 9 + bc * 3 + c])
            if len(set(box)) != 9:
                return False
    return True
