# sudoku
import json

from datasets import load_dataset, Features, Value

sample = {
    'question': '7.....8....2..5....3..6..9..7.6....1...1.3.7.....7.9.....4.2....9..3..1..2..1643.', 
    'answer': '769341852142895367538267194273689541954123678681574923316452789495738216827916435'
}

def get_dataset():
    return load_dataset("sapientinc/sudoku-extreme-1k", cache_dir="/dev/shm/datasets", features=Features({"question": Value("string"), "answer": Value("string")}), split="test_hard")

def get_prompt():
    return 'Solve the following 9x9 Sudoku puzzle.\n\nInput format:\nThe JSON array contains exactly 9 strings, one per row from top to bottom. Each string contains exactly 9 characters, one per cell from left to right. A digit \'1\'-\'9\' is a given clue; \'.\' is an empty cell.\n\nRules:\nThe supplied puzzle has at least one valid solution.\nFill the empty cells so that each row, each column, and each of the nine 3x3 boxes contains every digit from \'1\' to \'9\' exactly once.\nKeep every given clue unchanged in its original cell.\nFor cell coordinates, rows are indexed from 1 through 9, top to bottom, and columns from 1 through 9, left to right.\nThe grid is divided into the following nine 3x3 boxes:\n- Box 1: rows 1-3, columns 1-3.\n- Box 2: rows 1-3, columns 4-6.\n- Box 3: rows 1-3, columns 7-9.\n- Box 4: rows 4-6, columns 1-3.\n- Box 5: rows 4-6, columns 4-6.\n- Box 6: rows 4-6, columns 7-9.\n- Box 7: rows 7-9, columns 1-3.\n- Box 8: rows 7-9, columns 4-6.\n- Box 9: rows 7-9, columns 7-9.\nEach box contains all 9 cells in its stated row and column ranges; every cell belongs to exactly one box.\n\n<puzzle_rows>\n{question}\n</puzzle_rows>\n\nFinal output format:\nThe entire final response must be one JSON object with exactly one key, "answer".\nThe first non-whitespace character must be \'{{\', and the last non-whitespace character must be \'}}\'.\nAfter JSON decoding, the value of "answer" must be a string of exactly 81 digits from \'1\' to \'9\': the solved first row, followed by the solved second row, and so on through the ninth row, with each row read left to right.\n'

def format_prompt(question: str):
    if len(question) != 81:
        raise ValueError("Expected a Sudoku question with exactly 81 characters")
    rows = [question[start:start + 9] for start in range(0, 81, 9)]
    return get_prompt().format(question=json.dumps(rows, indent=2))

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
