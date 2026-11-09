# 3-sat dataset
import json

from functools import lru_cache
from pathlib import Path

from datasets import load_dataset

sample = {
    'n': 256,
    'assignment': [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 
    'clauses': [[-99, -25, -142], [-2, 178, 92], [-162, -75, -121], [165, -221, -26], [-218, 134, -194], [-69, 244, -8], [45, -230, 111], [3, 135, 149], [-214, -209, 41], [41, -126, 66], [-150, 69, -125], [172, 81, 29], [187, -130, 222], [227, -232, 155], [93, 164, 112], [-125, 219, 102], [-96, 170, 48], [-107, 166, 193], [25, 240, -9], [-19, 143, 146], [-61, 234, 16], [199, -225, 51], [183, 78, 254], [91, -100, -118], [-56, 184, 7], [-65, 187, 121], [-11, 48, -80], [-49, -105, 68], [35, 180, -154], [254, -2, -226], [178, 2, 71], [-229, -180, -198], [-110, -88, -40], [43, 117, 180], [-198, 114, -203], [114, -101, -63], [11, 156, -252], [-231, 246, 76], [236, 186, 52], [125, -28, 6], [192, -239, -37], [-19, 52, -204], [-39, 184, 196], [-198, 232, 223], [-47, -186, 203], [-21, 200, 68], [-44, -214, 109], [-30, 230, -160]], 
}

@lru_cache(maxsize=None)
def get_dataset(n: int = 256):
    if n not in (8, 16, 32, 64, 128, 256):
        raise ValueError(f"unsupported n={n}; expected one of 8, 16, 32, 64, 128, 256")
    data_file = Path(__file__).resolve().parent.parent / "SAT_data" / f"unique3sat_n{n}_2k.jsonl"
    return load_dataset("json", data_files=str(data_file), cache_dir="/dev/shm/datasets", split="train")

def get_prompt():
    return 'Find a satisfying assignment for the following 3-SAT formula.\n\nInput format:\nThe JSON array below is the list of clauses, with one three-integer clause per line. The outer array groups the clauses; each inner array groups the three literals of one clause.\n\nFormula and assignment semantics:\nThere are {n} Boolean variables, numbered 1 through {n}. Each variable has one value, either 0 (false) or 1 (true), shared by all of its occurrences in every clause.\nEach clause is an array of exactly three nonzero signed integers. Each integer has an absolute value from 1 through {n} and encodes a literal:\n- A positive integer k represents variable k itself; this literal is true exactly when variable k has value 1.\n- A negative integer -k represents the logical negation of variable k; this literal is true exactly when variable k has value 0.\nThe three literals within a clause are joined by OR: a clause is true when one, two, or all three of its literals are true.\nAll clauses are joined by AND: the same complete assignment must make every clause true simultaneously.\nThe supplied instance is satisfiable. Return one complete assignment that satisfies every clause.\n\n<clauses>\n{clauses}\n</clauses>\n\nFinal output format:\nThe entire final response must be one JSON object with exactly one key, "answer".\nThe first non-whitespace character must be \'{{\', and the last non-whitespace character must be \'}}\'.\nThe value of "answer" must be an array of exactly {n} integers, each either 0 or 1. Counting array positions from 1, position k gives the value of variable k, for k = 1 through {n}.\nInclude every variable in that order, including any variable absent from the clauses.\n'

def format_prompt(n: int, clauses: list):
    clause_rows = ",\n".join("  " + json.dumps(clause) for clause in clauses)
    return get_prompt().format(n=n, clauses="[\n" + clause_rows + "\n]")

def validate(clauses: list, assignment: list):
    # Check every clause is satisfied by the assignment
    for clause in clauses:
        satisfied = False
        for literal in clause:
            var_idx = abs(literal) - 1  # variables are 1-indexed
            if var_idx < 0 or var_idx >= len(assignment):
                return False
            val = assignment[var_idx]
            if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                satisfied = True
                break
        if not satisfied:
            return False
    return True
