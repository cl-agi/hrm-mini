# 3-sat dataset
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
    return """You are a Boolean satisfiability (3-SAT) solver.

You are given a 3-SAT problem with {n} Boolean variables (numbered 1 to {n}) and a set of clauses. Each clause contains exactly 3 literals. A positive integer k means variable k is TRUE; a negative integer -k means variable k is FALSE. A clause is satisfied if at least one of its literals is true. Your task is to find an assignment of all variables that satisfies every clause simultaneously.

The clauses are:
{clauses}

Respond with a JSON object containing the assignment as a list of {n} values (0 or 1), where the i-th element (0-indexed) is the value for variable (i+1). 1 means TRUE, 0 means FALSE.

{{"answer": [0, 1, 0, ...]}}
"""

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
