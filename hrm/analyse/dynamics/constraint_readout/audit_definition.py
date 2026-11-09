"""Check the vectorized hard target against independent unit enumeration."""
from hrm.paths import analysis_directory
import json
from .run import hard, checks
from pathlib import Path
import numpy as np
import torch

root=analysis_directory(__file__)
config=json.loads((root/'run_config.json').read_text())
rng=np.random.default_rng(config['seed'])
boards=rng.integers(1,10,(32,81))
clues=rng.integers(0,10,(32,81))
expected=[]
for board, clue in zip(boards,clues):
    grid=board.reshape(9,9)
    bad=np.zeros((9,9),bool)
    for row in range(9):
        for col in range(9):
            digit=grid[row,col]
            box=grid[(row//3)*3:(row//3+1)*3,(col//3)*3:(col//3+1)*3]
            bad[row,col]=((grid[row]==digit).sum()>1 or (grid[:,col]==digit).sum()>1 or (box==digit).sum()>1)
    clue_bad=(clue>0)&(board!=clue)
    expected.append([bad.sum(),(bad.ravel()|clue_bad).sum(),clue_bad.sum()])
actual=torch.stack(hard(torch.tensor(boards),torch.tensor(clues)),-1).numpy()
assert np.array_equal(actual,np.array(expected))
checks()
result={'random_boards_checked':32,'independent_unit_enumeration_matches':True,
        'special_cases':'valid solution, single changed digit with overlapping constraints, all-equal board, legal relabeling that violates clues',
        'seed':config['seed']}
(root/'definition_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
