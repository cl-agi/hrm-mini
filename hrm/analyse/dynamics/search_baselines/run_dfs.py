"""Generate E31 minimum-remaining-values DFS board histories from clues only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from hrm.paths import repository_root
from hrm.training.data import local_directory,puzzle_directory
from hrm.analyse.dynamics.jacobian.data import read_test_records,resolve_test_csv
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .run_sa import check_puzzle


def legal_candidates(board,cell):
    if board[cell]:return (int(board[cell]),)
    row,column=divmod(cell,9);used=set(board[row*9:row*9+9]);used.update(board[column::9])
    br,bc=3*(row//3),3*(column//3)
    used.update(board[(br+dr)*9+bc+dc] for dr in range(3) for dc in range(3))
    return tuple(digit for digit in range(1,10) if digit not in used)


def choose_mrv_cell(board):
    best_cell=None;best=()
    for cell,value in enumerate(board):
        if value:continue
        candidates=legal_candidates(board,cell)
        if not candidates:return cell,()
        if best_cell is None or len(candidates)<len(best):best_cell,best=cell,candidates
    return best_cell,best


def solve(puzzle,max_states=1000000):
    check_puzzle(puzzle)
    if max_states<1:raise ValueError('Positive state budget required')
    board=puzzle.copy();history=[board.copy()];actions=['initial']
    class BudgetExceeded(Exception):pass
    def record(action):
        if len(history)>=max_states:raise BudgetExceeded
        history.append(board.copy());actions.append(action)
    def visit():
        cell,options=choose_mrv_cell(board)
        if cell is None:return True
        if not options:record('dead_end');return False
        for digit in options:
            board[cell]=digit;record('assign')
            if visit():return True
            board[cell]=0;record('undo')
        return False
    try:status='solved' if visit() else 'unsatisfiable'
    except BudgetExceeded:status='state_budget_exhausted'
    return np.asarray(history,dtype=np.uint8),np.asarray(actions),status


def sampled_distance(boards,frames=99):
    times=np.arange(len(boards));targets=np.linspace(0,len(boards)-1,frames)
    indices=np.array([np.argmin(np.abs(times-t)) for t in targets])
    sampled=boards[indices]
    return indices,(sampled[:,None,:]!=sampled[None,:,:]).mean(-1)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/search'))
    p.add_argument('--dataset',type=Path);p.add_argument('--board-id',type=int,default=2)
    p.add_argument('--max-states',type=int,default=1000000);p.add_argument('--output-dir',type=Path,default=root/'dfs')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if a.max_states<1:p.error('Positive state budget required')
    if (a.output_dir/'started.json').exists():raise FileExistsError(a.output_dir)
    data=local_directory(a.dataset) if a.dataset else puzzle_directory('sudoku');records=read_test_records(data)
    if not 0<=a.board_id<len(records):raise ValueError('Board ID outside the test split')
    record=records[a.board_id];puzzle=np.array([int(c) if c!='.' else 0 for c in record.question],dtype=np.int64)
    check_puzzle(puzzle)
    plan=dict(board_id=a.board_id,puzzle=record.question,target=record.answer,dataset_sha256=sha(resolve_test_csv(data)),
        max_states=a.max_states,frames=99,cell_selection='MRV; first row-major cell on ties; immediate dead-end if any cell has no candidate',
        digit_order='ascending 1..9',record_events=['initial','assign','dead_end','undo'],code_sha256=sha(Path(__file__)))
    plan_path=a.output_dir/'run_config.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Prepared settings changed; use a new directory')
    else:
        if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
        a.output_dir.mkdir(parents=True,exist_ok=True);plan_path.write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (a.output_dir/'started.json').open('x') as f:json.dump(dict(plan_sha256=sha(plan_path)),f)
    boards,actions,status=solve(puzzle,a.max_states)
    if not np.all(boards[:,puzzle!=0]==puzzle[puzzle!=0]):raise ValueError('Given clues changed')
    if status=='solved':
        check_puzzle(boards[-1])
        if np.any(boards[-1]==0):raise ValueError('Incomplete purported solution')
    indices,distance=sampled_distance(boards)
    np.savez_compressed(a.output_dir/'trajectory.npz',puzzle=puzzle,boards=boards,operations=actions,
        sampled_indices=indices,distance=distance)
    report=dict(status='complete',solver_status=status,states=len(boards),operations=len(boards)-1,
        unique_sampled_frames=len(set(indices.tolist())),matches_target=bool(''.join(map(str,boards[-1]))==record.answer),
        window_ends_at_solution=status=='solved',plan_sha256=sha(plan_path),artifacts={'trajectory.npz':sha(a.output_dir/'trajectory.npz')},
        note='Deterministic clues-only MRV search; reference answer is used only after search. Budget-censored traces are not labelled solved.')
    (a.output_dir/'manifest.json').write_text(json.dumps(report,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
