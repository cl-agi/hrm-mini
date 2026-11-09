"""Compare one H versus one L at identical recorded L6 states."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from hrm.paths import repository_root
from .analyze import paired_records, sha


def peer_pairs():
    r, c = np.divmod(np.arange(81), 9)
    box = r // 3 * 3 + c // 3
    peer = (r[:, None] == r) | (c[:, None] == c) | (box[:, None] == box)
    return np.where(np.triu(peer, 1))


def conflicts(boards):
    i, j = peer_pairs()
    return ((boards[:, i] == boards[:, j]) & (boards[:, i] > 0)).sum(1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get('HRM_ANALYSIS_RECORDS', repository_root() / 'exp/dynamics/records/answer_direction'))
    p.add_argument('--records', type=Path, default=root / 'intervention')
    p.add_argument('--readout-records', type=Path, default=root / 'readout')
    p.add_argument('--output-dir', type=Path, default=root / 'single_step')
    a = p.parse_args(argv)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    meta, branches, trajectory, idx, y, blank, eligible, files = paired_records(a.records, a.readout_records)
    if not np.all(conflicts(y) == 0):
        raise ValueError('Reference targets contain Sudoku conflicts')
    before = branches['start'].argmax(-1)
    df = pd.DataFrame(dict(case=branches['ids'], cycle=branches['cycles']))
    for name in ('H_only', 'L_only'):
        z = branches[name].argmax(-1)
        changed = z != before
        count = changed.sum(1)
        df[name + '_k'] = count
        df[name + '_conf'] = conflicts(before) - conflicts(z)
        df[name + '_correct_fraction'] = np.divide(
            (changed & (z == y)).sum(1), count,
            out=np.full(len(count), np.nan), where=count > 0)
        df[name + '_error_reduction'] = (before != y).sum(1) - (z != y).sum(1)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir / 'started.json').open('x') as f:
        json.dump(dict(readout_manifest_sha256=sha(a.readout_records / 'manifest.json'),
                       intervention_manifest_sha256=sha(a.records / 'manifest.json')), f, indent=2)
    table = a.output_dir / 'same_state_single_step.csv'
    df.to_csv(table, index=False)
    checks = dict(status='complete', selected=meta['selected'], seed=meta['seed'],
        states=len(df), puzzles=int(df.case.nunique()), matched_board_checks=4 * len(df),
        all_selected_readout_branches_match=True, unique_peer_pairs=len(peer_pairs()[0]),
        all_targets_zero_conflicts=True, all_states_unsolved_at_cycle_entry=bool(eligible.all()),
        predicted_zero_cells={k:int((branches[k].argmax(-1) == 0).sum()) for k in ('start','H_only','L_only')},
        sources=[dict(path=str(f),sha256=sha(f)) for f in files],
        table_sha256=sha(table), code_sha256=sha(Path(__file__)),
        definition='Count each unordered Sudoku peer pair once (810 pairs); equal nonzero predictions conflict. Raw predictions, no clue clamping. Correct fraction uses only changed cells, undefined if none changed.')
    (a.output_dir / 'branch_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
