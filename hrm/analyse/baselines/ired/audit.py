"""Check dataset validity and demonstrate the upstream metric counterexample."""
import json
import torch
from torch.utils.data import DataLoader
from hrm.analyse.baselines.ired.run import ROOT, metrics, Extreme
from hrm_ired.sat_dataset import SudokuDataset, SudokuRRNDataset

torch.set_num_threads(4)
ds = SudokuDataset('sudoku', 'val')
x, y, mask = [a[None].float() for a in ds[0]]
assert metrics(y, y, x, mask)['strict_valid_and_clues'] == 1
invalid = torch.full_like(y, -1).reshape(-1, 9, 9, 9)
invalid[..., 4] = 1
r = metrics(invalid, y, x, mask)
assert r['board_accuracy'] == 1 and r['strict_valid_and_clues'] == 0 and r['exact_match'] == 0
(ROOT/'metric_audit.json').write_text(json.dumps({
    'counterexample': 'All 81 cells equal digit 5 evaluated against a real SAT-Net puzzle.',
    'metrics': r}, indent=2))
report = {}
for name, d in [('satnet_train', SudokuDataset('sudoku', 'train')),
                ('satnet_test', ds), ('rrn_test', SudokuRRNDataset('sudoku-rrn', 'test')),
                ('extreme_test_hard', Extreme())]:
    n = valid = 0
    for x, y, mask in DataLoader(d, batch_size=256):
        n += len(x)
        valid += round(metrics(y.float(), y.float(), x.float(), mask.float())['strict_valid_and_clues']*len(x))
    report[name] = {'n': n, 'valid_labels_preserving_clues': valid}
    assert valid == n, (name, valid, n)
(ROOT/'data_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
