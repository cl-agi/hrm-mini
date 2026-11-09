"""Fit a same-trajectory Mixer constraint probe using training time blocks only."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from hrm.paths import repository_root
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .replay_selected import read_screen, analysis_h_steps


def count(logits):
    b = logits.argmax(-1)[..., 1:].reshape(-1, 9, 9)
    o = torch.nn.functional.one_hot(b.long(), 10)[..., 1:]
    return ((o.sum(1) - 1).clamp_min(0).sum((1, 2)) +
            (o.sum(2) - 1).clamp_min(0).sum((1, 2)) +
            (o.reshape(-1, 3, 3, 3, 3, 9).sum((2, 4)) - 1).clamp_min(0).sum((1, 2, 3))).float()


def temporal_split(steps, l_steps, block_size=128):
    if l_steps < 1 or block_size < 1:
        raise ValueError('Positive L and block sizes required')
    # The final endpoint belongs to the preceding block (4096 -> block31).
    block = np.minimum(np.asarray(steps)//block_size, (l_steps-1)//block_size)
    return np.where(block % 5 == 3, 1, np.where(block % 5 == 4, 2, 0))


def metric(target, predicted):
    if not len(target):
        raise ValueError('Empty metric split')
    residual = target-predicted
    variance = ((target-target.mean())**2).sum()
    pearson = None
    if len(target) > 1 and np.std(target) > 0 and np.std(predicted) > 0:
        pearson = float(np.corrcoef(target, predicted)[0, 1])
    return dict(n=len(target), pearson=pearson, rmse=float(np.sqrt(np.mean(residual**2))),
                mae=float(np.mean(np.abs(residual))),
                r2=float(1-(residual**2).sum()/variance) if variance > 0 else None)


def fit(features, target, split):
    features, target = np.asarray(features, np.float64), np.asarray(target, np.float64)
    split = np.asarray(split)
    if features.ndim != 2 or target.shape != (len(features),) or split.shape != target.shape:
        raise ValueError('Invalid feature/target/split shapes')
    if not np.isfinite(features).all() or not np.isfinite(target).all() or set(split.tolist()) != {0, 1, 2}:
        raise ValueError('Finite data and nonempty train/validation/test splits required')
    x = torch.from_numpy(features[split == 0]); y = torch.from_numpy(target[split == 0])
    mean, ym = x.mean(0), y.mean()
    solution = torch.linalg.lstsq(x-mean, (y-ym)[:, None], driver='gelsd')
    weight = solution.solution[:, 0].numpy()
    bias = float(ym-mean@solution.solution[:, 0])
    predicted = features@weight+bias
    return weight, bias, predicted, dict(rank=int(solution.rank),
        singular_values_range=[float(solution.singular_values.min()), float(solution.singular_values.max())],
        metrics={name: metric(target[split == i], predicted[split == i]) for i, name in
                 enumerate(('train', 'validation', 'test'))})


@torch.inference_mode()
def run(records, output, screen_records=None, checkpoint=None, device=None, batch_size=128, block_size=128):
    records, output = Path(records), Path(output)
    if batch_size < 1 or block_size < 1:
        raise ValueError('Positive batch/block sizes required')
    replay = json.loads((records/'manifest.json').read_text())
    frozen = json.loads((records/'run_config.json').read_text())
    if replay.get('status') != 'complete' or any(replay.get(k) != v for k, v in frozen.items()):
        raise ValueError('Replay is incomplete or differs from its plan')
    source = Path(screen_records or replay['screen_records'])
    screen, cohort, _ = read_screen(source)
    if sha(source/'manifest.json') != replay['source_manifest_sha256']:
        raise ValueError('Screen differs from replay source')
    if sha(records/'cohort.npz') != replay['cohort_sha256']:
        raise ValueError('Replay cohort changed')
    checkpoint = Path(checkpoint or replay['checkpoint'])
    if sha(checkpoint) != replay['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    h_steps = analysis_h_steps(replay)
    steps = np.tile(np.arange(replay['l_steps']+1), h_steps)
    phases = np.repeat(np.arange(1, h_steps+1), replay['l_steps']+1)
    split = temporal_split(steps, replay['l_steps'], block_size)
    if set(split.tolist()) != {0, 1, 2}:
        raise ValueError('Trajectory too short for nonempty train/validation/test time blocks')
    files = []
    bits = replay['storage'] == 'uint16 BF16 bits'
    for h in range(1, h_steps+1):
        name = f'H{h:03d}_case{replay["case"]}' + ('_bits.npy' if bits else '.npy')
        if sha(records/name) != replay['trajectory_sha256'][name]:
            raise ValueError(f'Trajectory changed: {name}')
        files.append(records/name)
    if output.exists():
        raise FileExistsError(output)
    device = device or replay['device']
    arch = analysis_architecture(replay['config']); arch.pop('name', None)
    model = HRM(arch).to(device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)), strict=True)
    output.mkdir(parents=True)
    plan = dict(replay_manifest_sha256=sha(records/'manifest.json'), checkpoint_sha256=sha(checkpoint),
                seed=replay['seed'], case=replay['case'], selection=replay['selection'],
                h_steps=h_steps, batch_size=batch_size, block_size=block_size, device=device, code_sha256=sha(Path(__file__)))
    (output/'run_config.json').write_text(json.dumps(plan, indent=2))
    index = cohort['ids'].tolist().index(replay['case'])
    features, targets = [], []
    for h, path in enumerate(files, 1):
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        if raw.shape != (replay['l_steps']+1, 82, arch['hidden_size']) or raw.dtype != (np.uint16 if bits else np.float32):
            raise ValueError('Invalid trajectory shape/dtype')
        reference = torch.load(source/f'H{h:03d}.pt', map_location='cpu', weights_only=True)
        context = reference['H_before'][index].to(device)
        for start in range(0, len(raw), batch_size):
            chunk = np.array(raw[start:start+batch_size], copy=True)
            states = torch.from_numpy(chunk)
            if bits:
                states = states.view(torch.bfloat16)
            states = states.to(device)
            features.append(states.float().mean(1).cpu().numpy())
            targets.append(count(model.lm_head(model.H_level(context+states))).cpu().numpy())
    features = np.concatenate(features).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    weight, bias, predicted, report = fit(features, target, split)
    np.savez_compressed(output/'probe.npz', weight=weight, bias=bias, features=features, target=target,
                        predicted=predicted, phase=phases, step=steps, split=split)
    report.update(plan, status='complete', probe_sha256=sha(output/'probe.npz'),
        per_phase_test={str(h): metric(target[(phases == h)&(split == 2)], predicted[(phases == h)&(split == 2)])
                        for h in range(1, h_steps+1)},
        fit='Unregularized float64 OLS; token-mean L including BOS; training blocks only, no full-data refit',
        target='Raw argmax duplicate excess across 27 units, digits 1..9; no clue clamping; native H-lookahead',
        limitation='Temporal holdout within one correlated trajectory, not independent-puzzle generalization. No off-trajectory landscape is generated.')
    (output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main(argv=None):
    root = Path(os.environ.get('HRM_ANALYSIS_RECORDS', repository_root()/'exp/dynamics/records/mixer'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, default=root/'selected')
    p.add_argument('--output-dir', type=Path, default=root/'probe')
    p.add_argument('--screen-records', type=Path); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device'); p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--block-size', type=int, default=128)
    a = p.parse_args(argv)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps(run(a.records, a.output_dir, a.screen_records, a.checkpoint, a.device,
                         a.batch_size, a.block_size), indent=2))


if __name__ == '__main__':
    main()
