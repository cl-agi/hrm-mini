from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
"""Measure error readouts and state interventions on frozen HRM (CPU default).

Each observation precedes an actual L update. The extra H pass used to decode
the current joint state is diagnostic only and never modifies the trajectory.
All splits, preprocessing, and early stopping operate at puzzle level.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parent
pass
from hrm.analyse.dynamics.answer_error import common

def excess_error(prob):
    """Probability-mass excess, NOT expected pairwise conflicts."""
    p = prob.reshape(*prob.shape[:-2], 9, 9, 9)
    row = (p.sum(-2) - 1).clamp_min(0).sum((-2, -1))
    col = (p.sum(-3) - 1).clamp_min(0).sum((-2, -1))
    boxes = p.reshape(*p.shape[:-3], 3, 3, 3, 3, 9).sum((-4, -2))
    return row + col + (boxes - 1).clamp_min(0).sum((-3, -2, -1))

def corr(a, b):
    a, b = (np.asarray(a).ravel(), np.asarray(b).ravel())
    if min(a.std(), b.std()) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])

def summary(y, pred, delta):
    y, pred, delta = [np.asarray(v, dtype=np.float64) for v in (y, pred, delta)]
    return {'pearson_error': corr(y, pred), 'r2_error': float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)), 'rmse_error': float(np.mean((y - pred) ** 2) ** 0.5), 'pearson_next_L': corr(pred, delta), 'pearson_error_within_time': corr(y - y.mean(0), pred - pred.mean(0)), 'pearson_next_L_within_time': corr(pred - pred.mean(0), delta - delta.mean(0))}

@torch.inference_mode()
def collect(args, out):
    model, identity = common.model_load('hrm', args.model_seed, device=args.device)
    raw = list(csv.DictReader((repo_path('data/sudoku/sudoku-extreme-1k/test_hard.csv')).open()))
    training = list(csv.DictReader((repo_path('data/sudoku/sudoku-extreme-1k/train.csv')).open()))
    ids = np.random.default_rng(args.seed).permutation(len(raw))[:args.puzzles]
    questions = [raw[i]['question'] for i in ids]
    assert len(set(questions)) == len(questions)
    assert not set(questions) & {r['question'] for r in training}
    x = torch.tensor([[0] + [int(c) for c in q.replace('.', '0')] for q in questions])
    y = torch.tensor([[0] + [int(c) for c in raw[i]['answer']] for i in ids])
    parts = []
    start = time.monotonic()
    max_diff = 0.0
    for begin in range(0, len(x), args.batch_size):
        bx = x[begin:begin + args.batch_size].to(args.device)
        emb = model.embed(bx)
        h, l = (model.zH_init.expand_as(emb).clone(), model.zL_init.expand_as(emb).clone())
        initial = {'z_H': h.clone(), 'z_L': l.clone()}
        feats, errors, errors10, changes = ([], [], [], [])
        for outer in range(args.cycles):
            for hc in range(model.H_cycles):
                for lc in range(model.L_cycles):
                    feats.append(torch.cat([l.mean(1), h.mean(1)], -1).cpu())
                    logits = model.lm_head(model.H_level(h + l))[:, 1:].float()
                    errors.append(excess_error(logits[..., 1:10].softmax(-1)).cpu())
                    errors10.append(excess_error(logits.softmax(-1)[..., 1:10]).cpu())
                    next_l = model.L_level(l + h + emb)
                    changes.append((next_l - l).float().flatten(1).norm(dim=-1).cpu())
                    l = next_l
                h = model.H_level(h + l)
            if outer == 0:
                native, _ = model(initial, bx)
                diff = max(float((h - native['z_H']).abs().max()), float((l - native['z_L']).abs().max()))
                max_diff = max(max_diff, diff)
                assert diff < 1e-05, diff
        pred = model.lm_head(h).argmax(-1).cpu()
        parts.append({'features': torch.stack(feats, 1), 'error': torch.stack(errors, 1), 'error_10class': torch.stack(errors10, 1), 'delta': torch.stack(changes, 1), 'final_exact': pred[:, 1:].eq(y[begin:begin + len(bx), 1:]).all(-1)})
        print(json.dumps({'phase': 'collect', 'puzzles': begin + len(bx), 'seconds': round(time.monotonic() - start, 2)}), flush=True)
    data = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    data.update(ids=torch.tensor(ids), inputs=x, targets=y)
    torch.save(data, out / 'trajectory.pt')
    manifest = {'args': vars(args), 'checkpoint': identity, 'native_manual_max_abs_diff': max_diff, 'model_frozen': all((not p.requires_grad for p in model.parameters())), 'collection_seconds': time.monotonic() - start, 'observation': 'before each actual L update, extra H decoder is not committed', 'error': 'sum ReLU(digit mass - 1) over 9 rows, 9 columns, 9 boxes, all 9 digits', 'digit_normalization': 'primary: softmax over digits 1..9; sensitivity: softmax over 0..9 then discard 0', 'pooling': 'mean over 82 positions including CLS, concatenate L then H', 'split': [args.puzzles // 2, args.puzzles // 4, args.puzzles - args.puzzles // 2 - args.puzzles // 4], 'data_scope': 'existing development test_hard data, no exact original training-question overlap', 'script_sha256': common.sha(__file__), 'data_sha256': common.sha(repo_path('data/sudoku/sudoku-extreme-1k/test_hard.csv')), 'trajectory_sha256': common.sha(out / 'trajectory.pt')}
    common.js(out / 'manifest.json', manifest)
    return data

def fit_linear(x, y):
    mu, ym = (x.double().mean(0), y.double().mean())
    w = torch.linalg.lstsq(x.double() - mu, (y.double() - ym)[:, None], driver='gelsd').solution[:, 0]
    return (w, ym - mu @ w)

def fit_mlp(x, y, valx, valy, seed, epochs):
    torch.manual_seed(seed)
    mu, sd = (x.mean(0), x.std(0).clamp_min(1e-05))
    ym, ys = (y.mean(), y.std().clamp_min(1e-05))
    x, valx = ((x - mu) / sd, (valx - mu) / sd)
    y, valy = ((y - ym) / ys, (valy - ym) / ys)
    head = nn.Sequential(nn.Linear(x.shape[-1], 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
    best, stale, chosen = (float('inf'), 0, 0)
    for epoch in range(epochs):
        for idx in torch.randperm(len(x)).split(2048):
            loss = (head(x[idx]).squeeze(-1) - y[idx]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            loss = float((head(valx).squeeze(-1) - valy).square().mean())
        if loss < best:
            best, stale, chosen = (loss, 0, epoch + 1)
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= 15:
            break
    head.load_state_dict(state)
    return (head, {'mean': mu, 'scale': sd, 'target_mean': ym, 'target_scale': ys, 'state_dict': state, 'epoch': chosen, 'validation_standardized_mse': best})

def analyze(args, out, data):
    n, steps, width = data['features'].shape
    a, b = (n // 2, n // 2 + n // 4)
    slices = {'train': slice(0, a), 'validation': slice(a, b), 'test': slice(b, n)}
    x = data['features'].float()
    y, delta = (data['error'].numpy(), data['delta'].numpy())
    flatx = x[:a].flatten(0, 1)
    flaty = data['error'][:a].flatten()
    predictions, fits = ({}, {})
    for label, xx in [('linear_HL', x), ('linear_L', x[..., :width // 2]), ('linear_H', x[..., width // 2:])]:
        w, bias = fit_linear(xx[:a].flatten(0, 1), flaty)
        predictions[label] = (xx.double() @ w + bias).numpy()
        fits[label] = {'weight': w, 'bias': bias}
    predictions['time_only'] = np.broadcast_to(y[:a].mean(0), y.shape).copy()
    gen = torch.Generator().manual_seed(args.seed)
    shuffled = data['error'][:a][torch.randperm(a, generator=gen)].flatten()
    w, bias = fit_linear(flatx, shuffled)
    predictions['puzzle_shuffled_linear'] = (x.double() @ w + bias).numpy()
    fits['puzzle_shuffled_linear'] = {'weight': w, 'bias': bias}
    for seed in [1, 2, 3]:
        head, fit = fit_mlp(flatx, flaty, x[a:b].flatten(0, 1), data['error'][a:b].flatten(), seed, args.epochs)
        with torch.no_grad():
            pred = torch.cat([head((chunk - fit['mean']) / fit['scale']).squeeze(-1) for chunk in x.flatten(0, 1).split(4096)])
        predictions[f'mlp_seed{seed}'] = (pred * fit['target_scale'] + fit['target_mean']).reshape(n, steps).numpy()
        fits[f'mlp_seed{seed}'] = fit
        print(json.dumps({'phase': 'mlp', 'seed': seed, 'chosen_epoch': fit['epoch']}), flush=True)
    w10, b10 = fit_linear(flatx, data['error_10class'][:a].flatten())
    pred10 = (x.double() @ w10 + b10).numpy()
    fits['linear_10class'] = {'weight': w10, 'bias': b10}
    rows = []
    for label, pred in predictions.items():
        for split, sl in slices.items():
            rows.append({'method': label, 'split': split, **summary(y[sl], pred[sl], delta[sl])})
    common.csvout(out / 'metrics.csv', rows)
    report = {'metrics': rows, 'reference_metrics': {'pearson_error': 0.9915, 'pearson_next_L': 0.8511}, 'native_final_full81_exact': float(data['final_exact'][b:].float().mean()), 'normalization_sensitivity': summary(data['error_10class'][b:].numpy(), pred10[b:], delta[b:]), 'error_target_next_L': corr(y[b:], delta[b:]), 'error_10class_vs_9class': corr(data['error_10class'][b:].numpy(), y[b:]), 'without_initial_state': summary(y[b:, 1:], predictions['linear_HL'][b:, 1:], delta[b:, 1:]), 'bootstrap': {}, 'time_windows': {}}
    rng = np.random.default_rng(args.seed + 1)
    for label in ['linear_HL', 'mlp_seed1', 'time_only']:
        vals = []
        for _ in range(500):
            idx = rng.integers(b, n, size=n - b)
            vals.append([corr(y[idx], predictions[label][idx]), corr(predictions[label][idx], delta[idx])])
        report['bootstrap'][label] = dict(zip(['pearson_error_95ci', 'pearson_next_L_95ci'], np.quantile(vals, [0.025, 0.975], axis=0).T.tolist()))
    for label in ['linear_HL', 'mlp_seed1', 'time_only']:
        report['time_windows'][label] = {f'{lo}:{hi}': summary(y[b:, lo:hi], predictions[label][b:, lo:hi], delta[b:, lo:hi]) for lo, hi in [(0, steps // 4), (steps // 4, steps // 2), (steps // 2, steps)]}
    torch.save(fits, out / 'probes.pt')
    np.savez_compressed(out / 'predictions.npz', **predictions, true_error=y, next_L=delta, row_ids=data['ids'].numpy(), split=np.r_[np.zeros(a), np.ones(b - a), np.full(n - b, 2)])
    common.js(out / 'results.json', report)
    print(json.dumps({'phase': 'complete', 'test': [r for r in rows if r['split'] == 'test']}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/internal_critic_20261004')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--puzzles', type=int, default=512)
    parser.add_argument('--cycles', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--model-seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only and (out / 'trajectory.pt').exists():
        raise FileExistsError('Use a new output directory or --analyze-only; existing traces are preserved.')
    if args.puzzles < 8 or args.cycles < 1:
        raise ValueError('Need at least 8 puzzles and 1 cycle.')
    data = torch.load(out / 'trajectory.pt', weights_only=True) if args.analyze_only else collect(args, out)
    analyze(args, out, data)
if __name__ == '__main__':
    main()
