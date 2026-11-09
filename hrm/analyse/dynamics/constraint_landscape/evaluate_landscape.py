"""Evaluate the original HRM on explicit lifts of the archived 2-D projection.

No training, trajectory continuation, potential fitting, or target-based selection.
Each H phase keeps its recorded H fixed. The lift preserves the orthogonal
coordinates of the nearest recorded L, and changes only the two plotted axes.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from hrm.paths import work_path
ROOT = work_path('constraint_landscape')
pass
from hrm.analyse.dynamics.perturbation_controls.verify_test_time_scaling import load_metadata, sha256


def decode(bits):
    return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)


def soft_metrics(logits, inputs, target):
    """Expected equal-digit peer pairs, plus independently reported diagnostics."""
    p = logits[:, 1:, 1:10].float().softmax(-1).reshape(-1, 9, 9, 9)
    groups = [p, p.transpose(1, 2),
              p.reshape(-1, 3, 3, 3, 3, 9).permute(0, 1, 3, 2, 4, 5).reshape(-1, 9, 9, 9)]
    pair = sum(((g.sum(2).square() - g.square().sum(2)) * .5).sum((1, 2)) for g in groups)
    excess = sum((g.sum(2)-1).clamp_min(0).sum((1, 2)) for g in groups)
    flat = p.reshape(-1, 81, 9)
    clue = inputs[1:] > 0
    cp = flat[:, clue].gather(-1, (inputs[1:][clue]-1)[None, :, None].expand(len(p), -1, -1))
    clue_error = (1-cp.squeeze(-1)).mean(-1)
    logp = logits[:, 1:].float().log_softmax(-1)
    ce = -logp.gather(-1, target[1:][None, :, None].expand(len(p), -1, -1)).squeeze(-1).mean(-1)
    errors = (logits[:, 1:].argmax(-1) != target[None, 1:]).sum(-1)
    return dict(soft_conflict=pair, excess_error=excess, clue_error=clue_error, ce=ce, errors=errors)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arrays', type=Path, required=True)
    parser.add_argument('--nx', type=int, default=201)
    parser.add_argument('--ny', type=int, default=161)
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    source = ROOT/'experiment/L4096_case_search_20260928/shortlist'
    summary = json.loads((source/'summary.json').read_text())
    ci = next(i for i, row in enumerate(summary['selected']) if row['case'] == 6566)
    model, identity = load_metadata(argparse.Namespace(
        checkpoint=summary['model']['checkpoint'], config=summary['model']['config_source'],
        architecture='hrm', device='cuda:0'))
    assert identity['checkpoint_sha256'] == summary['model']['checkpoint_sha256']
    model.requires_grad_(False)
    d = np.load(args.arrays)
    basis = np.stack([d['global_x_basis'], d['global_y_basis']], -1)
    origin = d['origin']
    dim = len(origin)
    assert np.max(np.abs(basis.T@basis-np.eye(2))) < 1e-10
    bt = torch.tensor(basis, device='cuda', dtype=torch.float32)
    ot = torch.tensor(origin, device='cuda', dtype=torch.float32)
    initial = np.load(source/'initial.npz')
    inputs = torch.tensor(initial['inputs'][ci], device='cuda')
    target = torch.tensor(initial['targets'][ci], device='cuda')
    emb = model.embed(inputs[None])
    xyall = np.concatenate([d[f'H{h}_global_xy'] for h in range(1, 6)])
    lo, hi = xyall.min(0), xyall.max(0)
    lo, hi = lo-.18*(hi-lo), hi+.18*(hi-lo)
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], args.nx), np.linspace(lo[1], hi[1], args.ny))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    results = dict(x=xx, y=yy)
    audit = dict(checkpoint=identity, source=str(source), case=6566,
                 grid_shape=list(xx.shape), script_sha256=sha256(__file__),
                 lift='nearest recorded state within each phase; L*=L_i+sqrt(D)*B*(q-q_i); fixed H_before',
                 soft_conflict='sum expected equal-digit unordered pairs within each of 27 Sudoku units; row/box-overlapping pairs count in both units; probabilities normalized over digits 1..9',
                 decoding='lm_head(H_level(H_before + L*)); diagnostic H pass not committed',
                 target_usage='CE/errors are diagnostics only; targets never used in lift, phase selection, or soft_conflict',
                 phases=[])
    started = time.monotonic()
    for h in range(1, 6):
        raw = np.load(source/f'H{h:03d}_L_bits.npy', mmap_mode='r')
        states = torch.tensor(decode(raw[:, ci]).reshape(4097, -1), device='cuda')
        xy = d[f'H{h}_global_xy']
        reprojection = (states.double()-ot.double())@torch.tensor(basis, device='cuda')/dim**.5
        original_error = float((reprojection-torch.tensor(xy, device='cuda')).abs().max())
        assert original_error < 1e-10, original_error
        ctx = np.load(source/f'H{h:03d}_context.npz')
        fixed_h = torch.tensor(decode(ctx['H_before_bits'][ci]), device='cuda').to(torch.bfloat16)[None]
        distance, nearest = cKDTree(xy).query(grid)
        results[f'H{h}_distance'] = distance.reshape(xx.shape)
        results[f'H{h}_nearest_step'] = nearest.reshape(xx.shape)

        def evaluate(l):
            l = l.reshape(-1, 82, 512).to(torch.bfloat16)
            decoded_h = model.H_level(fixed_h+l)
            metrics = soft_metrics(model.lm_head(decoded_h), inputs, target)
            delta = model.L_level(l+fixed_h+emb).float()-l.float()
            metrics['L_residual'] = delta.square().mean((1, 2)).sqrt()
            metrics['flow_x'] = delta.flatten(1)@bt[:, 0]/dim**.5
            metrics['flow_y'] = delta.flatten(1)@bt[:, 1]/dim**.5
            return {k: v.cpu().numpy() for k, v in metrics.items()}, l

        chunks, max_pre, max_post = [], 0., 0.
        for start in range(0, len(grid), args.batch):
            ix = nearest[start:start+args.batch]
            shift = torch.tensor(grid[start:start+args.batch]-xy[ix], device='cuda', dtype=torch.float32)
            lifted = states[ix] + dim**.5*(shift@bt.T)
            projected = ((lifted-ot)@bt/dim**.5).cpu().numpy()
            max_pre = max(max_pre, float(np.max(np.abs(projected-grid[start:start+args.batch]))))
            values, rounded = evaluate(lifted)
            post = ((rounded.float().flatten(1)-ot)@bt/dim**.5).cpu().numpy()
            max_post = max(max_post, float(np.max(np.abs(post-grid[start:start+args.batch]))))
            chunks.append(values)
        for key in chunks[0]:
            results[f'H{h}_{key}'] = np.concatenate([v[key] for v in chunks]).reshape(xx.shape)
        # Actual states and terminal replay: verify semantics independently of synthetic grid.
        trajectory_chunks = []
        for start in range(0, len(states), args.batch):
            v, _ = evaluate(states[start:start+args.batch])
            trajectory_chunks.append(v)
        for key in trajectory_chunks[0]:
            results[f'H{h}_trajectory_{key}'] = np.concatenate([v[key] for v in trajectory_chunks])
        terminal = model.H_level(fixed_h+states[-1:].reshape(-1, 82, 512).to(torch.bfloat16))
        expected_h = torch.tensor(decode(ctx['H_after_bits'][ci]), device='cuda')
        pred = model.lm_head(terminal).argmax(-1)[0].cpu().numpy()
        assert np.array_equal(pred, ctx['predictions'][ci]), (h, pred.tolist())
        row = dict(H=h, original_projection_max_abs_error=original_error,
                   grid_projection_max_abs_error_float32=max_pre,
                   grid_projection_max_abs_error_bfloat16=max_post,
                   terminal_H_max_abs_difference=float((terminal.float()-expected_h).abs().max()),
                   terminal_prediction_matches_archive=True,
                   terminal_errors=int((pred[1:] != initial['targets'][ci, 1:]).sum()),
                   soft_conflict_range=[float(results[f'H{h}_soft_conflict'].min()), float(results[f'H{h}_soft_conflict'].max())],
                   terminal_soft_conflict=float(results[f'H{h}_trajectory_soft_conflict'][-1]))
        audit['phases'].append(row)
        print(json.dumps(row), flush=True)
        del states
    # Geometric atlas selection, not minimization of a model score.
    dist = np.stack([results[f'H{h}_distance'] for h in range(1, 6)])
    phase = dist.argmin(0)
    results['nearest_phase'] = phase+1
    results['nearest_distance'] = dist.min(0)
    results['phase_distance_gap'] = np.sort(dist, axis=0)[1]-dist.min(0)
    for key in chunks[0]:
        stack = np.stack([results[f'H{h}_{key}'] for h in range(1, 6)])
        results[f'atlas_{key}'] = np.take_along_axis(stack, phase[None], axis=0)[0]
    np.savez_compressed(args.out/'landscape.npz', **results)
    audit['elapsed_seconds'] = time.monotonic()-started
    audit['arrays_sha256'] = sha256(args.arrays)
    (args.out/'audit.json').write_text(json.dumps(audit, indent=2))
    print('COMPLETE', audit['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
