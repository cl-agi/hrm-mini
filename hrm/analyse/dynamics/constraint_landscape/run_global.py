"""One global coordinate-to-(H,L) map, with no phase or branch selection."""
from hrm.paths import repo_path, work_path
from pathlib import Path
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
import torch
OLD = Path(str(work_path('case6566_landscape')))
pass
from hrm.analyse.dynamics.constraint_landscape.evaluate_landscape import ROOT, decode, load_metadata, soft_metrics, sha256
D = 82 * 512
SQ = D ** 0.5

def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)

def source_data():
    src = ROOT / 'experiment/L4096_case_search_20260928/shortlist'
    meta = json.loads((src / 'summary.json').read_text())
    ci = next((i for i, x in enumerate(meta['selected']) if x['case'] == 6566))
    chunks, contexts = ([], [])
    for h in range(1, 6):
        bits = np.load(src / f'H{h:03d}_L_bits.npy', mmap_mode='r')
        chunks.append(torch.tensor(decode(bits[:, ci]).reshape(4097, D), device='cuda'))
        contexts.append(decode(np.load(src / f'H{h:03d}_context.npz')['H_before_bits'][ci]).reshape(D))
    return (src, meta, ci, torch.cat(chunks), torch.tensor(np.stack(contexts), device='cuda'))

def prepare(out):
    started = time.monotonic()
    src, meta, ci, L, H = source_data()
    original = np.load(OLD / 'case6566_L4096_arrays.npz')
    bn = np.stack([original['global_x_basis'], original['global_y_basis']], 1)
    B = torch.tensor(bn, device='cuda', dtype=torch.float32)
    origin = torch.tensor(original['origin'], device='cuda', dtype=torch.float32)
    q = np.concatenate([original[f'H{h}_global_xy'] for h in range(1, 6)])
    phase = np.repeat(np.arange(1, 6), 4097)
    step = np.tile(np.arange(4097), 5)
    mu = L.mean(0)
    print('GLOBAL_PCA_START', flush=True)
    with torch.no_grad():
        centered = (L - mu) / SQ
        _, sv, V = torch.pca_lowrank(centered, q=272, center=False, niter=3)
        V = V[:, :256]
        uq, rr = torch.linalg.qr(V - B @ (B.T @ V), mode='reduced')
        ur, sr, _ = torch.linalg.svd(rr, full_matrices=False)
        V = uq @ ur[:, sr > 0.0001]
        V = torch.linalg.qr(V - B @ (B.T @ V), mode='reduced').Q
        base = mu - B @ (B.T @ (mu - origin))
        lc = ((L - mu) @ V / SQ).cpu().numpy()
        hm = H.mean(0)
        _, hs, hvt = torch.linalg.svd((H - hm) / SQ, full_matrices=False)
        HV = hvt[:4].T
        hc = ((H - hm) @ HV / SQ).cpu().numpy()
    coeff = np.concatenate([lc, np.repeat(hc, 4097, axis=0)], axis=1)
    bins = np.clip(((q - q.min(0)) / np.maximum(np.ptp(q, axis=0), 1e-09) * 255).astype(int), 0, 255)
    _, inv, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    spatial = 1.0 / counts[inv]
    weights = 0.5 * np.ones(len(q)) / len(q) + 0.5 * spatial / spatial.sum()
    split = np.zeros(len(q), dtype=np.int8)
    split[step % 19 == 7] = 1
    split[step % 19 == 13] = 2
    split[(step == 0) | (step == 4096)] = 0
    np.savez_compressed(out / 'prepared.npz', q=q, coefficients=coeff, weights=weights, split=split, phase=phase, step=step, L_base=base.cpu().numpy(), L_V=V.cpu().numpy(), B=bn.astype(np.float32), origin=original['origin'].astype(np.float32), H_mean=hm.cpu().numpy(), H_V=HV.cpu().numpy(), H_contexts=H.cpu().numpy())
    oracle = []
    for start in range(0, len(q), 128):
        end = min(start + 128, len(q))
        pred = base + SQ * (torch.tensor(q[start:end], device='cuda', dtype=torch.float32) @ B.T + torch.tensor(lc[start:end], device='cuda') @ V.T)
        oracle.extend((pred - L[start:end]).square().mean(1).sqrt().cpu().numpy())
    report = dict(rank_L=int(V.shape[1]), rank_H=4, n=len(q), n_exact_unique_q=len(np.unique(q, axis=0)), pca='All observed states; transductive basis. Validation excludes regression labels only.', pca_L_reconstruction_rms=dict(mean=float(np.mean(oracle)), p95=float(np.quantile(oracle, 0.95))), H_basis_reconstruction_rms=float((hm + SQ * (torch.tensor(hc, device='cuda') @ HV.T) - H).square().mean().sqrt()), weights='0.5 uniform states + 0.5 uniform occupied 256x256 spatial bins; no phase labels used', split='Step modulo 19: validation=7, test=13; boundary duplicate samples retained in training. Near-trajectory interpolation only.', source=str(src), checkpoint=meta['model']['checkpoint_sha256'], prepare_seconds=time.monotonic() - started)
    (out / 'prepare_audit.json').write_text(json.dumps(report, default=serial, indent=2))
    print('PREPARED', json.dumps(report, default=serial), flush=True)

def run_method(out, method, steps):
    from hrm.analyse.dynamics.constraint_landscape.global_fit import fit_global
    started = time.monotonic()
    d = np.load(out / 'prepared.npz')
    q, coeff, weights, split = (d[k] for k in ['q', 'coefficients', 'weights', 'split'])
    tr, va, te = [np.flatnonzero(split == s) for s in [0, 1, 2]]
    vm, fit_report = fit_global(q, coeff, tr, va, weights=weights, method=method, device='cuda', max_steps=steps)
    validation_coeff = vm.predict(q[te])
    del vm
    fm, display_report = fit_global(q, coeff, np.arange(len(q)), np.array([], dtype=int), weights=weights, method=method, device='cuda', max_steps=steps, selected=fit_report)
    fm.save(out / f'{method}_model.pt')
    src, meta, ci, L, contexts = source_data()
    model, identity = load_metadata(argparse.Namespace(checkpoint=meta['model']['checkpoint'], config=meta['model']['config_source'], architecture='hrm', device='cuda'))
    model.eval().requires_grad_(False)
    assert identity['checkpoint_sha256'] == meta['model']['checkpoint_sha256']
    tensors = {k: torch.tensor(d[k], device='cuda', dtype=torch.float32) for k in ['L_base', 'L_V', 'B', 'origin', 'H_mean', 'H_V']}
    init = np.load(src / 'initial.npz')
    inp = torch.tensor(init['inputs'][ci], device='cuda')
    target = torch.tensor(init['targets'][ci], device='cuda')
    emb = model.embed(inp[None])
    rank = tensors['L_V'].shape[1]

    @torch.inference_mode()
    def lift(query, coefficients):
        qt = torch.tensor(query, device='cuda', dtype=torch.float32)
        ct = torch.tensor(coefficients, device='cuda', dtype=torch.float32)
        zL = tensors['L_base'] + SQ * (qt @ tensors['B'].T + ct[:, :rank] @ tensors['L_V'].T)
        zH = tensors['H_mean'] + SQ * (ct[:, rank:] @ tensors['H_V'].T)
        return (zL, zH)

    @torch.inference_mode()
    def measure(zL, zH):
        zl = zL.reshape(-1, 82, 512).to(torch.bfloat16)
        zh = zH.reshape(-1, 82, 512).to(torch.bfloat16)
        logits = model.lm_head(model.H_level(zh + zl))
        vals = soft_metrics(logits, inp, target)
        p = logits[:, 1:, 1:10].float().softmax(-1)
        vals['expected_errors'] = (1 - p.gather(-1, (target[1:] - 1)[None, :, None].expand(len(p), -1, -1)).squeeze(-1)).sum(-1)
        nxt = model.L_level(zl + zh + emb).float().flatten(1)
        delta = nxt - zl.float().flatten(1)
        flow = delta @ tensors['B'] / SQ
        vals.update(delta_rms=delta.square().mean(1).sqrt(), flow_x=flow[:, 0], flow_y=flow[:, 1])
        return ({k: v.cpu().numpy() for k, v in vals.items()}, nxt)
    report = dict(method=method, fit=fit_report, display_fit=display_report, checkpoint=identity['checkpoint_sha256'], no_phase_or_branch_inputs=True, script_sha256=sha256(__file__), validation={}, display={})
    replay = []
    with torch.inference_mode():
        for h in range(5):
            ix = np.array([0, 16, 127, 1024, 3073, 4095]) + h * 4097
            nxt = model.L_level(L[ix].reshape(-1, 82, 512).to(torch.bfloat16) + contexts[h].reshape(1, 82, 512).to(torch.bfloat16) + emb).float().flatten(1)
            replay.append(float((nxt - L[ix + 1]).abs().max()))
    report['native_replay_max_abs_error'] = max(replay)
    assert max(replay) == 0
    trajectory = {}
    for label, idx, prediction in [('validation', te, validation_coeff), ('display', np.arange(len(q)), fm.predict(q))]:
        metrics = {k: [] for k in ['L_rms', 'H_rms', 'expected_errors_abs', 'soft_conflict_abs', 'next_L_rms', 'projection_error', 'bf16_projection_error']}
        observed, predicted = ({}, {})
        for start in range(0, len(idx), 96):
            jj = idx[start:start + 96]
            zl, zh = lift(q[jj], prediction[start:start + len(jj)])
            true_h = contexts[torch.tensor(d['phase'][jj] - 1, device='cuda')]
            pm, pn = measure(zl, zh)
            tm, tn = measure(L[jj], true_h)
            values = dict(L_rms=(zl - L[jj]).square().mean(1).sqrt().cpu().numpy(), H_rms=(zh - true_h).square().mean(1).sqrt().cpu().numpy(), expected_errors_abs=np.abs(pm['expected_errors'] - tm['expected_errors']), soft_conflict_abs=np.abs(pm['soft_conflict'] - tm['soft_conflict']), next_L_rms=(pn - tn).square().mean(1).sqrt().cpu().numpy(), projection_error=np.linalg.norm(((zl - tensors['origin']) @ tensors['B'] / SQ).cpu().numpy() - q[jj], axis=1), bf16_projection_error=np.linalg.norm(((zl.to(torch.bfloat16).float() - tensors['origin']) @ tensors['B'] / SQ).cpu().numpy() - q[jj], axis=1))
            for k, v in values.items():
                metrics[k].extend(v)
            if label == 'display':
                for k, v in pm.items():
                    predicted.setdefault(k, []).append(v)
                for k, v in tm.items():
                    observed.setdefault(k, []).append(v)
        for k, v in metrics.items():
            v = np.asarray(v)
            report[label][k] = dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.quantile(v, 0.95)), max=float(v.max()))
        report[label]['per_phase'] = {}
        for h in range(1, 6):
            mask = d['phase'][idx] == h
            report[label]['per_phase'][str(h)] = {k: float(np.mean(np.asarray(v)[mask])) for k, v in metrics.items()}
        report[label]['early_vs_late'] = {name: {k: float(np.mean(np.asarray(v)[mask])) for k, v in metrics.items()} for name, mask in [('first_128', d['step'][idx] < 128), ('remaining', d['step'][idx] >= 128)]}
        if label == 'display':
            trajectory.update({f'trajectory_true_{k}': np.concatenate(v) for k, v in observed.items()})
            trajectory.update({f'trajectory_{method}_{k}': np.concatenate(v) for k, v in predicted.items()})
            trajectory.update({f'trajectory_{method}_{k}': np.asarray(v) for k, v in metrics.items()})
            boundary_indices = [j for h in range(1, 5) for j in [h * 4097 - 1, h * 4097]]
            report['boundary_samples'] = [dict(index=j, phase=int(d['phase'][j]), step=int(d['step'][j]), **{k: float(v[j]) for k, v in metrics.items()}) for j in boundary_indices]
            report['boundary_prediction_max_coefficient_difference'] = float(max((np.max(np.abs(prediction[h * 4097 - 1] - prediction[h * 4097])) for h in range(1, 5))))
        print('RECONSTRUCTION', method, label, json.dumps(report[label]), flush=True)
    dense = []
    for h in range(1, 6):
        trace = q[d['phase'] == h]
        for a, b in zip(trace[:-1], trace[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.001)))
            dense.append(a[None] + (b - a)[None] * (np.arange(n) / n)[:, None])
        dense.append(trace[-1:])
    dense = np.concatenate(dense)
    lo, hi = (q.min(0) - 0.05, q.max(0) + 0.05)
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 501), np.linspace(lo[1], hi[1], 451))
    grid = np.c_[xx.ravel(), yy.ravel()]
    dist = cKDTree(dense).query(grid)[0]
    valid = dist <= 0.045
    result = dict(x=xx, y=yy, distance=dist.reshape(xx.shape), valid=valid.reshape(xx.shape), observed_point_distance=cKDTree(q).query(grid)[0].reshape(xx.shape), trajectory_q=q, trajectory_phase=d['phase'], trajectory_step=d['step'], **trajectory)
    values = {}
    ix = np.flatnonzero(valid)
    for start in range(0, len(ix), 96):
        jj = ix[start:start + 96]
        zl, zh = lift(grid[jj], fm.predict(grid[jj]))
        metrics, _ = measure(zl, zh)
        for k, v in metrics.items():
            values.setdefault(k, np.full(len(grid), np.nan))[jj] = v
    for k, v in values.items():
        result[f'{method}_{k}'] = v.reshape(xx.shape)
    np.savez_compressed(out / f'{method}_fields.npz', **result)
    report.update(grid_shape=list(xx.shape), evaluated_grid_points=len(ix), band_radius=0.045, seconds=time.monotonic() - started)
    (out / f'{method}_audit.json').write_text(json.dumps(report, default=serial, indent=2))
    print('COMPLETE', method, report['seconds'], flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--method', choices=['kernel', 'fourier'])
    ap.add_argument('--steps', type=int, default=2500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(6566)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if args.prepare:
        prepare(args.out)
    else:
        run_method(args.out, args.method, args.steps)
