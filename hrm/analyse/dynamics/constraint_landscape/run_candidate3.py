"""Refit one global latent kernel in candidate3 coordinates and apply frozen L probe.

No phase input, gate, critic refit, scalar fitting, or terrain postprocessing.
The exact native states supply coefficient labels and reconstruction references.
"""
from hrm.paths import repo_path, work_path
from pathlib import Path
from hrm.paths import repo_path
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(str(work_path('case6566_global_lift')))
pass
from hrm.analyse.dynamics.constraint_landscape.global_fit import fit_global
D = 82 * 512
SQ = D ** 0.5

def decode(x):
    return (np.asarray(x).astype(np.uint32) << 16).view(np.float32)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def summary(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), rmse=float(np.sqrt(np.mean(x * x))), median=float(np.median(x)), p95=float(np.quantile(x, 0.95)), max=float(x.max()))

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()
    start = time.monotonic()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(6566)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    old_path = ROOT / 'results/prepared.npz'
    view_path = ROOT / 'projection_view/candidate3.npz'
    old = np.load(old_path)
    view = np.load(view_path)
    archive = Path(str(work_path('L4096_case_search/shortlist')))
    source_audit = json.loads((archive / 'summary.json').read_text())
    ci = next((i for i, x in enumerate(source_audit['selected']) if x['case'] == 6566))
    probe_path = Path(str(repo_path('checkpoint/dynamics/case6566_landscape/critic_result/probes.pt')))
    probe = torch.load(probe_path, map_location='cpu', weights_only=True)
    probe_audit = json.loads((probe_path.parent / 'critic_audit.json').read_text())
    old_audit = json.loads((ROOT / 'results/prepare_audit.json').read_text())
    checkpoint = probe['checkpoint_sha256']
    assert checkpoint == old_audit['checkpoint'] == source_audit['model']['checkpoint_sha256']
    assert sha(probe_audit['checkpoint']['checkpoint']) == checkpoint
    weight = probe['linear_L']['weight'].to('cuda', dtype=torch.float64)
    bias = probe['linear_L']['bias'].to('cuda', dtype=torch.float64)
    assert weight.shape == (512,) and bias.shape == ()
    chunks = []
    contexts = []
    for h in range(1, 6):
        raw = np.load(archive / f'H{h:03d}_L_bits.npy', mmap_mode='r')
        chunks.append(torch.tensor(decode(raw[:, ci]).reshape(4097, D), device='cuda'))
        contexts.append(decode(np.load(archive / f'H{h:03d}_context.npz')['H_before_bits'][ci]).reshape(D))
    L = torch.cat(chunks)
    del chunks
    H = torch.tensor(np.stack(contexts), device='cuda')
    phase = old['phase']
    step = old['step']
    split = old['split']
    q = view['xy'].astype(np.float64)
    n = len(q)
    origin64 = torch.tensor(view['origin'], device='cuda', dtype=torch.float64)
    B64 = torch.tensor(view['basis'], device='cuda', dtype=torch.float64)
    U = torch.tensor(np.column_stack([old['B'], old['L_V']]), device='cuda', dtype=torch.float64)
    g, v = torch.linalg.eigh(U.T @ U)
    S = U @ (v * g.rsqrt()[None] @ v.T)
    bcoords = S.T @ B64
    complement = torch.linalg.qr(bcoords, mode='complete').Q[:, 2:]
    V64 = S @ complement
    mu = L.mean(0, dtype=torch.float64)
    base64 = mu - B64 @ (B64.T @ (mu - origin64))
    normal_rank = V64.shape[1]
    coefficients = np.empty((n, normal_rank + 4), dtype=np.float32)
    actual_q = np.empty_like(q)
    for begin in range(0, n, 256):
        states = L[begin:begin + 256].double()
        coefficients[begin:begin + len(states), :normal_rank] = ((states - mu) @ V64 / SQ).float().cpu().numpy()
        actual_q[begin:begin + len(states)] = ((states - origin64) @ B64 / SQ).cpu().numpy()
    actual_q_error = np.linalg.norm(actual_q - q, axis=1)
    assert actual_q_error.max() < 1e-07
    hmean = torch.tensor(old['H_mean'], device='cuda', dtype=torch.float64)
    hv = torch.tensor(old['H_V'], device='cuda', dtype=torch.float64)
    hc = ((H.double() - hmean) @ hv / SQ).float().cpu().numpy()
    coefficients[:, normal_rank:] = np.repeat(hc, 4097, axis=0)
    bins = np.clip(((q - q.min(0)) / np.maximum(np.ptp(q, axis=0), 1e-09) * 255).astype(int), 0, 255)
    _, inv, count = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    spatial = 1.0 / count[inv]
    weights = 0.5 * np.ones(n) / n + 0.5 * spatial / spatial.sum()
    base = base64.float()
    B = B64.float()
    V = V64.float()
    origin = origin64.float()
    Hmean = hmean.float()
    HV = hv.float()
    np.savez_compressed(out / 'prepared.npz', q=q, coefficients=coefficients, weights=weights, split=split, phase=phase, step=step, L_base=base.cpu().numpy(), L_V=V.cpu().numpy(), B=B.cpu().numpy(), origin=origin.cpu().numpy(), H_mean=Hmean.cpu().numpy(), H_V=HV.cpu().numpy(), H_contexts=H.cpu().numpy(), B_float64=B64.cpu().numpy(), L_V_float64=V64.cpu().numpy(), L_base_float64=base64.cpu().numpy())
    prepare = dict(checkpoint=checkpoint, n=n, rank_L=normal_rank, rank_H=4, representation='Normal complement of new B inside orthonormal span of old [B,L_V]; coefficient/base labels recomputed from native L', basis='Prior global L PCA span is transductive; no new PCA target or quality fitting', q_formula='(native L - native final L) @ new unit orthonormal B / sqrt(41984)', q_archive_max_error=float(actual_q_error.max()), B_orthonormal_error=float((B64.T @ B64 - torch.eye(2, device='cuda')).abs().max()), V_orthonormal_error=float((V64.T @ V64 - torch.eye(normal_rank, device='cuda')).abs().max()), B_V_cross_error=float((B64.T @ V64).abs().max()), L_base_projection_error=float(((base64 - origin64) @ B64).abs().max() / SQ), weights='0.5 uniform states + 0.5 uniform occupied new-coordinate 256x256 bins', split='Existing modulo19 step split: train0, selection validation1(step7), independent regression test2(step13); native boundary duplicates in training', view_sha256=sha(view_path), old_prepared_sha256=sha(old_path), probe_sha256=sha(probe_path), n_exact_unique_q=len(np.unique(q, axis=0)), seconds=time.monotonic() - start)
    (out / 'prepare_audit.json').write_text(json.dumps(prepare, indent=2) + '\n')
    print('PREPARED', json.dumps(prepare), flush=True)

    def lift(query, prediction):
        qt = torch.tensor(query, device='cuda', dtype=torch.float32)
        ct = torch.tensor(prediction, device='cuda', dtype=torch.float32)
        zl = base + SQ * (qt @ B.T + ct[:, :normal_rank] @ V.T)
        zh = Hmean + SQ * (ct[:, normal_rank:] @ HV.T)
        return (zl, zh)
    native_probe = (L.reshape(-1, 82, 512).float().mean(1).double() @ weight + bias).cpu().numpy()

    def evaluate(idx, prediction):
        arrays = {k: [] for k in ['L_RMS', 'H_RMS', 'L_bf16_RMS', 'H_bf16_RMS', 'probe_error', 'probe', 'probe_before_bf16', 'projection_error', 'bf16_projection_error']}
        for begin in range(0, len(idx), args.batch):
            jj = idx[begin:begin + args.batch]
            zl, zh = lift(q[jj], prediction[begin:begin + len(jj)])
            nl = L[jj]
            nh = H[torch.tensor(phase[jj] - 1, device='cuda')]
            zlb = zl.to(torch.bfloat16).float()
            zhb = zh.to(torch.bfloat16).float()
            score = zlb.reshape(-1, 82, 512).mean(1).double() @ weight + bias
            before = zl.reshape(-1, 82, 512).mean(1).double() @ weight + bias
            values = dict(L_RMS=(zl - nl).square().mean(1).sqrt(), H_RMS=(zh - nh).square().mean(1).sqrt(), L_bf16_RMS=(zlb - nl).square().mean(1).sqrt(), H_bf16_RMS=(zhb - nh).square().mean(1).sqrt(), probe=score, probe_before_bf16=before, probe_error=score - torch.tensor(native_probe[jj], device='cuda'), projection_error=((zl - origin).double() @ B64 / SQ - torch.tensor(q[jj], device='cuda')).norm(dim=1), bf16_projection_error=((zlb - origin).double() @ B64 / SQ - torch.tensor(q[jj], device='cuda')).norm(dim=1))
            for k, value in values.items():
                arrays[k].append(value.cpu().numpy())
        arrays = {k: np.concatenate(v) for k, v in arrays.items()}

        def summarize(mask):
            r = {k: summary(v[mask]) for k, v in arrays.items() if k not in ['probe', 'probe_before_bf16', 'probe_error']}
            error = arrays['probe_error'][mask]
            r['probe'] = dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.mean(error ** 2))), bias=float(error.mean()), p95_absolute=float(np.quantile(np.abs(error), 0.95)), max_absolute=float(np.abs(error).max()))
            return r
        report = dict(n=len(idx), overall=summarize(np.ones(len(idx), dtype=bool)), per_phase={f'H{h}': summarize(phase[idx] == h) for h in range(1, 6)}, by_regime={name: summarize(mask) for name, mask in [('first128', step[idx] < 128), ('middle', (step[idx] >= 128) & (step[idx] < 3073)), ('tail1024', step[idx] >= 3073)]})
        return (arrays, report)
    oracle_arrays, oracle_report = evaluate(np.arange(n), coefficients)
    (out / 'compression_oracle.json').write_text(json.dumps(oracle_report, indent=2) + '\n')
    tr, va, te = [np.flatnonzero(split == i) for i in [0, 1, 2]]
    vm, fit_report = fit_global(q, coefficients, tr, va, weights=weights, method='kernel', device='cuda')
    vm.save(out / 'validation_kernel_model.pt')
    val_arrays, val_report = evaluate(va, vm.predict(q[va]))
    heldout_arrays, heldout_report = evaluate(te, vm.predict(q[te]))
    del vm
    print('HELDOUT', json.dumps(heldout_report['overall']), flush=True)
    fm, display_report = fit_global(q, coefficients, np.arange(n), [], weights=weights, method='kernel', device='cuda', selected=fit_report)
    fm.save(out / 'kernel_model.pt')
    prediction = fm.predict(q)
    display_arrays, display_evaluation = evaluate(np.arange(n), prediction)
    trajectory = dict(q=q, phase=phase, step=step, trajectory_q=q, trajectory_phase=phase, trajectory_step=step, native_probe=native_probe, fitted_probe=display_arrays['probe'], fitted_probe_before_bf16=display_arrays['probe_before_bf16'], oracle_probe=oracle_arrays['probe'], heldout_test_idx=te, heldout_native_probe=native_probe[te], heldout_fitted_probe=heldout_arrays['probe'])
    for h in range(1, 6):
        mask = phase == h
        trajectory[f'H{h}_global_xy'] = q[mask]
        trajectory[f'H{h}_xy'] = q[mask]
        trajectory[f'H{h}_probe_true'] = native_probe[mask]
        trajectory[f'H{h}_probe_fit'] = display_arrays['probe'][mask]
    for label, arrays in [('display', display_arrays), ('heldout_test', heldout_arrays), ('validation_selection', val_arrays)]:
        for k, v in arrays.items():
            trajectory[f'{label}_{k}'] = v
    np.savez_compressed(out / 'trajectory_arrays.npz', **trajectory)
    evaluation = dict(checkpoint_sha256=checkpoint, probe_sha256=sha(probe_path), kernel_model_sha256=sha(out / 'kernel_model.pt'), prepared_sha256=sha(out / 'prepared.npz'), script_sha256=sha(__file__), fitting_module_sha256=sha(ROOT / 'global_fit.py'), no_phase_or_branch_inputs=True, probe_refitted=False, objective='Only weighted latent coefficient squared errors; no probe or other score loss', compression_oracle=oracle_report, selection_validation=val_report, heldout_test=heldout_report, display_all_states=display_evaluation, fit=fit_report, display_fit=display_report, scalar='Frozen matching-checkpoint linear_L probe; predicted excess probability mass, not true model loss or expected wrong cells', probe_formula='Current reconstructed L -> bf16 -> float32 mean of all82 tokens incl CLS -> float64 affine readout', same_coordinate_H_boundaries='All four native L/H-switch boundaries share q with different H labels; one smooth global f compromises at these locations', seconds=time.monotonic() - start)
    (out / 'fit_audit.json').write_text(json.dumps(evaluation, indent=2) + '\n')
    print('DISPLAY', json.dumps(display_evaluation['overall']), flush=True)
    regions = {}
    for view_name in ['global', 'H1', 'H2', 'H3', 'H4']:
        if view_name == 'global':
            lower = q.min(0) - 0.05
            upper = q.max(0) + 0.05
            shape = (451, 501)
            tail_idx = np.arange(n)
            rule = 'All trajectory min/max with 0.05 RMS margin per side'
        else:
            h = int(view_name[1:])
            tail_idx = np.flatnonzero((phase == h) & (step >= 3073))
            tail = q[tail_idx]
            lo, hi = (tail.min(0), tail.max(0))
            pad = np.maximum(0.4 * (hi - lo), 0.002)
            lower, upper = (lo - pad, hi + pad)
            shape = (301, 301)
            rule = 'Per-axis last1024 native-coordinate bounds plus max(0.4*span,0.002 RMS)'
        xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], shape[1]), np.linspace(lower[1], upper[1], shape[0]))
        query = np.column_stack([xx.ravel(), yy.ravel()])
        scores = []
        before_scores = []
        for begin in range(0, len(query), args.batch):
            points = query[begin:begin + args.batch]
            zl, _ = lift(points, fm.predict(points))
            raw = zl.reshape(-1, 82, 512)
            before = raw.mean(1).double() @ weight + bias
            value = raw.to(torch.bfloat16).float().mean(1).double() @ weight + bias
            scores.append(value.cpu().numpy())
            before_scores.append(before.cpu().numpy())
        value = np.concatenate(scores).reshape(shape)
        before = np.concatenate(before_scores).reshape(shape)
        assert np.isfinite(value).all() and np.isfinite(before).all()
        path = out / f'existing_L_probe_{view_name}.npz'
        np.savez_compressed(path, x=xx, y=yy, probe=value, probe_before_bf16=before, tail_q=q[tail_idx], tail_probe=display_arrays['probe'][tail_idx], tail_probe_native=native_probe[tail_idx])
        regions[view_name] = dict(shape=list(shape), bounds=[lower.tolist(), upper.tolist()], bounds_rule=rule, min=float(value.min()), max=float(value.max()), mean=float(value.mean()), negative_count=int((value < 0).sum()), bf16_absolute_change_mean=float(np.abs(value - before).mean()), bf16_absolute_change_max=float(np.abs(value - before).max()), sha256=sha(path))
        print('GRID', view_name, json.dumps(regions[view_name]), flush=True)
        grid_audit = dict(checkpoint_sha256=checkpoint, kernel_model_sha256=evaluation['kernel_model_sha256'], probe_sha256=sha(probe_path), probe_refitted=False, formula=evaluation['probe_formula'], processing='None: no scalar interpolation, clipping, smoothing, walls or normalization', regions=regions, seconds=time.monotonic() - start)
        (out / 'grid_audit.json').write_text(json.dumps(grid_audit, indent=2) + '\n')
    print('COMPLETE', time.monotonic() - start, flush=True)
if __name__ == '__main__':
    main()
