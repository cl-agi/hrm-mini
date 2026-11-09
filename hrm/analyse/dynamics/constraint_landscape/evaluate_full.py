"""Evaluate the saved global lift across the full plane and newly sampled zooms."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from hrm.analyse.dynamics.constraint_landscape.global_fit import load_global
pass
from hrm.analyse.dynamics.constraint_landscape.evaluate_landscape import ROOT, load_metadata, soft_metrics, sha256, decode
D = 82 * 512
SQ = D ** 0.5

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--method', choices=['kernel', 'fourier'], required=True)
    ap.add_argument('--batch', type=int, default=192)
    args = ap.parse_args()
    out = args.results / 'full_plane'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.monotonic()
    fm = load_global(args.results / f'{args.method}_model.pt', device='cuda')
    d = np.load(args.results / 'prepared.npz')
    t = {key: torch.tensor(d[key], device='cuda', dtype=torch.float32) for key in ['L_base', 'L_V', 'B', 'origin', 'H_mean', 'H_V']}
    q, phase = (d['q'], d['phase'])
    rank = t['L_V'].shape[1]
    src = ROOT / 'experiment/L4096_case_search_20260928/shortlist'
    meta = json.loads((src / 'summary.json').read_text())
    ci = next((i for i, x in enumerate(meta['selected']) if x['case'] == 6566))
    model, identity = load_metadata(argparse.Namespace(checkpoint=meta['model']['checkpoint'], config=meta['model']['config_source'], architecture='hrm', device='cuda'))
    model.eval().requires_grad_(False)
    assert identity['checkpoint_sha256'] == meta['model']['checkpoint_sha256']
    init = np.load(src / 'initial.npz')
    inp = torch.tensor(init['inputs'][ci], device='cuda')
    target = torch.tensor(init['targets'][ci], device='cuda')
    emb = model.embed(inp[None])
    replay = []
    for h in range(1, 6):
        raw = np.load(src / f'H{h:03d}_L_bits.npy', mmap_mode='r')
        ix = np.array([0, 16, 127, 1024, 3073, 4095])
        zl = torch.tensor(decode(raw[ix, ci]), device='cuda').to(torch.bfloat16)
        truth = torch.tensor(decode(raw[ix + 1, ci]), device='cuda')
        zh = torch.tensor(d['H_contexts'][h - 1], device='cuda').reshape(1, 82, 512).to(torch.bfloat16)
        replay.append(float((model.L_level(zl + zh + emb).float() - truth).abs().max()))
    assert max(replay) == 0, replay

    def measure(query):
        c = torch.tensor(fm.predict(query), device='cuda')
        qt = torch.tensor(query, device='cuda', dtype=torch.float32)
        zl = t['L_base'] + SQ * (qt @ t['B'].T + c[:, :rank] @ t['L_V'].T)
        zh = t['H_mean'] + SQ * (c[:, rank:] @ t['H_V'].T)
        projection = torch.linalg.vector_norm((zl - t['origin']) @ t['B'] / SQ - qt, dim=1)
        zl = zl.reshape(-1, 82, 512).to(torch.bfloat16)
        zh = zh.reshape(-1, 82, 512).to(torch.bfloat16)
        logits = model.lm_head(model.H_level(zl + zh))
        p = logits[:, 1:, 1:10].float().softmax(-1)
        score = (1 - p.gather(-1, (target[1:] - 1)[None, :, None].expand(len(p), -1, -1)).squeeze(-1)).sum(-1)
        metrics = soft_metrics(logits, inp, target)
        delta = model.L_level(zl + zh + emb).float().flatten(1) - zl.float().flatten(1)
        flow = delta @ t['B'] / SQ
        return (dict(expected_errors=score.cpu().numpy(), soft_conflict=metrics['soft_conflict'].cpu().numpy(), delta_rms=delta.square().mean(1).sqrt().cpu().numpy(), flow_x=flow[:, 0].cpu().numpy(), flow_y=flow[:, 1].cpu().numpy()), float(projection.max()))
    regions = {'global': (q.min(0) - 0.05, q.max(0) + 0.05, 501, 451)}
    for h in [2, 3, 4]:
        tail = q[phase == h][-1024:]
        lo, hi = (tail.min(0), tail.max(0))
        pad = np.maximum((hi - lo) * 0.4, 0.002)
        regions[f'H{h}'] = (lo - pad, hi + pad, 301, 301)
    audit = dict(method=args.method, checkpoint_sha256=identity['checkpoint_sha256'], model_sha256=sha256(args.results / f'{args.method}_model.pt'), script_sha256=sha256(__file__), model_refitted=False, no_mask=True, native_replay_max_abs_error=max(replay), regions={})
    for name, (lo, hi, nx, ny) in regions.items():
        xs, ys = (np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny))
        xx, yy = np.meshgrid(xs, ys)
        query = np.c_[xx.ravel(), yy.ravel()]
        fields, max_projection = ({}, 0.0)
        region_started = time.monotonic()
        for start in range(0, len(query), args.batch):
            vals, pe = measure(query[start:start + args.batch])
            max_projection = max(max_projection, pe)
            for k, v in vals.items():
                fields.setdefault(k, []).append(v)
        result = dict(x=xx, y=yy, trajectory_q=q, trajectory_phase=phase, trajectory_step=d['step'])
        ranges = {}
        for k, v in fields.items():
            value = np.concatenate(v).reshape(xx.shape)
            assert np.isfinite(value).all(), (args.method, name, k)
            result[k] = value
            ranges[k] = dict(min=float(value.min()), median=float(np.median(value)), max=float(value.max()))
        np.savez_compressed(out / f'{args.method}_{name}.npz', **result)
        audit['regions'][name] = dict(shape=[ny, nx], bounds=[lo.tolist(), hi.tolist()], n_evaluated=len(query), projection_error_max=max_projection, ranges=ranges, seconds=time.monotonic() - region_started)
        (out / f'{args.method}_audit.json').write_text(json.dumps(audit, indent=2))
        print('REGION_COMPLETE', args.method, name, json.dumps(audit['regions'][name]), flush=True)
    audit['seconds'] = time.monotonic() - started
    (out / f'{args.method}_audit.json').write_text(json.dumps(audit, indent=2))
    print('COMPLETE', args.method, audit['seconds'], flush=True)
if __name__ == '__main__':
    main()
