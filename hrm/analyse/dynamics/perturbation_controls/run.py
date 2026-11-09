from hrm.paths import work_path
from hrm.paths import repo_path
import json, time, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(str(work_path('hrm/analyse/sudoku/core')))
pass
from hrm.analyse.dynamics.perturbation_controls.verify_test_time_scaling import load_metadata
from argparse import Namespace

def decode(x):
    return (np.asarray(x).astype(np.uint32) << 16).view(np.float32)

@torch.inference_mode()
def main():
    cfg = json.loads(Path('config.json').read_text())
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    src = ROOT / 'experiment/L4096_case_search_20260928/shortlist'
    summary = json.loads((src / 'summary.json').read_text())
    ci = next((i for i, x in enumerate(summary['selected']) if x['case'] == 6566))
    model, identity = load_metadata(Namespace(checkpoint=summary['model']['checkpoint'], config=summary['model']['config_source'], architecture='hrm', device='cuda'))
    assert identity['checkpoint_sha256'] == summary['model']['checkpoint_sha256']
    model.eval().requires_grad_(False)
    emb = model.embed(torch.tensor(np.load(src / 'initial.npz')['inputs'][ci:ci + 1], device='cuda'))
    D = 82 * 512
    rows = []
    saved = {}
    started = time.time()
    for h in range(1, 6):
        L = torch.tensor(decode(np.load(src / f'H{h:03d}_L_bits.npy', mmap_mode='r')[:, ci]).reshape(4097, D), device='cuda')
        H = torch.tensor(decode(np.load(src / f'H{h:03d}_context.npz')['H_before_bits'][ci]), device='cuda', dtype=torch.bfloat16)[None]
        ref = L[-1024:]
        mean = ref.mean(0)
        rc = ref - mean
        _, ev = torch.linalg.eigh(rc @ rc.T)
        basis = torch.linalg.qr(rc.T @ ev[:, -2:].flip(1)).Q
        starts = []
        meta = []
        for a in cfg['anchors']:
            tangent = L[a + 2] - L[a - 2]
            tangent /= tangent.norm()
            tp = tangent @ basis
            normal = basis @ torch.stack([-tp[1], tp[0]])
            normal -= normal.dot(tangent) * tangent
            normal /= normal.norm()
            dirs = [('tangent', tangent), ('tangent', -tangent), ('plane_normal', normal), ('plane_normal', -normal)]
            for j in range(6):
                v = torch.randn(D, device='cuda')
                v -= v.dot(tangent) * tangent
                v /= v.norm()
                dirs.append(('random_normal', v))
            starts.append(L[a])
            meta.append((a, 0.0, 'control', 0))
            for r in cfg['radii']:
                for j, (kind, v) in enumerate(dirs):
                    starts.append(L[a] + r * D ** 0.5 * v)
                    meta.append((a, r, kind, j))
        starts = torch.stack(starts)
        distances = []
        syncs = []
        coords = []
        actual = []
        control_error = 0.0
        refnorm = rc.square().sum(1)
        for lo in range(0, len(starts), 32):
            m = meta[lo:lo + 32]
            anchors = torch.tensor([x[0] for x in m], device='cuda')
            z = starts[lo:lo + 32].reshape(-1, 82, 512).to(torch.bfloat16)
            actual.extend((z.float().flatten(1) - L[anchors]).square().mean(1).sqrt().cpu().tolist())
            bd = []
            bs = []
            bq = []
            for t in range(cfg['steps'] + 1):
                flat = z.float().flatten(1)
                center = flat - mean
                bq.append((center @ basis / D ** 0.5).cpu().numpy())
                if t in cfg['measure_steps']:
                    ix = (center.square().sum(1)[:, None] + refnorm[None] - 2 * center @ rc.T).argmin(1)
                    bd.append((center - rc[ix]).square().mean(1).sqrt().cpu().numpy())
                    bs.append((flat - L[anchors + t]).square().mean(1).sqrt().cpu().numpy())
                    for k, item in enumerate(m):
                        if item[2] == 'control':
                            control_error = max(control_error, (flat[k] - L[item[0] + t]).abs().max().item())
                if t < cfg['steps']:
                    z = model.L_level(z + H + emb)
            distances.append(np.stack(bd, 1))
            syncs.append(np.stack(bs, 1))
            coords.append(np.stack(bq, 1))
        assert control_error == 0, control_error
        dist = np.concatenate(distances)
        sync = np.concatenate(syncs)
        q = np.concatenate(coords)
        for r in cfg['radii']:
            for kind in ['tangent', 'plane_normal', 'random_normal']:
                sel = np.array([x[1] == r and x[2] == kind for x in meta])
                ratio = dist[sel] / dist[sel, 0, None]
                rows.append(dict(H=h, radius=r, kind=kind, n=int(sel.sum()), set_ratio_median=np.median(ratio, 0).tolist(), fraction_closer=(ratio[:, -1] < 1).mean().item(), sync_ratio_median=np.median(sync[sel] / sync[sel, 0, None], 0).tolist()))
        for key, val in dict(distance=dist, sync=sync, q=q, meta=np.array(meta, dtype=str), actual_rms=actual, terminal_xy=(rc @ basis / D ** 0.5).cpu().numpy()).items():
            saved[f'H{h}_{key}'] = val
        np.savez_compressed('results.npz', **saved)
        Path('summary.json').write_text(json.dumps(dict(config=cfg, identity=identity, control_max_abs_error=control_error, rows=rows, elapsed_seconds=time.time() - started), indent=2))
        print(f'H{h} complete; {time.time() - started:.1f}s', flush=True)
if __name__ == '__main__':
    main()
