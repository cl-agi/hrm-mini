"""At matched L6 states, compare the next L with and without one H update."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import os
from hrm.paths import repository_root
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .train_secondary_hl_readout import model_hash

BRANCHES = ('before_L6', 'start', 'L_only', 'H_only', 'H_then_L', 'L_then_H', 'L_then_L')


def load_records(path):
    plan = json.loads((path / 'run_config.json').read_text())
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('status') != 'complete' or any(manifest.get(k) != v for k, v in plan.items()):
        raise ValueError('Incomplete or inconsistent readout records')
    for name in ('secondary_heads.pt', 'heldout_trajectories.npz'):
        if sha(path / name) != manifest['artifact_sha256'][name]:
            raise ValueError(f'Readout artifact changed: {name}')
    return manifest

def cos(a, b):
    a = a.flatten(1).double()
    b = b.flatten(1).double()
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-20)).clamp(-1, 1).cpu().numpy()

@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    records = Path(os.environ.get('HRM_ANALYSIS_RECORDS', repository_root() / 'exp/dynamics/records/answer_direction'))
    p.add_argument('--records', type=Path, default=records / 'readout')
    p.add_argument('--output-dir', type=Path, default=records / 'intervention')
    p.add_argument('--checkpoint', type=Path, help='Relocated copy of the identical readout checkpoint')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(argv)
    P, O = a.records.resolve(), a.output_dir.resolve()
    meta = load_records(P)
    horizon, batch_size = meta['horizon'], meta['batch_size']
    if horizon < 3:
        raise ValueError('At least 3 inference cycles required for eligible cycle 2')
    cp = a.checkpoint or Path(meta['checkpoint'])
    if sha(cp) != meta['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from readout checkpoint')
    if O.exists() and any(O.iterdir()):
        raise FileExistsError(O)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    with np.load(P / 'heldout_trajectories.npz', allow_pickle=False) as f:
        old = {k: f[k] for k in f.files}
    heads = torch.load(P / 'secondary_heads.pt', map_location='cpu', weights_only=True)
    if heads['selected'] != meta['selected']:
        raise ValueError('Selected readout differs from manifest')
    if len(np.unique(old['ids'])) != len(old['ids']) or old['ids'].tolist() != meta['splits']['test_rows']:
        raise ValueError('Heldout IDs differ from frozen split')
    if old['native_pred'].shape != (len(old['ids']), horizon * 7 + 1, 81):
        raise ValueError('Invalid native trajectory dimensions')
    m = HRM(analysis_architecture(meta['config'])).to(a.device).eval().requires_grad_(False)
    m.load_state_dict(unwrap_state_dict(torch.load(cp, map_location='cpu', weights_only=True)), strict=True)
    initial_hash = model_hash(m)
    if initial_hash != meta['unchanged_backbone_sha256']:
        raise ValueError('Backbone differs from readout backbone')
    O.mkdir(parents=True, exist_ok=True)
    with (O / 'started.json').open('x') as f:
        json.dump(dict(readout=str(P), manifest_sha256=sha(P / 'manifest.json'), device=a.device), f)
    ws = {n: h['weight'].to(a.device) for n, h in heads['heads'].items() if n.startswith('sum')}
    w = ws[heads['selected']]

    def logits(h, l, weight=w):
        return F.linear(h[:, 1:].float() + l[:, 1:].float(), weight)

    def project(d, blank, weight=w):
        v = F.linear(d[:, 1:].float(), weight)[:, :, 1:]
        v = v - v.mean(-1, keepdim=True)
        return v * blank[:, :, None]
    rows = []
    packs = []
    parity = 0
    for start in range(0, len(old['ids']), batch_size):
        ids = old['ids'][start:start + batch_size]
        x = torch.tensor(old['input'][start:start + batch_size], device=a.device)
        y = torch.tensor(old['target'][start:start + batch_size, 1:], device=x.device)
        blank = x[:, 1:] == 0
        e = m.embed(x)
        h = m.zH_init.expand_as(e).clone()
        l = m.zL_init.expand_as(e).clone()
        for c in range(1, horizon + 1):
            for k in range(6):
                prev = l
                l = m.L_level(l + h + e)
            new_h = m.H_level(h + l)
            native = m.lm_head(new_h)[:, 1:].argmax(-1).cpu().numpy()
            if not np.array_equal(native, old['native_pred'][start:start + len(ids), c * 7]):
                raise RuntimeError(f'Native trajectory mismatch at inference cycle {c}')
            parity += len(ids)
            eligible = (old['native_pred'][start:start + len(ids), (c - 1) * 7] != old['target'][start:start + len(ids), 1:]).any(1)
            if c >= 2 and c < horizon and eligible.any():
                l_without = m.L_level(l + h + e)
                l_after_h = m.L_level(l + new_h + e)
                reversed_h = m.H_level(h + l_without)
                l_twice = m.L_level(l_without + h + e)
                dprev = l.float() - prev.float()
                dno = l_without.float() - l.float()
                dyes = l_after_h.float() - l.float()
                metrics = dict(latent_cos_continue=cos(dprev, dno), latent_cos_after_H=cos(dprev, dyes), latent_cos_between_branches=cos(dno, dyes))
                for name, weight in ws.items():
                    vp = project(dprev, blank, weight)
                    vn = project(dno, blank, weight)
                    vh = project(dyes, blank, weight)
                    metrics[name + '_cos_continue'] = cos(vp, vn)
                    metrics[name + '_cos_after_H'] = cos(vp, vh)
                    metrics[name + '_cos_between_branches'] = cos(vn, vh)
                lg = {'before_L6': logits(h, prev), 'start': logits(h, l), 'L_only': logits(h, l_without), 'H_only': logits(new_h, l), 'H_then_L': logits(new_h, l_after_h), 'L_then_H': logits(reversed_h, l_without), 'L_then_L': logits(h, l_twice)}
                for key, step in [('before_L6', c * 7 - 2), ('start', c * 7 - 1), ('H_only', c * 7), ('H_then_L', c * 7 + 1)]:
                    expected = old[heads['selected'] + '_pred'][start:start + len(ids), step]
                    if not np.array_equal(lg[key].argmax(-1).cpu().numpy(), expected):
                        raise RuntimeError(f'Selected readout trajectory mismatch: {key}, cycle {c}')
                for key, v in lg.items():
                    metrics[key + '_errors'] = v.argmax(-1).ne(y).sum(1).cpu().numpy()
                    loss = F.cross_entropy(v.flatten(0, 1), y.flatten(), reduction='none').reshape_as(y)
                    metrics[key + '_blank_ce'] = ((loss * blank).sum(1) / blank.sum(1).clamp_min(1)).cpu().numpy()
                metrics['H_L_order_board_difference'] = lg['H_then_L'].argmax(-1).ne(lg['L_then_H'].argmax(-1)).sum(1).cpu().numpy()
                metrics['next_L_delta_rms_no_H'] = dno.square().mean((1, 2)).sqrt().cpu().numpy()
                metrics['next_L_delta_rms_after_H'] = dyes.square().mean((1, 2)).sqrt().cpu().numpy()
                for i, rid in enumerate(ids):
                    if not eligible[i]:
                        continue
                    rows.append(dict(case=int(rid), cycle=c, **{k: float(v[i]) for k, v in metrics.items()}))
                    packs.append({k: v[i].cpu().numpy().astype(np.float32) for k, v in lg.items()})
            h = new_h
        print('intervention batch', start + len(ids), flush=True)
    columns = ['case', 'cycle', 'latent_cos_continue', 'latent_cos_after_H', 'latent_cos_between_branches']
    columns += [name + suffix for name in ws for suffix in ('_cos_continue', '_cos_after_H', '_cos_between_branches')]
    columns += [key + suffix for key in BRANCHES for suffix in ('_errors', '_blank_ce')]
    columns += ['H_L_order_board_difference', 'next_L_delta_rms_no_H', 'next_L_delta_rms_after_H']
    pd.DataFrame(rows, columns=columns).to_csv(O / 'direction_interventions.csv', index=False)
    np.savez_compressed(O / 'branch_logits.npz',
        **{k: np.stack([pack[k] for pack in packs]) if packs else np.empty((0, 81, 10), dtype=np.float32) for k in BRANCHES},
        ids=np.array([r['case'] for r in rows], dtype=np.int64),
        cycles=np.array([r['cycle'] for r in rows], dtype=np.int64))
    if model_hash(m) != initial_hash or any(p.grad is not None or p.requires_grad for p in m.parameters()):
        raise RuntimeError('Backbone changed')
    report = dict(status='complete', readout=str(P), readout_manifest_sha256=sha(P / 'manifest.json'),
        checkpoint_sha256=sha(cp), selected=heads['selected'], seed=meta['seed'], device=a.device,
        unchanged_model_state_hash=initial_hash, native_boards_exactly_replayed=parity,
        states=len(rows), puzzles=len(set(r['case'] for r in rows)), heldout_puzzles=len(old['ids']),
        horizon=horizon, code_sha256=sha(Path(__file__)),
        protocol=f'Every unsolved-at-entry inference cycle 2..{horizon-1}. At L6, compare L versus H->L; compare H->L versus L->H with two calls each. Branches never feed back.',
        vector='All latent tokens for latent cosine; blank-cell digit 1..9 logits centered per cell for projected cosine.',
        artifact_sha256={name:sha(O/name) for name in ('direction_interventions.csv','branch_logits.npz')})
    (O / 'manifest.json').write_text(json.dumps(report, indent=2))
    print('complete', len(rows), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
