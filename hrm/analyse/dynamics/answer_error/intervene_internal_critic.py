from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
"""One-shot signed, norm-preserving critic intervention; fixed CPU protocol."""
from pathlib import Path
import argparse
import csv
import json
import sys
import time
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parent
pass
from hrm.analyse.dynamics.answer_error.reproduce_internal_critic import common, excess_error
SOURCE = work_path('internal_critic')
EVENTS = ['L'] * 6 + ['H'] + ['L'] * 6 + ['H']
DOSES = [0.002, 0.01, 0.03]
SEED = 20261006

def arms():
    result = [dict(module='native', direction='native', alpha=0.0, sign=0)]
    for module in ['L', 'H']:
        for alpha in DOSES:
            for direction in ['critic'] + [f'random{k}' for k in range(4)]:
                for sign in [1, -1]:
                    result.append(dict(module=module, direction=direction, alpha=alpha, sign=sign))
    return result

def tangent(s, v):
    return v - s * (s * v).sum(-1, keepdim=True) / s.square().sum(-1, keepdim=True).clamp_min(1e-30)

def directions(s, w, covariance):
    """PSD-preconditioned gradient and four per-token critic-null controls."""
    s = s.double()
    pg = tangent(s, w.double() / s.shape[1])
    v = tangent(s, pg @ covariance.double())
    result = {'critic': v}
    for k in range(4):
        gen = torch.Generator().manual_seed(SEED + 100 + k)
        r = tangent(s, torch.randn(s.shape[-1], generator=gen, dtype=torch.double).to(s.device))
        r = r - pg * (r * pg).sum(-1, keepdim=True) / pg.square().sum(-1, keepdim=True).clamp_min(1e-30)
        result[f'random{k}'] = r
    if s.norm(dim=-1).min() <= 1e-10 or pg.norm(dim=-1).min() <= 1e-10:
        raise ValueError('Degenerate state/critic tangent: retain cohort and diagnose.')
    for key, value in result.items():
        if value.norm(dim=-1).min() <= 1e-10:
            raise ValueError(f'Degenerate intervention direction {key}')
        result[key] = value / value.norm(dim=-1, keepdim=True)
    return result

def rotate(s, v, alpha, sign):
    theta = 2 * np.arcsin(alpha / 2)
    sd = s.double()
    return (sd * np.cos(theta) + sign * sd.norm(dim=-1, keepdim=True) * v * np.sin(theta)).to(s.dtype)

def cosine(a, b):
    a, b = (a.flatten(1), b.flatten(1))
    norms = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(norms > 1e-12, (a * b).sum(-1) / norms.clamp_min(1e-30), torch.nan)

class Probe:

    def __init__(self):
        self.fits = torch.load(SOURCE / 'probes.pt', weights_only=True)
        self.w = self.fits['linear_HL']['weight']
        self.b = self.fits['linear_HL']['bias']
        self.mlps = []
        for k in [1, 2, 3]:
            net = nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 1))
            fit = self.fits[f'mlp_seed{k}']
            net.load_state_dict(fit['state_dict'])
            net.eval().requires_grad_(False)
            self.mlps.append((net, fit))

    def scores(self, feature):
        mlp = torch.stack([net((feature - fit['mean']) / fit['scale']).squeeze(-1) * fit['target_scale'] + fit['target_mean'] for net, fit in self.mlps], -1)
        return (feature.double() @ self.w + self.b, mlp)

@torch.inference_mode()
def prepare(out):
    if (out / 'setup.pt').exists():
        return
    source = torch.load(SOURCE / 'trajectory.pt', mmap=True, weights_only=True)
    excluded = set(source['ids'].tolist())
    raw = list(csv.DictReader((repo_path('data/sudoku/sudoku-extreme-1k/test_hard.csv')).open()))
    original_train = list(csv.DictReader((repo_path('data/sudoku/sudoku-extreme-1k/train.csv')).open()))
    seen = {r['question'] for r in original_train} | {raw[i]['question'] for i in excluded}
    ids = []
    for i in np.random.default_rng(SEED).permutation(len(raw)):
        if int(i) not in excluded and raw[i]['question'] not in seen:
            ids.append(int(i))
            seen.add(raw[i]['question'])
        if len(ids) == 160:
            break
    assert len(ids) == 160
    x = torch.tensor([[0] + [int(c) for c in raw[i]['question'].replace('.', '0')] for i in ids])
    y = torch.tensor([[0] + [int(c) for c in raw[i]['answer']] for i in ids])
    covariance, mean, pc = ({}, {}, {})
    for module, sl in [('L', slice(0, 512)), ('H', slice(512, 1024))]:
        a = source['features'][:256, :, sl].flatten(0, 1).double()
        mean[module] = a.mean(0)
        a = a - mean[module]
        covariance[module] = a.T @ a / len(a)
        _, eig = torch.linalg.eigh(covariance[module])
        pc[module] = eig[:, -128:]
    setup = dict(ids=torch.tensor(ids), x=x, y=y, covariance=covariance, mean=mean, pc=pc)
    torch.save(setup, out / 'setup.pt')
    common.csvout(out / 'split.csv', [dict(ordinal=j, row_id=i, split='pilot' if j < 32 else 'test') for j, i in enumerate(ids)])
    common.js(out / 'execution_freeze.json', dict(protocol_sha256=common.sha(SOURCE / 'INTERVENTION_DESIGN.md'), script_sha256=common.sha(__file__), probes_sha256=common.sha(SOURCE / 'probes.pt'), source_manifest_sha256=common.sha(SOURCE / 'manifest.json'), setup_sha256=common.sha(out / 'setup.pt'), arms=arms(), events=EVENTS, split=[32, 128], device='cpu', dtype='float32', pc_rank=128, tolerances=dict(relative_dose_abs_error=2e-07, token_norm_relative_error=2e-06, random_signed_score_abs_error=0.001, native_forward_max_abs_error=1e-05), inference='four H cycles to fork; then one pulse and two H cycles of continuation', storage='all 15 states: pooled features, native/diagnostic logits, scores, error, quality; all 14 update norms/cosines. Full arm hidden tensors are not retained.', statistical_note='Paired sign-flip p-values assume symmetry of puzzle-level slopes under the null; interventions themselves are deterministic, not randomized assignment.', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))

@torch.inference_mode()
def observe(model, probe, h, l, x, y):
    feature = torch.cat([l.mean(1), h.mean(1)], -1)
    score, mlp = probe.scores(feature)
    logits = model.lm_head(h).float()
    decoded = model.lm_head(model.H_level(h + l)).float()
    probs = decoded[:, 1:, 1:].softmax(-1)
    pred = logits.argmax(-1)
    clues = x[:, 1:].ne(0)
    return dict(feature=feature, critic=score, mlp=mlp, native_logits=logits, decoded_logits=decoded, error=excess_error(probs), entropy=-(probs * probs.clamp_min(1e-30).log()).sum(-1).mean(-1), exact=pred[:, 1:].eq(y[:, 1:]).all(-1), accuracy=pred[:, 1:].eq(y[:, 1:]).float().mean(-1), clue_rate=((pred[:, 1:] == x[:, 1:]) & clues).sum(-1) / clues.sum(-1).clamp_min(1))

@torch.inference_mode()
def rollout(model, probe, h, l, x, y, reference=None):
    emb = model.embed(x)
    observations = [observe(model, probe, h, l, x, y)]
    increments, norms, alignment, state_dist = ([], [], [], [])
    states = [(h.clone(), l.clone())] if reference is None else None
    if reference is not None:
        state_dist.append(torch.stack([(h - reference['states'][0][0]).flatten(1).norm(dim=-1), (l - reference['states'][0][1]).flatten(1).norm(dim=-1)], -1))
    for j, event in enumerate(EVENTS):
        if event == 'L':
            nxt = model.L_level(l + h + emb)
            change = nxt - l
            l = nxt
        else:
            nxt = model.H_level(h + l)
            change = nxt - h
            h = nxt
        norms.append(change.flatten(1).norm(dim=-1))
        observations.append(observe(model, probe, h, l, x, y))
        if reference is None:
            increments.append(change.clone())
            states.append((h.clone(), l.clone()))
        else:
            alignment.append(cosine(change, reference['increments'][j]))
            state_dist.append(torch.stack([(h - reference['states'][j + 1][0]).flatten(1).norm(dim=-1), (l - reference['states'][j + 1][1]).flatten(1).norm(dim=-1)], -1))
    result = {k: torch.stack([o[k] for o in observations], 1) for k in observations[0]}
    result['step_norm'] = torch.stack(norms, 1)
    result['step_cosine'] = torch.ones_like(result['step_norm']) if reference is None else torch.stack(alignment, 1)
    if reference is None:
        result['step_cosine'][result['step_norm'] <= 1e-06] = torch.nan
    result['state_distance_HL'] = torch.zeros(len(x), 15, 2) if reference is None else torch.stack(state_dist, 1)
    if reference is None:
        return (result, dict(increments=increments, states=states))
    return result

@torch.inference_mode()
def execute(args, out):
    setup = torch.load(out / 'setup.pt', weights_only=True)
    frozen = json.loads((out / 'execution_freeze.json').read_text())
    assert common.sha(SOURCE / 'probes.pt') == frozen['probes_sha256']
    assert common.sha(out / 'setup.pt') == frozen['setup_sha256']
    model, identity = common.model_load('hrm', 1, device='cpu')
    assert model.H_cycles == 2 and model.L_cycles == 6
    probe = Probe()
    start, end = (0, 32) if args.split == 'pilot' else (32, 160)
    if args.split == 'test':
        assert json.loads((out / 'pilot_check.json').read_text())['passed'], 'Pilot must pass first.'
    dest = out / args.split
    dest.mkdir(exist_ok=True)
    if list(dest.glob('batch_*.pt')):
        raise FileExistsError('Preserve existing batches: use a new output directory.')
    began = time.monotonic()
    for first in range(start, end, args.batch_size):
        stop = min(first + args.batch_size, end)
        x, y = (setup['x'][first:stop], setup['y'][first:stop])
        carry = common.initial(model, len(x))
        for _ in range(2):
            carry = common.step(model, carry, x, 1)
        h, l = (carry['z_H'], carry['z_L'])
        native, ref = rollout(model, probe, h, l, x, y)
        native_end, _ = model(carry, x)
        parity = max(float((native_end['z_H'] - ref['states'][-1][0]).abs().max()), float((native_end['z_L'] - ref['states'][-1][1]).abs().max()))
        assert parity < 1e-05, parity
        vectors = {m: directions(z, probe.w[sl], setup['covariance'][m]) for m, z, sl in [('L', l, slice(0, 512)), ('H', h, slice(512, 1024))]}
        all_results = [native]
        audits = [dict(alpha=torch.zeros(len(x)), norm_error=torch.zeros(len(x)), displacement=torch.zeros(len(x)), distance_to_train_mean=torch.zeros(len(x)), pooled_tail_displacement=torch.zeros(len(x)))]
        for ai, arm in enumerate(arms()[1:], 1):
            module, alpha, sign = (arm['module'], arm['alpha'], arm['sign'])
            z = l if module == 'L' else h
            altered = rotate(z, vectors[module][arm['direction']], alpha, sign)
            dose = (altered - z).double().flatten(1).norm(dim=-1) / z.double().flatten(1).norm(dim=-1)
            norm_error = ((altered.double().norm(dim=-1) - z.double().norm(dim=-1)).abs() / z.double().norm(dim=-1)).max(-1).values
            assert float((dose - alpha).abs().max()) < 2e-07
            assert float(norm_error.max()) < 2e-06
            pool = altered.double().mean(1)
            pool_delta = pool - z.double().mean(1)
            pc = setup['pc'][module]
            audits.append(dict(alpha=dose, norm_error=norm_error, displacement=(altered - z).flatten(1).norm(dim=-1), distance_to_train_mean=(pool - setup['mean'][module]).norm(dim=-1), pooled_tail_displacement=(pool_delta - pool_delta @ pc @ pc.T).norm(dim=-1)))
            result = rollout(model, probe, h if module == 'L' else altered, altered if module == 'L' else l, x, y, ref)
            if module == 'L':
                assert torch.equal(result['native_logits'][:, 0], native['native_logits'][:, 0])
            all_results.append(result)
            if ai % 10 == 0:
                print(json.dumps(dict(split=args.split, first=first, arm=ai, seconds=round(time.monotonic() - began, 1))), flush=True)
        batch = {k: torch.stack([r[k] for r in all_results]) for k in native}
        batch['audit'] = {k: torch.stack([r[k] for r in audits]) for k in audits[0]}
        batch.update(ids=setup['ids'][first:stop], x=x, y=y, native_max_abs_error=parity)
        torch.save(batch, dest / f'batch_{first:03d}.pt')
        print(json.dumps(dict(split=args.split, completed=stop - start, total=end - start, seconds=round(time.monotonic() - began, 1))), flush=True)
    common.js(out / f'{args.split}_completion.json', dict(seconds=time.monotonic() - began, checkpoint=identity, script_sha256=common.sha(__file__), batch_size=args.batch_size, threads=args.threads, files={p.name: common.sha(p) for p in sorted(dest.glob('batch_*.pt'))}))

def load_batches(out, split):
    files = sorted((out / split).glob('batch_*.pt'))
    records = [torch.load(p, weights_only=True) for p in files]
    result = {k: torch.cat([r[k] for r in records], 1) for k in records[0] if k not in ['audit', 'ids', 'x', 'y', 'native_max_abs_error']}
    result['audit'] = {k: torch.cat([r['audit'][k] for r in records], 1) for k in records[0]['audit']}
    for k in ['ids', 'x', 'y']:
        result[k] = torch.cat([r[k] for r in records])
    result['native_max_abs_error'] = max((r['native_max_abs_error'] for r in records))
    return result

def check(out, split):
    d = load_batches(out, split)
    expected = 32 if split == 'pilot' else 128
    assert len(d['ids']) == expected and len(d['ids'].unique()) == expected
    setup = torch.load(out / 'setup.pt', weights_only=True)
    assert torch.equal(d['ids'], setup['ids'][:32] if split == 'pilot' else setup['ids'][32:])
    assert torch.isfinite(d['critic']).all() and torch.isfinite(d['error']).all()
    assert torch.isfinite(d['step_norm']).all()
    random_gap, min_critic_gap, max_dose_error = (0.0, float('inf'), 0.0)
    for i, arm in enumerate(arms()):
        max_dose_error = max(max_dose_error, float((d['audit']['alpha'][i] - arm['alpha']).abs().max()))
        if arm['sign'] != 1:
            continue
        gap = d['critic'][i, :, 0] - d['critic'][i + 1, :, 0]
        if arm['direction'] == 'critic':
            min_critic_gap = min(min_critic_gap, float(gap.min()))
        else:
            random_gap = max(random_gap, float(gap.abs().max()))
    assert max_dose_error < 2e-07
    assert float(d['audit']['norm_error'].max()) < 2e-06
    assert random_gap < 0.001, random_gap
    assert min_critic_gap > 0, min_critic_gap
    assert d['native_max_abs_error'] < 1e-05
    report = dict(passed=True, puzzles=expected, arms=61, max_dose_error=max_dose_error, max_token_norm_relative_error=float(d['audit']['norm_error'].max()), max_random_signed_critic_error=random_gap, min_critic_signed_gap=min_critic_gap, native_max_abs_error=d['native_max_abs_error'])
    common.js(out / f'{split}_check.json', report)
    print(json.dumps(report), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'check'])
    parser.add_argument('--output', default='artifacts/critic_intervention_20261004')
    parser.add_argument('--split', choices=['pilot', 'test'], default='pilot')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)
    if args.action == 'prepare':
        prepare(out)
    elif args.action == 'run':
        execute(args, out)
    else:
        check(out, args.split)
if __name__ == '__main__':
    main()
