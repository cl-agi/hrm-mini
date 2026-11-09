"""Causal forward interventions for the README no-HH HRM (no flow bias)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from hrm.models.ablations import HRM


def load_model(checkpoint, device='cuda', hh_link=False):
    checkpoint = Path(checkpoint)
    cfg = yaml.safe_load((checkpoint.parent / 'model_config.json').read_text())
    arch = dict(cfg['arch'])
    arch.update(seq_len=82, vocab_size=10, is_causal=False, hh_link=hh_link)
    model = HRM(arch).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    return model, cfg


def initial_state(model, x):
    shape = (*x.shape, model.zH_init.shape[-1])
    return model.zH_init.expand(shape), model.zL_init.expand(shape)


def low_update(model, h, l, x):
    for _ in range(model.L_cycles):
        l = model.L_level(l + h + x)
    return l


def commit(model, h, l):
    return model.H_level(l + h if model.hh_link else l)


def replace_h(model, h, mode):
    if mode == 'reset':
        return model.zH_init.expand_as(h)
    if mode in ('shuffle', 'norm_shuffle'):
        donor = h.roll(1, 0)
        if mode == 'norm_shuffle':
            # Match each token's norm; preserve the donor's direction.
            scale = h.float().norm(dim=-1, keepdim=True) / donor.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
            donor = (donor.float() * scale).to(h.dtype)
        return donor
    raise ValueError(mode)


@torch.no_grad()
def rollout(model, ids, cycles=16, mode='baseline', at=1, sustained=False):
    x = model.embed(ids)
    h, l = initial_state(model, ids)
    if mode == 'recompute':
        if model.hh_link:
            raise ValueError('L-only reduction is only valid without H-H links')
        # Persist only L across commit boundaries. Move each H-module call to
        # the next update's start; reserve the final call for the readout.
        # This uses exactly the same number of H and L calls as the original.
        for step in range(cycles * model.H_cycles):
            h = model.zH_init.expand_as(l) if step == 0 else model.H_level(l)
            l = low_update(model, h, l, x)
            del h
        h = model.H_level(l)
        return h, l, model.lm_head(h)
    for step in range(cycles * model.H_cycles):
        routed_h = h
        if mode not in ('baseline', 'recompute') and (step >= at if sustained else step == at):
            routed_h = replace_h(model, h, mode)
        l = low_update(model, routed_h, l, x)
        h = commit(model, h, l)
    return h, l, model.lm_head(h)


@torch.no_grad()
def original_rollout(model, ids, cycles):
    carry = model.initial_carry
    for _ in range(cycles):
        carry, logits = model(carry, ids)
    return carry, logits


@torch.no_grad()
def mediation(model, ids, steps=(1, 8, 24), cycles=16):
    x = model.embed(ids)
    h, l = initial_state(model, ids)
    result = []
    for step in range(cycles * model.H_cycles):
        new_l = low_update(model, h, l, x)
        new_h = commit(model, h, new_l)
        if step in steps:
            donor = replace_h(model, h, 'shuffle')
            alt_l = low_update(model, donor, l, x)
            total = commit(model, donor, alt_l)
            indirect = commit(model, h, alt_l)
            direct = commit(model, donor, new_l)
            def relative(a, b):
                return ((a.float()-b.float()).flatten(1).norm(dim=1) / b.float().flatten(1).norm(dim=1).clamp_min(1e-12)).cpu().numpy()
            result.append(dict(step=step, total=relative(total, new_h),
                               indirect=relative(indirect, new_h), direct=relative(direct, new_h),
                               mediator=relative(alt_l, new_l),
                               mediated_identity_error=relative(total, indirect)))
        h, l = new_h, new_l
    return result


def read_data(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    x = np.array([[0] + [int(c) for c in r['question'].replace('.', '0')] for r in rows], dtype=np.int64)
    y = np.array([[0] + [int(c) for c in r['answer']] for r in rows], dtype=np.int64)
    return x, y


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def paired_summary(correct, baseline, batch_size=32):
    delta = correct.astype(float) - baseline.astype(float)
    rng = np.random.default_rng(20261005)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
    # Shuffle donors couple examples within a batch. Also report a batch-cluster
    # bootstrap, rather than treating those examples as independent replicates.
    blocks = [delta[i:i+batch_size] for i in range(0, len(delta), batch_size)]
    sums = np.array([b.sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    picks = rng.integers(len(blocks), size=(2000, len(blocks)))
    cluster = sums[picks].sum(1) / sizes[picks].sum(1)
    return dict(exact=float(correct.mean()), delta_pp=float(100*delta.mean()),
                ci95_pp=(100*np.quantile(boot, [0.025, 0.975])).tolist(),
                batch_cluster_ci95_pp=(100*np.quantile(cluster, [0.025, 0.975])).tolist(),
                harmed=int(((baseline == 1) & (correct == 0)).sum()),
                helped=int(((baseline == 0) & (correct == 1)).sum()))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--data', default='downloaded-datasets/sudoku-extreme-1k/test_hard.csv')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--cycles', type=int, default=16)
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--reduction-only', action='store_true')
    p.add_argument('--hh-link', action='store_true')
    p.add_argument('--random-init', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    model, cfg = load_model(a.checkpoint, a.device, a.hh_link)
    if a.random_init:
        arch = dict(cfg['arch']) | dict(seq_len=82, vocab_size=10, is_causal=False, hh_link=a.hh_link)
        model = HRM(arch).to(a.device).eval()
    x, y = read_data(a.data)
    indices = np.random.default_rng(20261005).permutation(len(x))[:a.samples]
    conditions = [('baseline', 'baseline', 1, False)]
    if not a.baseline_only:
        if not a.hh_link:
            conditions.append(('recompute', 'recompute', 1, False))
        for mode in (() if a.reduction_only else ('reset', 'shuffle', 'norm_shuffle')):
            for at in (1, 8, 24):
                conditions.append((f'{mode}@{at}', mode, at, False))
            conditions.append((f'{mode}@1+', mode, 1, True))
    outcomes = {name: [] for name, *_ in conditions}
    preds = {name: [] for name, *_ in conditions}
    path_values = {}
    errors = dict(original_logits=0., original_h=0., original_l=0., recompute_logits=0.)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(indices), a.batch_size):
        ix = indices[start:start+a.batch_size]
        ids = torch.as_tensor(x[ix], device=a.device)
        target = y[ix]
        base_logits = None
        for name, mode, at, sustained in conditions:
            h, l, logits = rollout(model, ids, a.cycles, mode, at, sustained)
            pred = logits.argmax(-1).cpu().numpy()
            outcomes[name].append((pred == target).all(-1))
            preds[name].append(pred)
            if mode == 'baseline':
                base_logits = logits
                # Check the hand-unrolled evaluator against the actual implementation.
                if start == 0:
                    ref, ref_logits = original_rollout(model, ids, a.cycles)
                    for key, left, right in [('original_logits', logits, ref_logits), ('original_h', h, ref['z_H']), ('original_l', l, ref['z_L'])]:
                        errors[key] = float((left.float()-right.float()).abs().max())
            if mode == 'recompute':
                errors['recompute_logits'] = max(errors['recompute_logits'], float((logits.float()-base_logits.float()).abs().max()))
        if not a.baseline_only and not a.reduction_only:
            for record in mediation(model, ids, cycles=a.cycles):
                for key, value in record.items():
                    if key != 'step':
                        path_values.setdefault(f'{record["step"]}/{key}', []).append(value)
        print(f'{start+len(ix)}/{len(indices)}', flush=True)
    outcomes = {k: np.concatenate(v) for k, v in outcomes.items()}
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    path_values = {k: np.concatenate(v) for k, v in path_values.items()}
    summary = {}
    for name in outcomes:
        summary[name] = paired_summary(outcomes[name], outcomes['baseline'], a.batch_size)
        summary[name]['token_accuracy'] = float((preds[name][:, 1:] == y[indices, 1:]).mean())
        summary[name]['prediction_disagreement'] = float((preds[name] != preds['baseline']).any(-1).mean())
    result = dict(checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_sha256=digest(a.checkpoint),
                  data_sha256=digest(a.data), config=cfg, hh_link=a.hh_link, random_init=a.random_init,
                  samples=len(indices), batch_size=a.batch_size, cycles=a.cycles,
                  equality_errors=errors, conditions=summary,
                  mediation={k: dict(mean=float(v.mean()), max=float(v.max()), min=float(v.min())) for k, v in path_values.items()})
    np.savez_compressed(out/'paired.npz', indices=indices, target=y[indices],
                        **{f'pred/{k}': v for k, v in preds.items()}, **path_values)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
