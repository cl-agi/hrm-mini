"""Frozen HRM, shared per-token linear readout of H+L, puzzle-disjoint evaluation.

No gradients or probe predictions enter the recurrent model. Validation chooses
training steps; test puzzles are only evaluated after the choice is frozen.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
CYCLES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

def digits(frame):

    def parse(col):
        return np.stack([np.pad(np.frombuffer(s.replace('.', '0').encode(), dtype=np.uint8).astype(np.int64) - 48, (1, 0)) for s in frame[col]])
    return (parse('question'), parse('answer'))

def model_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def gather(model, x_all, y_all, ids, rng, name, device="cuda", horizon=64, cells_per_board=24, batch_size=32):
    hs = []
    ls = []
    labels = []
    phases = []
    puzzle = []
    cycles = []
    blanks = []
    for start in range(0, len(ids), batch_size):
        ii = ids[start:start + batch_size]
        x = torch.tensor(x_all[ii], device=device)
        y = torch.tensor(y_all[ii], device=device)
        emb = model.embed(x)
        h = model.zH_init.expand_as(emb).clone()
        l = model.zL_init.expand_as(emb).clone()
        for c in range(1, horizon+1):
            for slot in range(1, 8):
                if slot < 7:
                    l = model.L_level(l + h + emb)
                else:
                    h = model.H_level(h + l)
                if c not in CYCLES:
                    continue
                cells = torch.tensor(np.stack([rng.choice(np.arange(1, 82), cells_per_board, replace=False) for _ in ii]), device=x.device)
                bi = torch.arange(len(ii), device=x.device)[:, None]
                hs.append(h[bi, cells].reshape(-1, h.shape[-1]).cpu())
                ls.append(l[bi, cells].reshape(-1, h.shape[-1]).cpu())
                labels.append(y[bi, cells].flatten().cpu())
                blanks.append(x[bi, cells].eq(0).flatten().cpu())
                phases.append(torch.full((len(ii) * cells_per_board,), slot, dtype=torch.uint8))
                puzzle.append(torch.tensor(np.repeat(ii, cells_per_board), dtype=torch.int32))
                cycles.append(torch.full((len(ii) * cells_per_board,), c, dtype=torch.uint8))
        print(f'cache {name}: {min(start + batch_size, len(ids))}/{len(ids)} puzzles', flush=True)
    return dict(H=torch.cat(hs), L=torch.cat(ls), y=torch.cat(labels), slot=torch.cat(phases), puzzle=torch.cat(puzzle), cycle=torch.cat(cycles), blank=torch.cat(blanks))

@torch.no_grad()
def validation(weights, features, y, slot, blank):
    result = {}
    for name, w in weights.items():
        totals = {s: np.zeros(4, dtype=np.float64) for s in range(1, 8)}
        for i in range(0, len(y), 32768):
            sl = slice(i, i + 32768)
            v = features['sum' if name.startswith('sum') else 'H'][sl].float()
            logits = F.linear(v, w)
            loss = F.cross_entropy(logits, y[sl], reduction='none')
            correct = logits.argmax(-1).eq(y[sl])
            for s in range(1, 8):
                m = slot[sl] == s
                b = m & blank[sl]
                totals[s] += np.array([float(loss[m].sum()), int(m.sum()), int(correct[b].sum()), int(b.sum())])
        r = {str(s): dict(ce=float(v[0] / v[1]), blank_accuracy=float(v[2] / v[3]) if v[3] else None) for s, v in totals.items()}
        r['balanced_ce'] = float((r['7']['ce'] + np.mean([r[str(s)]['ce'] for s in range(1, 7)])) / 2)
        result[name] = r
    return result

def fit(train, val, steps, output, seed, device="cuda", half_batch=2048):

    def gpu_cache(a):
        h = a['H'].to(device)
        l = a['L'].to(device)
        features = dict(sum=h.float() + l.float(), H=h)
        return (features, a['y'].to(device), a['slot'].to(device), a['blank'].to(device))
    tf, ty, ts, tb = gpu_cache(train)
    vf, vy, vs, vb = gpu_cache(val)
    names = ['sum_seed0', 'sum_seed1', 'sum_seed2', 'H_control']
    weights = {}
    for j, name in enumerate(names):
        gen = torch.Generator(device=device).manual_seed(seed + j)
        weights[name] = nn.Parameter(torch.randn((10, train['H'].shape[-1]), generator=gen, device=device) * 0.01)
    optimizer = torch.optim.AdamW(list(weights.values()), lr=0.003, weight_decay=0.01)
    ids_h = torch.where(ts == 7)[0]
    ids_l = torch.where(ts < 7)[0]
    gen = torch.Generator(device=device).manual_seed(seed + 99)
    best = {n: float('inf') for n in names}
    saved = {}
    history = []
    for step in range(1, steps + 1):
        ix = torch.cat([ids_h[torch.randint(len(ids_h), (half_batch,), device=device, generator=gen)], ids_l[torch.randint(len(ids_l), (half_batch,), device=device, generator=gen)]])
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for name, w in weights.items():
            features = tf['sum' if name.startswith('sum') else 'H'][ix].float()
            losses.append(F.cross_entropy(F.linear(features, w), ty[ix]))
        torch.stack(losses).sum().backward()
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            result = validation(weights, vf, vy, vs, vb)
            for name in names:
                if result[name]['balanced_ce'] < best[name]:
                    best[name] = result[name]['balanced_ce']
                    saved[name] = dict(weight=weights[name].detach().cpu().clone(), step=step, validation=result[name])
            history.append(dict(step=step, validation=result, train_ce={n: float(v.detach()) for n, v in zip(names, losses)}))
            print('fit', step, {n: round(result[n]['balanced_ce'], 5) for n in names}, flush=True)
    selected = min([n for n in names if n.startswith('sum')], key=lambda n: best[n])
    torch.save(dict(heads=saved, selected=selected, initialization_seeds={n:seed+j for j,n in enumerate(names)}, sampling_seed=seed+99, architecture=f"Linear({train['H'].shape[-1]},10,bias=False)", input='H.float()+L.float()', frozen=True), Path(output) / 'secondary_heads.pt')
    (Path(output) / 'training_history.json').write_text(json.dumps(history, indent=2))
    return (saved, selected)

@torch.no_grad()
def evaluate_batch(model, weights, xn, yn, cycles=64, device="cuda"):
    x = torch.tensor(xn, device=device)
    y = torch.tensor(yn[:, 1:], device=x.device)
    e = model.embed(x)
    h = model.zH_init.expand_as(e).clone()
    l = model.zL_init.expand_as(e).clone()
    out = {n: [] for n in ['native', *weights]}
    ce = {n: [] for n in out}
    rms = []
    steps = []
    margin = []
    blank = x[:, 1:] == 0

    def record(c, s, r):
        logits = {'native': model.lm_head(h)[:, 1:].float()}
        for name, w in weights.items():
            v = h[:, 1:].float() + l[:, 1:].float() if name.startswith('sum') else h[:, 1:].float()
            logits[name] = F.linear(v, w)
        for n, v in logits.items():
            out[n].append(v.argmax(-1).to(torch.uint8).cpu().numpy())
            token_ce = F.cross_entropy(v.reshape(-1, 10), y.reshape(-1), reduction='none').reshape_as(y)
            ce[n].append(((token_ce * blank).sum(1) / blank.sum(1)).cpu().numpy())
        rms.append(r)
        steps.append((c, s))
    record(0, 0, np.full(len(xn), np.nan))
    for c in range(1, cycles + 1):
        for s in range(1, 8):
            if s < 7:
                old = l
                l = model.L_level(l + h + e)
                r = (l.float() - old.float()).square().mean((1, 2)).sqrt()
            else:
                old = h
                h = model.H_level(h + l)
                r = (h.float() - old.float()).square().mean((1, 2)).sqrt()
            record(c, s, r.cpu().numpy())
    return dict(pred={n: np.stack(v, axis=1) for n, v in out.items()}, ce={n: np.stack(v, axis=1) for n, v in ce.items()}, rms=np.stack(rms, axis=1), steps=np.array(steps))


def main():
    from hrm.analyse.dynamics.answer_direction.readout import main as entry
    return entry()

if __name__ == '__main__':
    main()
