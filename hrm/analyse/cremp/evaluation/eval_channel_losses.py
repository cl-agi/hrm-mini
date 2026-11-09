"""Read-only fixed-noise validation decomposed by graph-defined angle support."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

def loss_terms(prediction, target):
    delta = (target - prediction + torch.pi) % (2 * torch.pi) - torch.pi
    absolute = delta.abs()
    beta = torch.pi / 10
    return torch.where(absolute < beta, 0.5 * delta.square() / beta, absolute - 0.5 * beta)

def support_masks(batch):
    valid = batch['attn_mask'].bool().unsqueeze(-1).expand_as(batch['feat_mask'])
    backbone = torch.zeros_like(valid)
    backbone[..., :2] = valid[..., :2]
    existing = valid & batch['feat_mask'].bool() & ~backbone
    missing = valid & ~batch['feat_mask'].bool() & ~backbone
    assert torch.equal(backbone.to(torch.int8) + existing + missing, valid.to(torch.int8))
    assert bool(batch['feat_mask'][..., :2][batch['attn_mask'].bool()].all())
    return {'backbone': backbone, 'existing_sidechain': existing, 'missing_sidechain': missing}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--training-metrics', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch', type=int, default=64)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.output.exists():
        raise ValueError('Refusing to overwrite loss-decomposition evidence')
    pass
    from hrm_cremp.angle_data import AngleDataset, sha
    from hrm_cremp.angle_model import make_model
    from hrm_cremp.angle_diffusion import noised_dataset, noise_loss
    from hrm_cremp.ringer_support import FEATURES
    before = sha(a.checkpoint)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = make_model(state['config']).to(a.device)
    model.load_state_dict(state['model'])
    model.eval()
    ds = AngleDataset(a.data, state['means'])
    torch.manual_seed(8173)
    if str(a.device).startswith('cuda'):
        torch.cuda.synchronize()
    start = time.perf_counter()
    sums = np.zeros(12, np.float64)
    counts = np.zeros(12, np.int64)
    groups = {k: {'sum': 0.0, 'count': 0, 'channel_sums': np.zeros(12, np.float64), 'channel_counts': np.zeros(12, np.int64)} for k in ['backbone', 'existing_sidechain', 'missing_sidechain']}
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(noised_dataset(ds), batch_size=a.batch)):
            batch = {k: v.to(a.device) for k, v in batch.items()}
            pred = model(batch['corrupted'], batch['t'], attention_mask=batch['attn_mask'], atom_features=batch['atom_features'])
            terms = loss_terms(pred, batch['known_noise'])
            valid = batch['attn_mask'].bool().unsqueeze(-1).expand_as(terms)
            if i == 0:
                reconstructed = (terms * valid).sum((0, 1)) / valid.sum((0, 1))
                assert torch.allclose(reconstructed.mean(), noise_loss(pred, batch), atol=1e-07, rtol=1e-06)
            sums += (terms.double() * valid).sum((0, 1)).cpu().numpy()
            counts += valid.sum((0, 1)).cpu().numpy()
            for name, mask in support_masks(batch).items():
                g = groups[name]
                cs = (terms.double() * mask).sum((0, 1)).cpu().numpy()
                cc = mask.sum((0, 1)).cpu().numpy()
                g['channel_sums'] += cs
                g['channel_counts'] += cc
                g['sum'] += float(cs.sum())
                g['count'] += int(cc.sum())
    if str(a.device).startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    aggregate = float(np.mean(sums / counts))
    records = json.loads(a.training_metrics.read_text())
    expected = next((r['val_loss'] for r in records if r['step'] == state['step']))
    if abs(aggregate - expected) > 1e-06:
        raise ValueError(f'Validation stream mismatch: {aggregate} vs {expected}')
    for g in groups.values():
        g['mean_per_supported_scalar'] = g['sum'] / g['count']
        c = g['channel_counts']
        s = g['channel_sums']
        g['channel_means'] = [float(x / y) if y else None for x, y in zip(s, c)]
        g['mean_of_nonempty_channel_means'] = float(np.mean(s[c > 0] / c[c > 0]))
        g['channel_sums'] = s.tolist()
        g['channel_counts'] = c.tolist()
    pooled = sum((g['sum'] for g in groups.values())) / sum((g['count'] for g in groups.values()))
    assert abs(pooled - aggregate) < 1e-12
    assert sha(a.checkpoint) == before
    result = dict(checkpoint_sha256=before, checkpoint_step=state['step'], data_sha256=sha(a.data), evaluation_seconds=elapsed, batch=a.batch, noise_seed=8173, angular_loss='Wrapped smooth L1, beta=pi/10, no circle penalty', aggregation='Primary group mean weights each supported scalar equally; channel means/counts also provided', aggregate_original=aggregate, training_record_val_loss=expected, aggregate_difference=aggregate - expected, features=list(FEATURES), channel_means=(sums / counts).tolist(), channel_counts=counts.tolist(), groups=groups, model_training_unchanged=True, reference_usage='Supervised validation only; no generation or checkpoint selection changes')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['groups', 'features', 'channel_means', 'channel_counts']}), flush=True)
if __name__ == '__main__':
    main()
