"""Matched inference interventions on a frozen cohort and checkpoint; no training."""
from hrm.paths import analysis_directory
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
ROOT = analysis_directory(__file__)
BASE = ROOT.parent / 'near_complete_failure_20261025'
pass
from hrm.models.hrm import HRM

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rms(x):
    return x.float().square().mean((-2, -1)).sqrt()

def blend(old, proposal, alpha, normalize):
    if alpha == 1:
        return proposal
    result = (old.float() + alpha * (proposal.float() - old.float())).to(proposal.dtype)
    if normalize:
        result = F.rms_norm(result.float(), (result.shape[-1],), eps=1e-06).to(proposal.dtype)
    return result

@torch.inference_mode()
def main():
    c = json.loads((ROOT / 'config.json').read_text())
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(c['threads'])
    torch.set_num_interop_threads(1)
    base = json.loads((BASE / 'config.json').read_text())
    co = np.load(BASE / 'results/cohort.npz')
    assert sha(base['checkpoint']) == base['checkpoint_sha256']
    assert sha(BASE / 'results/cohort.npz') == c['cohort_sha256']
    cfg = yaml.safe_load((BASE / 'sources/model_config.json').read_text())
    rows = np.array(c['rows'])
    labels = co['labels'][rows]
    cells = co['blank_cell'][rows] + 1
    cells = np.array(c['target_tokens'])
    inputs = labels.copy()
    for i, t in enumerate(cells):
        inputs[i, t] = 0
    np.savez_compressed(out / 'cohort.npz', row_id=rows, input_ids=inputs, labels=labels, target_token=cells)
    protocol = c | {'checkpoint': base['checkpoint'], 'checkpoint_sha256': base['checkpoint_sha256'], 'torch': torch.__version__, 'collector_sha256': sha(__file__), 'architecture': cfg, 'source_sha256': {str(p.relative_to(BASE)): sha(p) for p in sorted((BASE / 'sources').rglob('*.py')) if not p.name.startswith('._')}, 'definition': 'Interpolate old state toward proposed module output at every selected H or individual L update. FP32 interpolation arithmetic then cast to recurrent dtype. renorm variants additionally apply per-token RMS normalization. No input feedback or answer stopping.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for v in c['variants']:
        if (out / (v['name'] + '.npz')).exists():
            continue
        torch.manual_seed(v['seed'])
        arch = {k: x for k, x in cfg['arch'].items() if k != 'name'}
        arch['forward_dtype'] = v['precision']
        m = HRM(arch | dict(vocab_size=10, seq_len=82, is_causal=False))
        m.load_state_dict(torch.load(base['checkpoint'], map_location='cpu', weights_only=True), assign=True)
        if v['precision'] == 'float32':
            m.float()
        m.eval()
        dtype = getattr(torch, v['precision'])
        assert m.zH_init.dtype == dtype and m.embed.cast_to == dtype
        weights = m.lm_head.weight.to(dtype).float().cpu().numpy()
        parts = []
        started = time.monotonic()
        for start in range(0, len(rows), c['batch']):
            ids = torch.tensor(inputs[start:start + c['batch']], dtype=torch.int32)
            x = m.embed(ids)
            bs = len(ids)
            index = torch.arange(bs)
            token = torch.tensor(cells[start:start + bs])
            h = m.zH_init.expand_as(x)
            l = m.zL_init.expand_as(x)
            data = {k: [] for k in ['predictions', 'target_logits', 'H_target', 'L_target', 'delta_H_rms', 'delta_H_target_rms', 'delta_H_other_rms', 'delta_L_block_rms', 'L_substep_rms', 'L_substep_target_rms', 'H_rms', 'L_rms', 'cos_delta_H', 'H_displacement_from_H4', 'H_pathlength_after_H4']}
            prev_delta = None
            anchor = None
            path = torch.zeros(bs)
            for step in range(1, v['h_updates'] + 1):
                active = v['first_H'] <= step <= v['last_H']
                alpha = v['alpha'] if active else 1.0
                ah = alpha if 'H' in v['modules'] else 1.0
                al = alpha if 'L' in v['modules'] else 1.0
                h0 = h
                l0 = l
                sub = []
                subtarget = []
                for _ in range(m.L_cycles):
                    old = l
                    l = blend(old, m.L_level(old + h + x), al, v['renorm'])
                    d = l.float() - old.float()
                    sub.append(rms(d))
                    subtarget.append(d[index, token].square().mean(-1).sqrt())
                h = blend(h0, m.H_level(h0 + l), ah, v['renorm'])
                delta = h.float() - h0.float()
                logits = m.lm_head(h).float()
                target = h[index, token].float()
                targetdelta = delta[index, token].square().mean(-1).sqrt()
                total = delta.square().sum((1, 2))
                others = ((total - delta[index, token].square().sum(-1)) / (81 * delta.shape[-1])).clamp_min(0).sqrt()
                cos = torch.zeros(bs) if prev_delta is None else F.cosine_similarity(delta.flatten(1), prev_delta.flatten(1), dim=1)
                if step == 4:
                    anchor = h.float().clone()
                if step > 4:
                    path += delta.flatten(1).norm(dim=1)
                disp = torch.zeros(bs) if anchor is None else (h.float() - anchor).flatten(1).norm(dim=1)
                values = [logits.argmax(-1).numpy().astype(np.uint8), logits[index, token].numpy(), target.numpy(), l[index, token].float().numpy(), rms(delta).numpy(), targetdelta.numpy(), others.numpy(), rms(l.float() - l0.float()).numpy(), torch.stack(sub, 1).numpy(), torch.stack(subtarget, 1).numpy(), rms(h).numpy(), rms(l).numpy(), cos.numpy(), disp.numpy(), path.numpy().copy()]
                for k, value in zip(data, values):
                    data[k].append(value)
                prev_delta = delta
            parts.append({k: np.stack(value, 1) for k, value in data.items()})
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez_compressed(out / (v['name'] + '.npz'), row_id=rows, readout_weight=weights, H_init=m.zH_init.float().numpy(), L_init=m.zL_init.float().numpy(), **arrays)
        (out / (v['name'] + '.json')).write_text(json.dumps(v | {'seconds': time.monotonic() - started, 'completed': True}, indent=2) + '\n')
        exact = (arrays['predictions'] == labels[:, None, :]).all(-1)
        print(json.dumps({'variant': v['name'], 'seconds': time.monotonic() - started, 'final_correct': int(exact[:, -1].sum()), 'rows': rows.tolist(), 'final_flags': exact[:, -1].tolist()}), flush=True)
    (out / 'complete.json').write_text(json.dumps({'complete': True, 'variants': len(c['variants'])}) + '\n')
if __name__ == '__main__':
    main()
