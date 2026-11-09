"""Inference-only dense-clue failure screen; record every committed H update."""
from hrm.paths import analysis_directory
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT = analysis_directory(__file__)
pass
from hrm.models.hrm import HRM

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rms(x):
    return x.float().square().mean((-2, -1)).sqrt()

@torch.inference_mode()
def main():
    c = json.loads((ROOT / 'config.json').read_text())
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(c['threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(c['seed'])
    assert sha(c['checkpoint']) == c['checkpoint_sha256']
    config = yaml.safe_load((ROOT / 'sources/model_config.json').read_text())
    arch = {k: v for k, v in config['arch'].items() if k != 'name'}
    m = HRM(arch | dict(vocab_size=10, seq_len=82, is_causal=False))
    m.load_state_dict(torch.load(c['checkpoint'], map_location='cpu', weights_only=True), assign=True)
    m.eval()
    rows = list(csv.DictReader(open(c['data'])))[:c['n']]
    y = np.array([[0] + list(map(int, r['answer'])) for r in rows], dtype=np.uint8)
    original = np.array([[0] + list(map(int, r['question'].replace('.', '0'))) for r in rows], dtype=np.uint8)
    rng = np.random.default_rng(c['seed'])
    cells = rng.integers(0, 81, len(rows))
    blank_rows = rng.integers(0, 9, len(rows))
    one = y.copy()
    one[np.arange(len(y)), 1 + cells] = 0
    row = y.copy()
    for i, j in enumerate(blank_rows):
        row[i, 1 + j * 9:1 + (j + 1) * 9] = 0
    inputs = np.stack([original, one, row, y])
    np.savez_compressed(out / 'cohort.npz', inputs=inputs, labels=y, row_id=np.arange(len(y)), blank_cell=cells, blank_row=blank_rows)
    protocol = c | {'torch': torch.__version__, 'dtype': str(m.zH_init.dtype), 'L_per_H': m.L_cycles, 'H_per_native_forward': m.H_cycles, 'data_sha256': sha(c['data']), 'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve().parent / 'run.py', ROOT / 'config.json', *sorted((ROOT / 'sources').rglob('*.py')), ROOT / 'sources/model_config.json']}, 'scoring': '82-token exact incl BOS plus 81-cell exact, no clue clamping, no early stop', 'native_forward_check': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    started = time.monotonic()
    for ci, condition in enumerate(c['conditions']):
        for start in range(0, len(y), c['batch']):
            ids = torch.tensor(inputs[ci, start:start + c['batch']], dtype=torch.int32)
            x = m.embed(ids)
            h = m.zH_init.expand_as(x)
            l = m.zL_init.expand_as(x)
            preds = []
            dh = []
            dl = []
            rh = []
            rl = []
            margin = []
            for step in range(c['h_updates']):
                h0 = h
                l0 = l
                for _ in range(m.L_cycles):
                    l = m.L_level(l + h + x)
                h = m.H_level(h + l)
                if ci == 0 and start == 0 and (step == 1):
                    native, logits = m(m.initial_carry, ids)
                    assert torch.equal(h, native['z_H']) and torch.equal(l, native['z_L'])
                    protocol['native_forward_check'] = True
                    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
                logits = m.lm_head(h).float()
                p = logits.argmax(-1)
                label = torch.tensor(y[start:start + len(ids)], dtype=torch.int64)
                correct_logit = logits.gather(-1, label[..., None]).squeeze(-1)
                other = logits.clone()
                other.scatter_(-1, label[..., None], -torch.inf)
                margin.append((correct_logit - other.max(-1).values).numpy())
                preds.append(p.numpy().astype(np.uint8))
                dh.append(rms(h - h0).numpy())
                dl.append(rms(l - l0).numpy())
                rh.append((rms(h - h0) / rms(h0).clamp_min(1e-12)).numpy())
                rl.append((rms(l - l0) / rms(l0).clamp_min(1e-12)).numpy())
            np.savez_compressed(out / f'{condition}_{start:04d}.npz', row_id=np.arange(start, start + len(ids)), predictions=np.stack(preds, 1), delta_H_rms=np.stack(dh, 1), delta_L_block_rms=np.stack(dl, 1), relative_H=np.stack(rh, 1), relative_L_block=np.stack(rl, 1), true_margin=np.stack(margin, 1))
            print(json.dumps({'condition': condition, 'completed': start + len(ids), 'seconds': time.monotonic() - started}), flush=True)
    (out / 'complete.json').write_text(json.dumps({'completed': True, 'seconds': time.monotonic() - started}) + '\n')
if __name__ == '__main__':
    main()
