"""Controlled checkpoint diagnostics; fixed subsamples, no extreme tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from hrm.training.ired_core import model, evaluate
from hrm_ired.sat_dataset import SudokuDataset, SudokuRRNDataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', choices=['ema','raw'], default='ema')
    p.add_argument('--inner-steps', type=int, default=20)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--splits', nargs='+', choices=['train','satnet','rrn'], default=['train','satnet','rrn'])
    p.add_argument('--device',default='cuda')
    p.add_argument('--skip-denoise-step', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, help='Directory containing satnet/ and sudoku-rrn/; omitted sources download from upstream')
    a = p.parse_args()
    if a.n<1 or a.inner_steps<1:p.error('n and inner-steps must be positive')
    from hrm_ired.public_data import directory
    def dataset(name):
        identifier='sudoku-rrn' if name=='rrn' else 'sudoku'
        folder='sudoku-rrn' if name=='rrn' else 'satnet'
        root=directory(identifier,a.data_root/folder if a.data_root else None)
        return SudokuRRNDataset(identifier,'test',root) if name=='rrn' else SudokuDataset(identifier,'train' if name=='train' else 'val',root)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    a.output.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    m = model()
    state = ckpt['model'] if a.weights == 'raw' else {
        k.removeprefix('ema_model.'): v for k,v in ckpt['ema'].items() if k.startswith('ema_model.')}
    m.load_state_dict(state, strict=True)
    m.to(a.device).eval()
    original = m.opt_step
    def opt_step(*args, **kwargs):
        kwargs['step'] = a.inner_steps
        return original(*args, **kwargs)
    m.opt_step = opt_step
    if a.skip_denoise_step:
        def passthrough(inp, img, t, *args, **kwargs):
            # Preserve upstream random-number consumption for paired initial noise.
            if t > 0:
                torch.randn_like(img)
            return img, img
        m.p_sample = passthrough
    info = dict(weights=a.weights, inner_steps=a.inner_steps, checkpoint=str(a.checkpoint),
                checkpoint_step=ckpt['step'], torch=torch.__version__, n_requested=a.n,
                skip_denoise_step=a.skip_denoise_step, splits=a.splits,subset_seed=20261005,evaluation_seed=42,
                data_root=str(a.data_root) if a.data_root else "verified public upstream cache")
    (a.output/'config.json').write_text(json.dumps(info, indent=2))
    for name in a.splits:
        ds=dataset(name)
        indices = np.random.default_rng(20261005).permutation(len(ds))[:min(a.n,len(ds))]
        np.save(a.output/(name+'.indices.npy'),indices)
        evaluate(m, Subset(ds, indices.tolist()), a.device, 64, 42, a.output/(name+'.json'))

if __name__ == '__main__':
    main()
