"""Pinned upstream IRED, original Sudoku training, and strict evaluation.

Upstream model, loss and sampler are imported unchanged. No symbolic repair.
The extreme evaluation is gated on the full original-data evaluation.
"""
from hrm.paths import analysis_directory
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from hrm.paths import repository_root
from hrm.training.data import freeze_training_seed
from hrm_ired.public_data import directory as public_directory, digest
ROOT = analysis_directory(__file__) if os.environ.get("HRM_ANALYSIS_RECORDS") else (repository_root()/"exp/sudoku/records/ired_diagnostics") / "hrm/analyse/baselines/ired"
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from ema_pytorch import EMA
from hrm_ired.models import SudokuEBM, DiffusionWrapper
from hrm_ired.diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, sudoku_accuracy
from hrm_ired.sat_dataset import SudokuDataset, SudokuRRNDataset

def model():
    return GaussianDiffusion1D(DiffusionWrapper(SudokuEBM(729, 729)), seq_length=32, timesteps=10, sampling_timesteps=10, objective='pred_noise', supervise_energy_landscape=True, use_innerloop_opt=True, show_inference_tqdm=False, sudoku=True)

def metrics(pred, target, inp, mask):
    p = pred.reshape(-1, 9, 9, 9).argmax(-1)
    y = target.reshape(-1, 9, 9, 9).argmax(-1)
    x = inp.reshape(-1, 9, 9, 9).argmax(-1)
    given = mask.reshape(-1, 9, 9, 9)[..., 0].bool()
    oh = torch.nn.functional.one_hot(p, 9)
    valid = (oh.sum(1) == 1).all((1, 2)) & (oh.sum(2) == 1).all((1, 2)) & (oh.reshape(-1, 3, 3, 3, 3, 9).sum((2, 4)) == 1).all((1, 2, 3))
    clues = ((p == x) | ~given).all((1, 2))
    result = {k: float(v) for k, v in sudoku_accuracy(pred, target, mask).items()}
    result.update(exact_match=(p == y).all((1, 2)).float().mean().item(), strict_valid_and_clues=(valid & clues).float().mean().item(), clues_preserved=clues.float().mean().item())
    return result

def evaluate(m, ds, device, batch_size, seed, output):
    torch.manual_seed(seed)
    m.eval()
    totals, n, predictions = ({}, 0, [])
    start = time.monotonic()
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        x, y, mask = [v.float().to(device) for v in batch]
        pred = m.sample(x, None, mask, batch_size=len(x))
        for k, v in metrics(pred, y, x, mask).items():
            totals[k] = totals.get(k, 0.0) + v * len(x)
        n += len(x)
        predictions.append(pred.reshape(-1, 81, 9).argmax(-1).cpu().numpy().astype('uint8') + 1)
    out = {k: v / n for k, v in totals.items()}
    out.update(n=n, seed=seed, seconds=time.monotonic() - start)
    output.write_text(json.dumps(out, indent=2))
    np.save(output.with_suffix('.predictions.npy'), np.concatenate(predictions))
    print(json.dumps({'evaluation': output.name, **out}), flush=True)
    return out

class Extreme(torch.utils.data.Dataset):

    def __init__(self):
        with np.load(ROOT / 'extreme_test_hard.npz') as data:
            self.x, self.y = (data['x'], data['y'])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.tensor(self.x[i], dtype=torch.long)
        y = torch.tensor(self.y[i], dtype=torch.long)
        f = torch.nn.functional.one_hot(x, 10)[:, 1:].float()
        label = torch.nn.functional.one_hot(y - 1, 9).float()
        mask = (x != 0)[:, None].expand(-1, 9).float()
        return ((f * 2 - 1).flatten(), (label * 2 - 1).flatten(), mask.flatten())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--eval-batch-size', type=int, default=64)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--satnet-dir',type=Path)
    ap.add_argument('--rrn-dir',type=Path)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--output', type=Path, default=ROOT / 'run')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--train-only', action='store_true')
    ap.add_argument('--resume-lr', type=float, default=None)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old_config=args.output/'config.json'
    if args.seed is None:
        saved_seed=old_config if old_config.exists() else args.output/'inputs.json'
        args.seed=json.loads(saved_seed.read_text())['seed'] if saved_seed.exists() else freeze_training_seed(args.output)
    data_root=public_directory('sudoku',args.satnet_dir)
    args.satnet_dir=data_root
    prepared=dict(seed=args.seed,satnet_files={n:digest(data_root/n) for n in ('features.pt','labels.pt')})
    preparation=args.output/'inputs.json'
    if preparation.exists() and json.loads(preparation.read_text())!=prepared:raise ValueError('Prepared seed/data changed; use a new run directory')
    preparation.write_text(json.dumps(prepared,indent=2))
    if args.prepare_only:
        print(json.dumps(prepared));return
    if args.steps<=0 or args.batch_size<=0 or args.save_every<=0:ap.error('steps, batch-size and save-every must be positive')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    m = model().to(args.device)
    opt = torch.optim.Adam(m.parameters(), lr=0.0001, betas=(0.9, 0.99))
    ema = EMA(m, beta=0.995, update_every=10).to(args.device)
    ds = SudokuDataset('sudoku', 'train', args.satnet_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    step = 0
    ckpt = args.output / 'latest.pt'
    if ckpt.exists():
        if not old_config.exists():raise ValueError('Resume requires the saved config.json')
        previous=json.loads(old_config.read_text())
        for key in ('seed','batch_size'):
            if previous[key]!=getattr(args,key):raise ValueError(f'Resume cannot change {key}')
        state = torch.load(ckpt, map_location=args.device, weights_only=False)
        m.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        ema.load_state_dict(state['ema'])
        step = state['step']
        torch.set_rng_state(state['rng'].cpu())
        if 'cuda_rng' in state and args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    if args.resume_lr is not None:
        for group in opt.param_groups:
            group['lr'] = args.resume_lr
    config = vars(args).copy()
    for key,value in config.items():
        if isinstance(value,Path):config[key]=str(value)
    config.update(upstream_commit='3d74b85fab7fcf5e28aaf15e9ed3bf51c1a1d545', torch=torch.__version__, ema_pytorch='0.7.7', train_n=len(ds), parameter_count=sum((p.numel() for p in m.parameters())), restart_note='DataLoader shuffle restarts; resume is not bitwise identical.')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    start = time.monotonic()
    start_step = step
    while step < args.steps:
        for batch in loader:
            m.train()
            x, y, mask = [v.float().to(args.device) for v in batch]
            opt.zero_grad(set_to_none=True)
            loss, parts = m(x, y, mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss at {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            ema.update()
            step += 1
            if step <= 3 or step % 100 == 0:
                status = dict(step=step, loss=loss.item(), seconds=time.monotonic() - start, seconds_per_step=(time.monotonic() - start) / (step - start_step))
                (args.output / 'status.json').write_text(json.dumps(status, indent=2))
                print(json.dumps(status), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                state = dict(step=step, model=m.state_dict(), opt=opt.state_dict(), ema=ema.state_dict(), rng=torch.get_rng_state())
                if args.device.startswith('cuda'):
                    state['cuda_rng'] = torch.cuda.get_rng_state_all()
                temp = ckpt.with_suffix('.tmp')
                torch.save(state, temp)
                temp.replace(ckpt)
            if step >= args.steps:
                break
    if args.train_only:
        return
    if args.smoke:
        evaluate(ema.ema_model, Subset(SudokuDataset('sudoku', 'val', args.satnet_dir), range(1)), args.device, 1, args.seed, args.output / 'smoke_eval.json')
        return
    id_result = evaluate(ema.ema_model, SudokuDataset('sudoku', 'val', args.satnet_dir), args.device, args.eval_batch_size, args.seed, args.output / 'satnet.json')
    ood_result = evaluate(ema.ema_model, SudokuRRNDataset('sudoku-rrn', 'test', public_directory('sudoku-rrn',args.rrn_dir)), args.device, args.eval_batch_size, args.seed, args.output / 'rrn.json')
    passed = abs(id_result['strict_valid_and_clues'] - 0.994) <= 0.01 and abs(ood_result['strict_valid_and_clues'] - 0.621) <= 0.03
    gate = dict(passed=passed, metric='strict_valid_and_clues', id_target=0.994, id_tolerance=0.01, rrn_target=0.621, rrn_tolerance=0.03, note='Practical agreement only; reference seed/checkpoint/metric provenance unclear.')
    (args.output / 'reproduction_gate.json').write_text(json.dumps(gate, indent=2))
    if not passed:
        print('Original-data reproduction gate failed. Extreme evaluation not started.', flush=True)
        return
    from hrm.analyse.baselines.ired.eval_extreme import data as extreme_data
    extreme,_=extreme_data()
    evaluate(ema.ema_model, Subset(extreme,range(20000)), args.device, args.eval_batch_size, args.seed, args.output / 'extreme_test_hard.json')
if __name__ == '__main__':
    main()
