"""Shared IRED Sudoku model and scoring; no analysis-record prerequisites."""
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from hrm_ired.models import SudokuEBM, DiffusionWrapper
from hrm_ired.diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, sudoku_accuracy
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
