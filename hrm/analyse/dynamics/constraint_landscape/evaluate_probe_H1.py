"""Evaluate the saved L-only probe on a tightly framed H1 terminal grid."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from hrm.analyse.dynamics.constraint_landscape.global_fit import load_global


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stats(value):
    value = np.asarray(value)
    return dict(min=float(value.min()), max=float(value.max()),
                mean=float(value.mean()), std=float(value.std()),
                p95_absolute=float(np.quantile(np.abs(value), .95)))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    probe = torch.load(args.probe, map_location='cpu', weights_only=True)
    prepare_audit = json.loads((args.results/'prepare_audit.json').read_text())
    full_audit = json.loads((args.results/'full_plane/kernel_audit.json').read_text())
    probe_audit = json.loads((args.probe.parent/'critic_audit.json').read_text())
    checkpoint_hash = probe['checkpoint_sha256']
    assert checkpoint_hash == prepare_audit['checkpoint'] == full_audit['checkpoint_sha256']
    assert sha256(probe_audit['checkpoint']['checkpoint']) == checkpoint_hash
    model_path = args.results/'kernel_model.pt'
    assert sha256(model_path) == full_audit['model_sha256']
    model = load_global(model_path, device='cuda')
    prepared_path = args.results/'prepared.npz'
    d = np.load(prepared_path)
    q, phase = d['q'], d['phase']
    tail = q[phase == 1][-1024:]
    lo, hi = tail.min(0), tail.max(0)
    span = hi-lo
    pad = np.maximum(.4*span, .25*span.max()) if span.max() > 0 else np.full(2, 1e-5)
    lower, upper = lo-pad, hi+pad
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 301),
                         np.linspace(lower[1], upper[1], 301))
    print('TAIL_GEOMETRY', json.dumps(dict(lo=lo.tolist(), hi=hi.tolist(), span=span.tolist(),
          padding=pad.tolist(), lower=lower.tolist(), upper=upper.tolist())), flush=True)
    base, basis, normal = [torch.tensor(d[k], dtype=torch.float32, device='cuda')
                           for k in ('L_base', 'B', 'L_V')]
    weight = probe['linear_L']['weight'].to(device='cuda', dtype=torch.float64)
    bias = probe['linear_L']['bias'].to(device='cuda', dtype=torch.float64)
    assert base.shape == (82*512,) and weight.shape == (512,) and bias.shape == ()
    rank, sqrt_dim = normal.shape[1], (82*512)**.5

    def evaluate(query):
        quantized, before_quantization = [], []
        for begin in range(0, len(query), args.batch):
            points = query[begin:begin+args.batch]
            coeff = torch.tensor(model.predict(points), dtype=torch.float32, device='cuda')
            xy = torch.tensor(points, dtype=torch.float32, device='cuda')
            zL = base + sqrt_dim*(xy @ basis.T + coeff[:, :rank] @ normal.T)
            zL = zL.reshape(-1, 82, 512)
            before_quantization.append((zL.mean(1).double() @ weight + bias).cpu().numpy())
            zL = zL.to(torch.bfloat16)
            quantized.append((zL.float().mean(1).double() @ weight + bias).cpu().numpy())
        return np.concatenate(quantized), np.concatenate(before_quantization)

    value, unquantized = evaluate(np.column_stack([xx.ravel(), yy.ravel()]))
    value, unquantized = value.reshape(xx.shape), unquantized.reshape(xx.shape)
    assert np.isfinite(value).all() and np.isfinite(unquantized).all()
    tail_value, tail_unquantized = evaluate(tail)
    out = args.results/'existing_L_probe'
    out.mkdir(exist_ok=True)
    path = out/'existing_L_probe_H1.npz'
    np.savez_compressed(path, x=xx, y=yy, probe=value,
                        probe_before_bf16=unquantized, tail_q=tail,
                        tail_probe=tail_value, tail_probe_before_bf16=tail_unquantized)
    quantization_delta = value-unquantized
    differences = {label: {axis: stats(np.abs(np.diff(array, axis=index)))
                          for axis, index in [('y', 0), ('x', 1)]}
                   for label, array in [('bf16_probe', value), ('before_bf16_probe', unquantized)]}
    audit = dict(
        checkpoint_sha256=checkpoint_hash, kernel_model_sha256=sha256(model_path),
        probe_sha256=sha256(args.probe), prepared_sha256=sha256(prepared_path),
        script_sha256=sha256(__file__), matrix_sha256=sha256(path),
        lift_refitted=False, probe_refitted=False, negative_values_clipped=False,
        formula='Existing kernel global f(q).L -> bf16 -> float32 mean over all82 tokens incl CLS -> float64 affine probe',
        target='Predicted excess probability mass over27 Sudoku units; not an original checkpoint critic head',
        tail_steps=[3073, 4096], n_tail_points=len(tail),
        tail_bounds=[lo.tolist(), hi.tolist()], tail_span=span.tolist(),
        padding=pad.tolist(), padding_rule='max(0.4*axis_span,0.25*max_span); all-zero span uses1e-5',
        grid_shape=list(value.shape), grid_bounds=[lower.tolist(), upper.tolist()],
        grid_spacing=((upper-lower)/300).tolist(),
        grid_probe=stats(value), grid_probe_before_bf16=stats(unquantized),
        negative_count=int((value < 0).sum()),
        tail_probe_on_lift=stats(tail_value), tail_probe_on_lift_before_bf16=stats(tail_unquantized),
        bf16_roundtrip_delta=stats(quantization_delta),
        bf16_roundtrip_mean_absolute_change=float(np.abs(quantization_delta).mean()),
        bf16_roundtrip_max_absolute_change=float(np.abs(quantization_delta).max()),
        neighbor_absolute_differences=differences,
        display_instruction='Plot probe directly without smoothing. probe_before_bf16 is only a numerical diagnostic.',
        seconds=time.monotonic()-start)
    (out/'existing_L_probe_H1_metadata.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('COMPLETE', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
