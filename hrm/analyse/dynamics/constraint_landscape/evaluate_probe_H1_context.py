"""Add a wider H1 context grid without changing the existing tight H1 grid."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from hrm.analyse.dynamics.constraint_landscape.global_fit import load_global
from hrm.analyse.dynamics.constraint_landscape.evaluate_probe_H1 import sha256, stats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    out = args.results/'existing_L_probe'
    tight_path = out/'existing_L_probe_H1.npz'
    tight_hash = sha256(tight_path)
    tight_audit = json.loads((out/'existing_L_probe_H1_metadata.json').read_text())
    probe = torch.load(args.probe, map_location='cpu', weights_only=True)
    model_path = args.results/'kernel_model.pt'
    prepared_path = args.results/'prepared.npz'
    assert probe['checkpoint_sha256'] == tight_audit['checkpoint_sha256']
    assert sha256(args.probe) == tight_audit['probe_sha256']
    assert sha256(model_path) == tight_audit['kernel_model_sha256']
    assert sha256(prepared_path) == tight_audit['prepared_sha256']
    model = load_global(model_path, device='cuda')
    d = np.load(prepared_path)
    tail = d['q'][d['phase'] == 1][-1024:]
    lo, hi = tail.min(0), tail.max(0)
    span = hi-lo
    pad = np.maximum(.4*span, .002)
    xx, yy = np.meshgrid(np.linspace(lo[0]-pad[0], hi[0]+pad[0], 301),
                         np.linspace(lo[1]-pad[1], hi[1]+pad[1], 301))
    query = np.column_stack([xx.ravel(), yy.ravel()])
    base, basis, normal = [torch.tensor(d[k], dtype=torch.float32, device='cuda')
                           for k in ('L_base', 'B', 'L_V')]
    weight = probe['linear_L']['weight'].to(device='cuda', dtype=torch.float64)
    bias = probe['linear_L']['bias'].to(device='cuda', dtype=torch.float64)
    assert base.shape == (82*512,) and weight.shape == (512,) and bias.shape == ()
    rank, sqrt_dim = normal.shape[1], (82*512)**.5
    quantized, unquantized = [], []
    for begin in range(0, len(query), 256):
        points = query[begin:begin+256]
        coeff = torch.tensor(model.predict(points), dtype=torch.float32, device='cuda')
        xy = torch.tensor(points, dtype=torch.float32, device='cuda')
        zL = (base + sqrt_dim*(xy @ basis.T + coeff[:, :rank] @ normal.T)).reshape(-1, 82, 512)
        unquantized.append((zL.mean(1).double() @ weight + bias).cpu().numpy())
        quantized.append((zL.to(torch.bfloat16).float().mean(1).double() @ weight + bias).cpu().numpy())
    value = np.concatenate(quantized).reshape(xx.shape)
    before = np.concatenate(unquantized).reshape(xx.shape)
    assert np.isfinite(value).all() and np.isfinite(before).all()
    path = out/'existing_L_probe_H1_context.npz'
    np.savez_compressed(path, x=xx, y=yy, probe=value, probe_before_bf16=before, tail_q=tail)
    assert sha256(tight_path) == tight_hash
    audit = dict(
        checkpoint_sha256=probe['checkpoint_sha256'], kernel_model_sha256=sha256(model_path),
        probe_sha256=sha256(args.probe), prepared_sha256=sha256(prepared_path),
        script_sha256=sha256(__file__), helper_sha256=sha256(Path(__file__).with_name('evaluate_probe_H1.py')),
        matrix_sha256=sha256(path), tight_grid_unchanged_sha256=tight_hash,
        lift_refitted=False, probe_refitted=False, negative_values_clipped=False,
        formula=tight_audit['formula'], target=tight_audit['target'],
        tail_steps=[3073, 4096], n_tail_points=len(tail), tail_span=span.tolist(),
        tail_bounds=[lo.tolist(), hi.tolist()], padding=pad.tolist(),
        padding_rule='max(0.4*axis_span,0.002)', grid_shape=list(value.shape),
        grid_bounds=[(lo-pad).tolist(), (hi+pad).tolist()], grid_spacing=((span+2*pad)/300).tolist(),
        tail_span_in_grid_cells=(span/((span+2*pad)/300)).tolist(),
        grid_probe=stats(value), grid_probe_before_bf16=stats(before), negative_count=int((value < 0).sum()),
        bf16_roundtrip_delta=stats(value-before),
        bf16_roundtrip_mean_absolute_change=float(np.abs(value-before).mean()),
        bf16_roundtrip_max_absolute_change=float(np.abs(value-before).max()),
        tail_probe_on_lift=tight_audit['tail_probe_on_lift'],
        display_instruction='Plot probe directly without smoothing. probe_before_bf16 is only a numerical diagnostic.',
        seconds=time.monotonic()-start)
    (out/'existing_L_probe_H1_context_metadata.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('COMPLETE', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
