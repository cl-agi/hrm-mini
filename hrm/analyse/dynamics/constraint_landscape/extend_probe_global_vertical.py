"""Extend the existing global probe grid by 15% per vertical side after horizontal extension."""
from hrm.paths import repo_path, work_path
from pathlib import Path
from hrm.paths import repo_path
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from hrm.analyse.dynamics.constraint_landscape.global_fit import load_global
ROOT = Path(str(work_path('case6566_global_lift/results')))
PROBE = Path(str(repo_path('checkpoint/dynamics/case6566_landscape/critic_result/probes.pt')))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    source = ROOT / 'existing_L_probe/existing_L_probe_global_wide.npz'
    old = np.load(source)
    probe = torch.load(PROBE, map_location='cpu', weights_only=True)
    audit = json.loads((ROOT / 'existing_L_probe/existing_L_probe_H1_context_metadata.json').read_text())
    for key, path in [('probe_sha256', PROBE), ('kernel_model_sha256', ROOT / 'kernel_model.pt'), ('prepared_sha256', ROOT / 'prepared.npz')]:
        assert sha(path) == audit[key]
    model = load_global(ROOT / 'kernel_model.pt', device='cuda')
    d = np.load(ROOT / 'prepared.npz')
    base, basis, normal = [torch.tensor(d[k], dtype=torch.float32, device='cuda') for k in ('L_base', 'B', 'L_V')]
    weight = probe['linear_L']['weight'].to(device='cuda', dtype=torch.float64)
    bias = probe['linear_L']['bias'].to(device='cuda', dtype=torch.float64)
    xs, ys = (old['x'][0], old['y'][:, 0])
    n = int(np.ceil(0.15 * (len(ys) - 1)))
    padding = 0.15 * (ys[-1] - ys[0])
    extra_y = np.r_[np.linspace(ys[0] - padding, ys[0], n + 1)[:-1], np.linspace(ys[-1], ys[-1] + padding, n + 1)[1:]]
    x, y = np.meshgrid(xs, extra_y)
    points = np.column_stack([x.ravel(), y.ravel()])
    values = []
    for start in range(0, len(points), 256):
        q = points[start:start + 256]
        c = torch.tensor(model.predict(q), dtype=torch.float32, device='cuda')
        qt = torch.tensor(q, dtype=torch.float32, device='cuda')
        z = (base + base.numel() ** 0.5 * (qt @ basis.T + c[:, :normal.shape[1]] @ normal.T)).reshape(-1, 82, 512)
        values.append((z.bfloat16().float().mean(1).double() @ weight + bias).cpu().numpy())
    added = np.concatenate(values).reshape(x.shape)
    value = np.row_stack([added[:n], old['probe'], added[n:]])
    xx, yy = np.meshgrid(xs, np.r_[extra_y[:n], ys, extra_y[n:]])
    assert np.isfinite(value).all() and (value > 0).all()
    assert np.array_equal(value[n:-n], old['probe'])
    out = source.with_name('existing_L_probe_global_padded.npz')
    np.savez_compressed(out, x=xx, y=yy, probe=value)
    metadata = dict(source_sha256=sha(source), output_sha256=sha(out), added_points=len(points), horizontal_expansion_per_side=0.15, vertical_expansion_per_side=0.15, original_values_bitwise_preserved=True, refitted=False, bounds=[float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max())], shape=list(value.shape), value_range=[float(value.min()), float(value.max())], probe_sha256=sha(PROBE), kernel_model_sha256=audit['kernel_model_sha256'], formula='Same kernel lift, bf16 L roundtrip, token-mean L-only affine probe')
    out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)
if __name__ == '__main__':
    main()
