"""Validate selected views against native terminal PCA, independent of low-rank search."""
from hrm.paths import repo_path, work_path
from pathlib import Path
import json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
base = Path(str(work_path('case6566_global_lift/projection_view')))
archive = Path(str(work_path('L4096_case_search/shortlist')))
data = [np.load(base / f'candidate{i}.npz') for i in range(1, 4)]
result = {f'candidate{i}': dict(native_ring_PC12_area_retention=[], native_ring_variance_retention=[], start_to_H2_H3_H4_tail_min_distance=[]) for i in range(1, 4)}
with torch.inference_mode():
    for h in [2, 3, 4]:
        bits = np.load(archive / f'H{h:03d}_L_bits.npy', mmap_mode='r')[-1024:, 1]
        states = torch.tensor((bits.astype(np.uint32) << 16).view(np.float32).reshape(1024, -1), device='cuda', dtype=torch.float64)
        centered = (states - states.mean(0)) / states.shape[1] ** 0.5
        gram = centered @ centered.T
        eigen, vec = torch.linalg.eigh(gram)
        for i, d in enumerate(data, 1):
            tail = torch.tensor(d[f'H{h}_xy'][-1024:], device='cuda', dtype=torch.float64)
            tail = tail - tail.mean(0)
            projection = vec[:, -2:].T @ tail / eigen[-2:].sqrt()[:, None]
            result[f'candidate{i}']['native_ring_PC12_area_retention'].append(float(torch.linalg.det(projection).abs()))
            result[f'candidate{i}']['native_ring_variance_retention'].append(float(tail.square().sum() / gram.diag().sum()))
            start = d['H1_xy'][0]
            result[f'candidate{i}']['start_to_H2_H3_H4_tail_min_distance'].append(float(np.linalg.norm(d[f'H{h}_xy'][-1024:] - start, axis=1).min()))
result['definition'] = 'Terminal PCA is independently recomputed from original full-space native bf16 states promoted to float64; last1024 steps. PC1-PC2 area factor is abs determinant of projection of the two orthonormal native PCs onto each candidate unit basis.'
(base / 'native_geometry.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
