from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn
HIDDEN = 512

def readout_basis(model: nn.Module) -> tuple[Tensor, dict[str, Any]]:
    weight = model.lm_head.weight.detach().to(torch.float32)
    if weight.ndim != 2 or weight.shape[1] != HIDDEN:
        raise ValueError(f'Unexpected lm_head weight shape: {tuple(weight.shape)}')
    rank = int(torch.linalg.matrix_rank(weight).item())
    if rank != weight.shape[0]:
        raise ValueError(f'lm_head rows are not full rank ({rank} vs {weight.shape[0]}); reduced QR would not identify the row space uniquely')
    q, _ = torch.linalg.qr(weight.T, mode='reduced')
    eye = torch.eye(rank, device=q.device, dtype=q.dtype)
    orthogonality_error = float((q.T @ q - eye).abs().max().item())
    head_null_error = float(torch.linalg.matrix_norm(weight - weight @ q @ q.T).item() / torch.linalg.matrix_norm(weight).clamp_min(1e-30).item())
    return (q, {'readout_rank': rank, 'null_rank': HIDDEN - rank, 'q_orthogonality_max_abs': orthogonality_error, 'relative_head_outside_readout_space': head_null_error})
