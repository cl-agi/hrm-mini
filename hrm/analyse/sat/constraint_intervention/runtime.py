"""Frozen local HRM inference and observations; this module never launches a run."""
from __future__ import annotations
from hrm.paths import analysis_directory
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Callable, Mapping
import numpy as np
import torch
from torch import Tensor, nn
from hrm.paths import repo_path
from . import study
ROOT = study.ROOT
CHECKPOINT_SHA256 = '09146cffba54e07921b485165c03850ea7f573b4c7be9031aaa58902806160f9'
Observer = Callable[[dict, dict[str, Tensor]], None]

def load_config() -> dict:
    return study.protocol()['model_config']

def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Read a local NPZ without importing the training entrypoint."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}

def verify_checkpoint() -> dict:
    """Verify the selected local raw checkpoint before deserializing it."""
    checkpoint=study.protocol()['checkpoint']
    path=Path(checkpoint['path'])
    digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    if digest!=checkpoint['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
    return {'path':path,'sha256':digest,'weight_key':'model'}

@contextmanager
def _no_bytecode():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous

def _native_hrm_class():
    from hrm.models.hrm_flow_bias import HRM
    return HRM

def _native_encoding():
    from hrm.analyse.sat.rt import encoding
    return encoding

def configure_precision() -> dict:
    """Require full FP32 matrix products rather than TF32 approximations."""
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'float32_matmul_precision': torch.get_float32_matmul_precision(), 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'autocast': 'disabled during forward_outer; caller must also disable for direct module calls'}

def load_model(device: str | torch.device, forward_dtype: str='float32') -> nn.Module:
    """Load unchanged raw checkpoint values into the selected forward precision.

    Configure precision before construction: the native embedding's cast_to
    and initial carry buffers otherwise remain BF16 even after model.float().
    The original BF16 path is available only when explicitly requested.
    """
    if forward_dtype not in ('float32', 'bfloat16'):
        raise ValueError('forward_dtype must be float32 or bfloat16')
    precision = configure_precision()
    metadata = verify_checkpoint()
    config = load_config()
    arch = dict(config['arch'], forward_dtype=forward_dtype)
    native = _native_hrm_class()
    with torch.device(device):
        model = native(arch)
    if forward_dtype == 'float32':
        model.float()
    checkpoint = torch.load(metadata['path'], map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    model.runtime_forward_dtype = forward_dtype
    model.runtime_precision = precision
    return model

def inference_batch(arrays: Mapping[str, np.ndarray], case_id: int, device: str | torch.device, *, seed: int | None=None, n_vars: int | None=None, config: Mapping | None=None) -> dict:
    """One target-free native start; answer values are never used.

    If n_vars is omitted, only the answer array's shape supplies the variable
    count. Noise follows common.batch's independent seed + 810000000 stream.
    CPU and CUDA generators are not expected to produce identical samples.
    """
    if not 0 <= case_id < len(arrays['clauses']):
        raise IndexError('case_id is outside the supplied arrays')
    if n_vars is None:
        if 'answers' not in arrays:
            raise ValueError('n_vars is required when the arrays have no answer shape')
        n_vars = int(arrays['answers'].shape[1])
    seed = 801000 + case_id if seed is None else int(seed)
    config = load_config() if config is None else config
    clauses = torch.as_tensor(arrays['clauses'][case_id:case_id + 1], device=device).long()
    dummy_answers = torch.zeros((1, n_vars), dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = _native_encoding().sat_batch(clauses, dummy_answers, generator, augment=False, delete_fraction=0.0)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 810000000)
    eps = torch.randn((1, n_vars), device=device, generator=noise_generator)
    eps = eps * config['noise_std']
    probability = (eps + torch.zeros_like(eps)).sigmoid()
    weights = torch.stack((1 - probability, probability), -1)
    pairs = torch.stack((weights, weights.flip(-1)), 2).flatten(1, 2)
    mixture = torch.zeros((1, batch['x'].shape[1], 3), device=device)
    mixture[:, :, 2] = 1.0
    mixture[:, 1:2 * n_vars + 1, :2] = pairs
    mixture[:, 1:2 * n_vars + 1, 2] = 0.0
    batch['x'][:, 1:2 * n_vars + 1] = 0
    batch.pop('y')
    batch['mix'] = mixture
    batch['p'] = torch.zeros(1, device=device)
    batch['seed'] = seed
    batch['case_id'] = case_id
    return batch

def bind_graph_mask(model: nn.Module, mask: Tensor) -> None:
    for module in model.modules():
        if module.__class__.__name__ == 'Attention':
            module.graph_mask = mask

def initial_carry(model: nn.Module, batch: Mapping) -> dict[str, Tensor]:
    return dict(model.initial_carry, mixture_weights=batch['mix'])

def pooled_features(carry: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Unweighted token means, float32; includes global, literal and clause nodes."""
    result = {}
    for name in ('H', 'L'):
        state = carry[f'z_{name}']
        if state.ndim != 3:
            raise ValueError('Pooled features require expanded [batch, token, hidden] states')
        result[name] = state.float().mean(dim=1)
    return result

def decode_assignment(logits: Tensor, n_vars: int) -> Tensor:
    return logits[:, 1:2 * n_vars + 1:2].argmax(dim=-1)

def clause_metrics(clauses: Tensor, prediction: Tensor) -> dict[str, Tensor]:
    """Count clauses with no satisfied literal; codes 0/3 satisfy neither sign."""
    if clauses.ndim != 3 or prediction.ndim != 2:
        raise ValueError('Expected clauses [batch, clause, literal] and predictions [batch, variable]')
    if clauses.shape[0] != prediction.shape[0]:
        raise ValueError('Clause and prediction batch sizes differ')
    selected = prediction.gather(1, (clauses.abs().long() - 1).flatten(1)).reshape_as(clauses)
    literal_satisfied = torch.where(clauses > 0, selected == 2, selected == 1)
    satisfied = literal_satisfied.any(dim=-1)
    legal = ((prediction == 1) | (prediction == 2)).all(dim=-1)
    violated = (~satisfied).sum(dim=-1)
    return {'violated': violated, 'legal': legal, 'valid': legal & (violated == 0)}

def _norms(state: Tensor) -> tuple[np.ndarray, np.ndarray]:
    flat = state.detach().double().flatten(1)
    return (torch.linalg.vector_norm(flat, dim=1).cpu().numpy(), flat.square().mean(dim=1).sqrt().cpu().numpy())

@torch.no_grad()
def forward_outer(model: nn.Module, carry: Mapping[str, Tensor], input_ids: Tensor, *, observer: Observer | None=None, outer: int=1) -> tuple[dict[str, Tensor], Tensor]:
    """Call the original native forward once, optionally observing every H/L call.

    callback(event, states) receives norm arrays and detached state copies, never
    recurrent storage. Callback return values are ignored; callbacks should not
    sample RNG or mutate the model. No decoder is applied to L. Compilation and
    GPU numerical parity must be checked separately before an experimental run.
    """
    if 'mixture_weights' not in carry:
        raise ValueError('An explicit fixed mixture is required for target-free replay')
    if observer is None:
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            return model(carry, input_ids)
    states = {key: carry[key] for key in ('z_H', 'z_L')}
    counts = {'H': 0, 'L': 0, 'events': 0}

    def make_hook(name: str):

        def hook(_module, _args, output):
            key = f'z_{name}'
            source = states[key].expand_as(output)
            source_l2, source_rms = _norms(source)
            destination_l2, destination_rms = _norms(output)
            delta_l2, delta_rms = _norms(output.double() - source.double())
            states[key] = output
            counts[name] += 1
            counts['events'] += 1
            if name == 'L':
                h_cycle = (counts['L'] - 1) // model.L_cycles + 1
                l_step = (counts['L'] - 1) % model.L_cycles + 1
            else:
                h_cycle, l_step = (counts['H'], model.L_cycles)
            event = {'outer': outer, 'event_index': counts['events'], 'module': name, 'h_cycle': h_cycle, 'l_step': l_step, 'source_l2': source_l2, 'source_rms': source_rms, 'destination_l2': destination_l2, 'destination_rms': destination_rms, 'delta_l2': delta_l2, 'delta_rms': delta_rms}
            snapshots = {key: value.expand_as(output).detach().clone() for key, value in states.items()}
            observer(event, snapshots)
            return None
        return hook
    handles = [model.L_level.register_forward_hook(make_hook('L')), model.H_level.register_forward_hook(make_hook('H'))]
    try:
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            return model(carry, input_ids)
    finally:
        for handle in handles:
            handle.remove()
