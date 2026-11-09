"""Prepare and compute local Jacobian and Taylor records from an explicit or published HRM."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from hrm import hub
from hrm.paths import checkpoint_config, repository_root
from hrm.training.data import puzzle_directory
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader, dataset_sha256
from hrm.analyse.dynamics.jacobian.models import HRM, unwrap_state_dict, analysis_architecture, load_config
from . import h_l_joint_transition_jacobian as local
from . import h_l_natural_taylor_prediction as taylor


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(kind, argv=None, *, mixer=False):
    p = argparse.ArgumentParser(description=f'Generate {kind} Jacobian records (six L updates per H update).')
    p.add_argument('--checkpoint', type=Path, required=mixer)
    p.add_argument('--config', type=Path)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--boards', type=int, default=64)
    p.add_argument('--selection', choices=['random', 'all', 'sequential'], default='random')
    p.add_argument('--seed', type=int)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--probes', type=int, default=16 if kind == 'triangle' else 1)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--rollout-sources', type=int, nargs='+', default=[0, 4, 8, 12, 16, 24])
    p.add_argument('--max-horizon', type=int, default=8)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(argv)
    if min(a.boards, a.batch_size, a.probes, a.max_horizon) <= 0:
        p.error('Counts must be positive')
    if a.max_horizon < 2 or any(s < 0 or s >= 31 for s in a.rollout_sources):
        p.error('Taylor horizon must be >=2 and source in [0, 31)')
    if a.config is not None and a.checkpoint is None:
        p.error('--config requires --checkpoint')
    records = ('exp/dynamics/records/mixer' if mixer else
               'exp/memory/records/local_jacobian' if kind == 'local' else
               'exp/memory/records/taylor')
    output = (a.output_dir or Path(os.environ.get('HRM_ANALYSIS_RECORDS', repository_root()/records))).resolve()
    if mixer and a.output_dir is None:
        output = output/kind
    plan_path = output/'run_config.json'
    old = json.loads(plan_path.read_text()) if plan_path.exists() else None
    seed = a.seed if a.seed is not None else (old['seed'] if old else int(time.time()))
    checkpoint = hub.checkpoint('sudoku', a.checkpoint)
    config_path = (a.config or checkpoint_config(checkpoint)) if a.checkpoint else None
    config = load_config(config_path) if config_path else {'arch': hub.model_config('sudoku')}
    arch = analysis_architecture(config)
    if mixer and not arch.get('is_mlp_mixer', False):
        p.error('Mixer audit requires an is_mlp_mixer checkpoint configuration')
    if arch.pop('name', '') not in ('hrm@HRM', 'hrm.models.hrm@HRM'):
        p.error('Local Jacobian and Taylor analyses require a standard HRM checkpoint')
    arch.setdefault('vocab_size', 10)
    arch.setdefault('seq_len', 82)
    if arch['vocab_size'] != 10 or arch['seq_len'] != 82:
        p.error('Expected Sudoku vocabulary 10 and sequence length 82')
    if arch.get('H_cycles') != 2 or arch.get('L_cycles') != 6 or arch.get('rope_steps') is not None:
        p.error('This analysis requires the 2H6L schedule and stationary positional encoding')
    records = read_test_records(a.dataset if a.dataset is not None else puzzle_directory('sudoku'))
    ids = select_board_ids(len(records), boards=a.boards, offset=0, selection=a.selection, selection_seed=seed, explicit=None)
    plan = dict(kind=kind, mixer=mixer, seed=seed, selection=a.selection, board_ids=list(ids), population=len(records),
                dataset_sha256=dataset_sha256(records), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                config=config, batch_size=a.batch_size, probes=a.probes, device=a.device,
                rollout_sources=a.rollout_sources, max_horizon=a.max_horizon,
                code_sha256={f:sha(Path(__file__).parent/f) for f in ['local_analysis.py','h_l_joint_transition_jacobian.py','h_l_natural_taylor_prediction.py','../jacobian/models.py','../jacobian/data.py','../jacobian/compute.py']},
                public_model_revision=hub.RELEASES['sudoku'][0] if a.checkpoint is None else None,
                public_dataset_revision=hub.RELEASES['sudoku'][1] if a.dataset is None else None)
    if old is not None and old != plan:
        raise ValueError('Inputs/settings changed; use a new output directory')
    if old is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Output must be empty: {output}')
        local.atomic_json(plan_path, plan)
    if a.prepare_only:
        return 0
    # Exclusive marker prevents overwriting completed or interrupted evidence.
    with (output/'started.json').open('x') as f:
        json.dump({'started_unix': time.time()}, f)
    device = torch.device(a.device)
    model = HRM(arch).to(device)
    model.load_state_dict(unwrap_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)), strict=True)
    model.eval().requires_grad_(False).float()
    model.embed.cast_to = torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    if kind == 'triangle':
        from hrm.analyse.dynamics.jacobian.compute import compute_jtm
        manifest = compute_jtm(model, build_loader(records, ids, a.batch_size), device=device,
            output_dir=output, outer_steps=16, probes=a.probes, random_seed=seed,
            save_per_probe=False, manifest_metadata=plan)
        checks = manifest['checks']
        if not checks['all_values_finite'] or not checks['probe_row_count_matches'] or checks['boards_processed'] != len(ids):
            manifest['status'] = 'failed'
            local.atomic_json(output/'manifest.json', manifest)
            raise RuntimeError('Output-Jacobian integrity checks failed')
        return 0
    generator = torch.Generator(device=device).manual_seed(seed)
    rows, rollouts = [], []
    parity = 0.
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for x, _, batch_ids in build_loader(records, ids, a.batch_size):
            if kind == 'local':
                batch, error = local.run_batch(model=model, x=x.to(device), board_ids=batch_ids.tolist(), probes=a.probes, generator=generator)
            else:
                batch, rollout, error = taylor.run_batch(model, x.to(device), batch_ids.tolist(), a.rollout_sources, a.max_horizon)
                rollouts.extend(rollout)
            rows.extend(batch)
            parity = max(parity, error)
    if kind == 'local':
        summary, matrix = local.summarize(rows)
        local.atomic_csv(output/'per_board_probe.csv', rows)
        local.atomic_csv(output/'summary_long.csv', summary)
        local.atomic_csv(output/'stage_matrix.csv', matrix)
    else:
        taylor.atomic_csv(output/'local_secant.csv', rows)
        taylor.atomic_csv(output/'frozen_rollout.csv', rollouts)
    local.atomic_json(output/'manifest.json', dict(plan, status='complete', rows=len(rows), rollout_rows=len(rollouts), maximum_clean_forward_parity_error=parity))
    return 0
