"""CPU-only configuration and provenance helpers for three independent D seeds."""
from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
from pathlib import Path
import hashlib
import json
from pathlib import Path
ROOT = analysis_directory(__file__)
DEFAULT_DATA_ROOT = Path(str(repo_path('data/maze')))
SELECTION_PATH = ROOT / 'data_selection.json'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_protocol():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    training = protocol['training']
    expected = {'epochs': 20, 'cycles_per_data': 16, 'repeat': 25, 'world_size': 8, 'local_batch_size': 96, 'global_batch_size': 768, 'unique_examples': 1000, 'batches_per_epoch': 32, 'total_optimizer_steps': 10240, 'lr': 6e-05, 'lr_warmup_steps': 2000, 'lr_min_ratio': 1.0, 'beta1': 0.9, 'beta2': 0.95, 'weight_decay': 0.5, 'ema': 0.999, 'log_interval': 5}
    for key, value in expected.items():
        if training[key] != value:
            raise ValueError(f'Approved baseline mismatch: training.{key} must be {value}')
    expected_arch = {'num_layers': 2, 'hidden_size': 512, 'intermediate_size': 2048, 'head_dim': 64, 'norm_eps': 1e-06, 'rope_theta': 10000.0, 'H_cycles': 2, 'L_cycles': 6, 'bptt': True, 'forward_dtype': 'bfloat16', 'is_mlp_mixer': False, 'vocab_size': 6, 'seq_len': 901, 'is_causal': False, 'allowed_sizes': [15, 20, 25, 30, 32, 35, 40]}
    if protocol['architecture'] != expected_arch:
        raise ValueError('Architecture differs from the approved HRM-mini baseline')
    for name, (mode, rope) in {'A': ('original', '1d'), 'B': ('original', '2d'), 'C': ('mixed', '1d'), 'D': ('mixed', '2d')}.items():
        entry = protocol['experiments'][name]
        if (entry['data_mode'], entry['rope_type']) != (mode, rope):
            raise ValueError(f'Experiment {name} does not implement the approved comparison')
    if protocol['seeds'] != [1, 2, 3]:
        raise ValueError('This independent D replication uses seeds 1, 2, 3')
    evaluation = protocol['evaluation']
    if evaluation['every_epoch'] != ['maze_30_original'] or evaluation['final_epoch'] != ['maze_15', 'maze_20', 'maze_25', 'maze_30_original', 'maze_30_control', 'maze_32', 'maze_35', 'maze_40'] or evaluation['checkpoint_epoch_zero_based'] != 19 or (evaluation['weights'] != 'EMA') or (evaluation['examples_per_dataset'] != 1000) or (evaluation['metric'] != 'shortest_path_accuracy') or (evaluation.get('metric_version') != 'maze_shortest_path_v1'):
        raise ValueError('Evaluation must retain the full fixed final-EMA comparison')
    if protocol['runtime'] != {'compile_dynamic': False, 'compile_fullgraph': True, 'dynamo_recompile_limit': 64, 'dynamo_accumulated_recompile_limit': 256, 'num_workers': 1}:
        raise ValueError('Runtime settings differ from the fixed ABCD protocol')
    return protocol

def resolved_config(experiment, seed):
    protocol = load_protocol()
    if experiment != 'D' or seed not in protocol['seeds']:
        raise ValueError('This copy supports D with explicit seed 1, 2, or 3')
    variant = protocol['experiments'][experiment]
    return {'experiment': experiment, 'seed': seed, 'data_mode': variant['data_mode'], 'architecture': dict(protocol['architecture'], rope_type=variant['rope_type']), 'training': protocol['training'], 'runtime': protocol['runtime'], 'evaluation': protocol['evaluation']}

def run_identity(experiment, data_root=DEFAULT_DATA_ROOT, *, seed):
    """Fingerprint inputs and scientific code; never starts a model or queries GPUs."""
    data_root = Path(data_root)
    manifest_path = data_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['datasets']:
        if sha256(data_root / entry['path']) != entry['sha256']:
            raise ValueError(f"Dataset changed: {entry['path']}")
    source_names = ['arch/hrm.py', 'arch/layers.py', 'adam_atan2.py', 'maze_dataset.py', 'experiment_config.py', 'train_abcd.py', 'evaluation.py', 'maze_metrics.py']
    resolved_config(experiment, seed)
    return {'experiment': experiment, 'seed': seed, 'protocol_sha256': sha256(ROOT / 'protocol.json'), 'data_manifest_sha256': sha256(manifest_path), 'data_selection_sha256': sha256(SELECTION_PATH), 'source_sha256': {name: sha256(ROOT / name) for name in source_names}}

def write_json(path, value):
    """Atomic JSON replacement inside the caller's run directory."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)

def validate_evaluation_migration(training_identity, evaluation_identity):
    """Allow only the archived metric-only transition; never relax training resume."""
    receipt = json.loads((ROOT / 'evaluation_migration.json').read_text())
    if receipt.get('metric_version') != 'maze_shortest_path_v1':
        raise ValueError('Unknown evaluation migration')
    for key in ('source_sha256', 'protocol_sha256'):
        if evaluation_identity[key] != receipt['after'][key]:
            raise ValueError(f'Current {key} differs from the verified metric migration')
    prior = dict(evaluation_identity, **receipt['before'])
    if training_identity != prior:
        raise ValueError('Training identity is not the exact archived pre-migration identity')
    return {'metric_version': receipt['metric_version'], 'archive': receipt['archive'], 'training_math_unchanged': receipt['training_math_unchanged']}
