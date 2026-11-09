"""Read-only scientific checks for D's unscaled RoPE cache extension to 80.

No model forward, GPU access, parameter update, training, or dataset mutation.
Only checks.json in this independent evaluation directory is written.
"""
from hrm.paths import analysis_directory
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = analysis_directory(__file__)
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE / 'Maze-lr6e5-wd05-3seed'
RUNTIME = PROJECT / 'd_runtime'
DATA = WORKSPACE / 'revision_packages/comment_2_10/maze_size_generalization/evaluation_d_large_tts/data_large_20261007'
PARENT_MANIFEST = ROOT.parent / 'main_results_retest_20261011/manifest.json'
pass
pass
from hrm.models.hrm import HRM
from hrm.analyse.maze.extrapolation.scoring import METRIC_NAME, METRIC_VERSION, score_predictions

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def encode(rows, field):
    vocab = {char: index for index, char in enumerate('~# SGo')}
    return np.asarray([[0] + [vocab[c] for c in row[field]] for row in rows], dtype=np.int32)

def main():
    torch.set_num_threads(1)
    parent = json.loads(PARENT_MANIFEST.read_text())
    config = json.loads((PROJECT / 'runs/D_2D_seed1/resolved_config.json').read_text())
    original_arch = config['architecture']
    assert original_arch['rope_type'] == '2d' and original_arch['rope_theta'] == 10000.0
    assert original_arch['head_dim'] == 64 and (not original_arch['is_mlp_mixer'])
    assert original_arch['allowed_sizes'] == [15, 20, 25, 30, 32, 35, 40]
    added = [45, 50, 60, 80]
    extended_arch = dict(original_arch, allowed_sizes=sorted(original_arch['allowed_sizes'] + added))
    assert {key for key in original_arch if original_arch[key] != extended_arch[key]} == {'allowed_sizes'}
    with torch.device('cpu'):
        original = HRM(original_arch)
        extended = HRM(extended_arch)
    original_state, extended_state = (original.state_dict(), extended.state_dict())
    assert set(original_state) == set(extended_state)
    assert all((original_state[key].shape == extended_state[key].shape for key in original_state))
    assert all(('cos_' not in key and 'sin_' not in key for key in extended_state))
    parameter_count = sum((parameter.numel() for parameter in extended.parameters()))
    assert parameter_count == 12589056
    for level in ('H_level', 'L_level'):
        old_rope = getattr(original, level).rotary_emb
        new_rope = getattr(extended, level).rotary_emb
        for size in original_arch['allowed_sizes']:
            for prefix in ('cos', 'sin'):
                assert torch.equal(getattr(old_rope, f'{prefix}_{size * size + 1}'), getattr(new_rope, f'{prefix}_{size * size + 1}'))
        axis_dim = original_arch['head_dim'] // 2
        inv_freq = 1.0 / 10000.0 ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        for size in added:
            r, c = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing='ij')
            row_phase = torch.cat((torch.zeros(1), r.flatten()))[:, None] * inv_freq[None, :]
            col_phase = torch.cat((torch.zeros(1), c.flatten()))[:, None] * inv_freq[None, :]
            phases = torch.cat((row_phase, col_phase, row_phase, col_phase), dim=1)
            actual_cos, actual_sin = new_rope(size * size + 1)
            assert torch.equal(actual_cos, phases.cos())
            assert torch.equal(actual_sin, phases.sin())
            assert torch.equal(actual_cos[0], torch.ones(64))
            assert torch.equal(actual_sin[0], torch.zeros(64))
            reference_cos, reference_sin = old_rope(30 * 30 + 1)
            ids = 1 + (torch.arange(30)[:, None] * size + torch.arange(30)[None, :]).flatten()
            assert torch.equal(actual_cos[ids], reference_cos[1:])
            assert torch.equal(actual_sin[ids], reference_sin[1:])
    checkpoint_checks = []
    for job in [job for job in parent['jobs'] if job['variant'] == 'D_2D']:
        path = Path(job['checkpoint'])
        assert sha(path) == job['checkpoint_sha256']
        weights = torch.load(path, map_location='cpu', weights_only=True)
        original.load_state_dict(weights, strict=True)
        extended.load_state_dict(weights, strict=True)
        expected_hash = state_hash(weights)
        assert state_hash(original.state_dict()) == expected_hash
        assert state_hash(extended.state_dict()) == expected_hash
        checkpoint_checks.append({'seed': job['seed'], 'path': str(path), 'checkpoint_sha256': sha(path), 'state_tensor_sha256': expected_hash, 'strict_load_original_and_extended': True, 'all_state_values_unchanged': True})
    assert len(checkpoint_checks) == 3
    data_manifest = json.loads((DATA / 'manifest.json').read_text())
    assert sorted(((entry['size'], entry['count']) for entry in data_manifest['datasets'])) == [(s, 1000) for s in added]
    data_checks = []
    for entry in data_manifest['datasets']:
        path = DATA / entry['path']
        assert sha(path) == entry['sha256']
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1000
        size = entry['size']
        assert all((len(row['question']) == size * size and len(row['answer']) == size * size for row in rows))
        inputs, targets = (encode(rows, 'question'), encode(rows, 'answer'))
        scores = score_predictions(inputs, targets, targets, size=size)
        assert scores['shortest_path_correct'].all()
        assert scores['format_correct'].all() and scores['canonical_correct'].all()
        ratings = np.asarray([int(row['rating']) for row in rows])
        assert np.array_equal(ratings, scores['shortest_distance'] + 1)
        assert np.all(ratings >= entry['min_nodes'])
        wall_masks = set()
        for board in inputs[:, 1:]:
            walls = (board == 1).reshape(size, size)
            orbit = [np.packbits(np.rot90(mirror, turn)).tobytes() for mirror in (walls, np.fliplr(walls)) for turn in range(4)]
            canonical = min(orbit)
            assert canonical not in wall_masks
            wall_masks.add(canonical)
        data_checks.append({'size': size, 'path': str(path), 'sha256': entry['sha256'], 'count': len(rows), 'reference_shortest_correct': int(scores['shortest_path_correct'].sum()), 'd4_wall_layouts_unique': len(wall_masks), 'ratings_match_bfs_node_count': True, 'minimum_required_nodes': entry['min_nodes'], 'observed_min_nodes': int(ratings.min()), 'observed_max_nodes': int(ratings.max())})
        print(f'CPU_DATA_CHECK_PASS size={size} count=1000', flush=True)
    report = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'cpu_only': True, 'model_forward_executed': False, 'training_or_parameter_update': False, 'metric_name': METRIC_NAME, 'metric_version': METRIC_VERSION, 'architecture_before': original_arch, 'architecture_after': extended_arch, 'only_architecture_change': 'allowed_sizes', 'state_dict_keys_shapes_unchanged': True, 'nonpersistent_caches_only': True, 'parameters': parameter_count, 'old_cache_sizes_elementwise_equal': original_arch['allowed_sizes'], 'new_cache_raw_coordinates_theta10000_verified': added, 'same_row_column_phase_across_grid_sizes': True, 'checkpoint_checks': checkpoint_checks, 'data_checks': data_checks, 'data_manifest_sha256': sha(DATA / 'manifest.json'), 'data_provenance': data_manifest['provenance'], 'source_sha256': {str(path): sha(path) for path in (Path(__file__), RUNTIME / 'arch/hrm.py', RUNTIME / 'arch/layers.py', WORKSPACE / 'maze_evaluation/scoring.py')}}
    temporary = ROOT / 'checks.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(ROOT / 'checks.json')
    print('CPU_EXTENSION_CHECKS_PASS', flush=True)
if __name__ == '__main__':
    main()
