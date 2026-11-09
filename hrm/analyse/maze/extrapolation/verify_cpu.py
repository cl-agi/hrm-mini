"""Read source/data and check eight-rank D accounting without GPU or training."""
from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
from pathlib import Path
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from hrm.analyse.maze.extrapolation.Maze_lr6e5_wd05_3seed_d_runtime_experiment_config import DEFAULT_DATA_ROOT, load_protocol, resolved_config, run_identity
from hrm.analyse.maze.extrapolation.maze_dataset import create_train_loader, create_eval_loader, EVAL_DATASETS, prepare_selection
from hrm.analyse.maze.extrapolation.Maze_lr6e5_wd05_3seed_d_runtime_train_abcd import check_resume
ROOT = analysis_directory(__file__)
SOURCE = Path(str(work_path('maze_size_generalization/training_abcd')))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def function_ast(path, name):
    tree = ast.parse(Path(path).read_text())
    return ast.dump(next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)))

def main():
    unchanged = ['arch/hrm.py', 'arch/layers.py', 'adam_atan2.py', 'data_selection.json']
    for name in unchanged:
        assert digest(ROOT / name) == digest(SOURCE / name), name
    for name in ('train_step', 'update_lr', 'check_resume', 'restore_metric_log'):
        assert function_ast(Path(__file__).resolve().parent / 'train_abcd.py', name) == function_ast(SOURCE / 'train_abcd.py', name), name
    for name in ('evaluate', 'configure_compiler'):
        assert function_ast(Path(__file__).resolve().parent / 'evaluation.py', name) == function_ast(SOURCE / 'evaluation.py', name), name
    for name in ('SizeBucketBatchSampler', 'create_train_loader', 'create_eval_loader', 'collate_fn'):
        assert function_ast(Path(__file__).resolve().parent / 'maze_dataset.py', name) == function_ast(SOURCE / 'maze_dataset.py', name), name
    protocol = load_protocol()
    selection = prepare_selection()
    assert [x['size'] for x in selection['mixed']['sources']] == [15, 20, 25, 30]
    assert all((x['count'] == 250 for x in selection['mixed']['sources']))
    seed_results, identities, schedule_digests = ({}, [], [])
    for seed in (1, 2, 3):
        config = resolved_config('D', seed)
        assert config['data_mode'] == 'mixed' and config['architecture']['rope_type'] == '2d'
        loaders = [create_train_loader('mixed', rank=rank, world_size=8, seed=seed, num_workers=0)[0] for rank in range(8)]
        schedules = [list(loader.batch_sampler) for loader in loaders]
        assert all((len(loader) == 32 for loader in loaders))
        dataset = loaders[0].dataset
        assert len(dataset.rows) == 1000
        assert Counter(dataset.sizes) == {15: 250, 20: 250, 25: 250, 30: 250}
        all_slots, sizes_count = ([], Counter())
        rank_slots = [set() for _ in range(8)]
        for batch_index in range(32):
            global_slots, batch_sizes = ([], set())
            for rank in range(8):
                slots = schedules[rank][batch_index]
                assert len(slots) == 96
                global_slots.extend(slots)
                rank_slots[rank].update(slots)
                batch_sizes.update((dataset.size_at(index) for index in slots))
            assert len(batch_sizes) == 1
            assert len(global_slots) == len(set(global_slots)) == 768
            sizes_count[next(iter(batch_sizes))] += 1
            all_slots.extend(global_slots)
        assert sizes_count == {15: 8, 20: 8, 25: 8, 30: 8}
        assert len(all_slots) == len(set(all_slots)) == 24576
        assert {index % 1000 for index in all_slots} == set(range(1000))
        for left in range(8):
            assert list(loaders[left].batch_sampler) == schedules[left]
            for right in range(left + 1, 8):
                assert rank_slots[left].isdisjoint(rank_slots[right])
        identity = run_identity('D', DEFAULT_DATA_ROOT, seed=seed)
        identities.append(identity)
        checkpoint = {'format_version': 1, 'identity': identity, 'weights': 'non_ema_training', 'world_size': 8, 'total_steps': 10240, 'epoch': 0, 'step': 512, 'rng_states': [{} for _ in range(8)]}
        check_resume(checkpoint, identity, config['training'])
        mismatched = dict(identity, seed=seed % 3 + 1)
        try:
            check_resume(checkpoint, mismatched, config['training'])
        except ValueError:
            pass
        else:
            raise AssertionError('Resume failed to reject a different seed')
        fingerprint = hashlib.sha256(json.dumps(schedules).encode()).hexdigest()
        schedule_digests.append(fingerprint)
        seed_results[str(seed)] = {'batches_per_epoch': 32, 'optimizer_steps_per_epoch': 512, 'optimizer_steps_total': 10240, 'consumed_slots_per_epoch': len(all_slots), 'unique_questions': 1000, 'examples_by_size': dict(Counter(dataset.sizes)), 'schedule_sha256': fingerprint, 'resume_seed_checked': True}
    assert len(set(schedule_digests)) == 3
    assert [identity['seed'] for identity in identities] == [1, 2, 3]
    evaluations = {}
    for name in EVAL_DATASETS:
        indices, batch_lengths = ([], [])
        for rank in range(8):
            loader, metadata = create_eval_loader(name, batch_size=96, rank=rank, world_size=8, num_workers=0)
            indices.extend(loader.sampler)
            lengths = [len(batch) for batch in loader.batch_sampler]
            assert lengths == [96, 29]
            batch_lengths.append(lengths)
        assert sorted(indices) == list(range(1000))
        evaluations[name] = {'examples': 1000, 'per_rank_batches': batch_lengths}
    assert len(evaluations) == 8
    receipt = {'passed': True, 'world_size': 8, 'gpu_used': False, 'training_started': False, 'original_D_core_unchanged': unchanged, 'selection_sha256': digest(ROOT / 'data_selection.json'), 'training': protocol['training'], 'seeds': seed_results, 'evaluation_coverage': evaluations, 'test_policy': protocol['evaluation'], 'gpu_memory_fit_verified': False}
    (ROOT / 'checks_cpu.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
if __name__ == '__main__':
    main()
