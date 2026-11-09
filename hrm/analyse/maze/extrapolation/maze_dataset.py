"""Frozen 1,000-example Maze training sets and unpadded size-aware loaders.

A/B retain the original CSV order and upstream DistributedSampler behavior.
C/D select exactly 250 examples from each of 15/20/25/30, then form global
same-size batches. Both use 25 repetitions, 32 global batches per epoch, and
the same fixed ordering every epoch (the original trainer never set_epoch).
"""
from __future__ import annotations
from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
from pathlib import Path
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(str(repo_path('data/maze')))
DEFAULT_SELECTION_PATH = None
SELECTION_SEED = 20261007
TRAIN_SIZES = (15, 20, 25, 30)
GLOBAL_BATCH_SIZE = 768
VOCAB = '~# SGo'
CHAR_TO_ID = {char: index for index, char in enumerate(VOCAB)}
TRAIN_DATASETS = {size: f'maze-{size}x{size}-hard-1k/train.csv' for size in TRAIN_SIZES}
EVAL_DATASETS = {'maze_15': 'maze-15x15-hard-1k/test.csv', 'maze_20': 'maze-20x20-hard-1k/test.csv', 'maze_25': 'maze-25x25-hard-1k/test.csv', 'maze_30_original': 'maze-30x30-hard-1k/test.csv', 'maze_30_control': 'maze-30x30-reconstructed-control/test.csv', 'maze_32': 'maze-32x32-hard-1k/test.csv', 'maze_35': 'maze-35x35-hard-1k/test.csv', 'maze_40': 'maze-40x40-hard-1k/test.csv'}

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _read_verified_csv(data_root: Path, relative_path: str) -> tuple[list[dict], dict]:
    """Check the immutable bundle manifest before reading any train/test file."""
    bundle = json.loads((data_root / 'manifest.json').read_text())
    entries = [entry for entry in bundle['datasets'] if entry['path'] == relative_path]
    if len(entries) != 1:
        raise ValueError(f'Expected exactly one bundle entry for {relative_path}')
    entry = entries[0]
    path = data_root / relative_path
    if sha256_file(path) != entry['sha256']:
        raise ValueError(f'Dataset SHA256 mismatch: {path}')
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not {'question', 'answer'}.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing question/answer columns: {path}')
        rows = list(reader)
    if len(rows) != 1000 or len(rows) != entry['count']:
        raise ValueError(f'Expected exactly 1,000 rows: {path}')
    area = entry['size'] ** 2
    for index, row in enumerate(rows):
        if len(row['question']) != area or len(row['answer']) != area:
            raise ValueError(f'Incorrect board size: {path}, data row {index}')
        if set(row['question']) - set('# SG') or set(row['answer']) - set('# SGo'):
            raise ValueError(f'Unknown board symbol: {path}, data row {index}')
    return (rows, entry)

def prepare_selection(data_root: Path | str=DEFAULT_DATA_ROOT, path: Path | str=DEFAULT_SELECTION_PATH) -> dict:
    """Create or verify the shared A/B/C/D selection; never silently overwrite.

    A single NumPy default_rng(20261007), using PCG64, is consumed in ascending
    size order. For each size, the first 250 indices of permutation(1000) are
    selected and sorted into source CSV order. No test file participates.
    """
    data_root, path = (Path(data_root), Path(path) if path is not None else analysis_directory(__file__) / 'data_selection.json')
    generator = np.random.default_rng(SELECTION_SEED)
    mixed = []
    for size in TRAIN_SIZES:
        rows, entry = _read_verified_csv(data_root, TRAIN_DATASETS[size])
        indices = sorted(generator.permutation(len(rows))[:250].tolist())
        mixed.append({'size': size, 'path': entry['path'], 'source_sha256': entry['sha256'], 'source_count': len(rows), 'count': len(indices), 'row_indices': indices})
    original = dict(mixed[-1], count=1000, row_indices=list(range(1000)))
    expected = {'schema_version': 1, 'selection_seed': SELECTION_SEED, 'selection_method': 'One numpy.random.default_rng(20261007), PCG64; for sizes 15,20,25,30 in this order: sorted(rng.permutation(1000)[:250]). Only training CSVs are read. A/B use all original 30x30 rows in CSV order.', 'row_index_semantics': 'Zero-based data row index in the source CSV, excluding the header; the corresponding CSV line number is row_index + 2.', 'source_bundle_manifest_sha256': sha256_file(data_root / 'manifest.json'), 'original': {'training_examples': 1000, 'sources': [original]}, 'mixed': {'training_examples': 1000, 'sources': mixed}}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != expected:
            raise ValueError(f'Frozen data selection differs from the expected selection: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('x', encoding='utf-8') as handle:
                json.dump(expected, handle, indent=2)
                handle.write('\n')
        except FileExistsError:
            if json.loads(path.read_text()) != expected:
                raise ValueError(f'Concurrent selection creation mismatch: {path}')
    return expected

class RepeatedMazeDataset(Dataset):
    """Hugging Face Dataset.repeat semantics, without loading its dependencies."""

    def __init__(self, rows: list[dict], sizes: list[int], repeat: int=1):
        if not rows or len(rows) != len(sizes) or repeat < 1:
            raise ValueError('Expected nonempty rows, matching sizes, and positive repeat')
        self.rows = rows
        self.sizes = sizes
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.rows) * self.repeat

    def __getitem__(self, index: int) -> dict:
        return self.rows[index % len(self.rows)]

    def size_at(self, index: int) -> int:
        return self.sizes[index % len(self.rows)]

def collate_fn(batch: list[dict]) -> tuple[Tensor, Tensor]:
    """Identical BOS, int32, vocabulary, and target handling to HRM-mini Maze."""
    xs, ys = ([], [])
    for item in batch:
        xs.append(np.array([CHAR_TO_ID[c] for c in '~' + item['question']], dtype=np.int32))
        ys.append(np.array([CHAR_TO_ID[c] for c in '~' + item['answer']], dtype=np.int32))
    return (torch.from_numpy(np.stack(xs, axis=0)), torch.from_numpy(np.stack(ys, axis=0)))

class SizeBucketBatchSampler(Sampler[list[int]]):
    """One shared global size schedule, strided across distributed ranks.

    A local torch.Generator(seed) first permutes the 6,250 repeated slots for
    each size in ascending order. Eight 768-slot batches per size are kept,
    dropping 106 slots per size. The same generator then permutes all 32
    global batches. Each rank takes batch[rank::world_size]. Reiteration keeps
    this exact schedule; it deliberately does not advance an epoch seed.
    """

    def __init__(self, dataset: RepeatedMazeDataset, batch_size: int, rank: int, world_size: int, seed: int):
        if not 0 <= rank < world_size or batch_size * world_size != GLOBAL_BATCH_SIZE:
            raise ValueError('Need valid rank and global training batch size 768')
        self.rank, self.world_size, self.batch_size = (rank, world_size, batch_size)
        generator = torch.Generator().manual_seed(seed)
        global_batches = []
        for size in sorted(set(dataset.sizes)):
            slots = [index for index in range(len(dataset)) if dataset.size_at(index) == size]
            permutation = torch.randperm(len(slots), generator=generator).tolist()
            shuffled = [slots[index] for index in permutation]
            kept_count = len(shuffled) // GLOBAL_BATCH_SIZE * GLOBAL_BATCH_SIZE
            global_batches.extend((tuple(shuffled[start:start + GLOBAL_BATCH_SIZE]) for start in range(0, kept_count, GLOBAL_BATCH_SIZE)))
        order = torch.randperm(len(global_batches), generator=generator).tolist()
        self.global_batches = tuple((global_batches[index] for index in order))

    def __len__(self) -> int:
        return len(self.global_batches)

    def __iter__(self) -> Iterator[list[int]]:
        for global_batch in self.global_batches:
            yield list(global_batch[self.rank::self.world_size])

def _loader_kwargs(num_workers: int) -> dict:
    if num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    kwargs = dict(collate_fn=collate_fn, pin_memory=True, num_workers=num_workers)
    if num_workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return kwargs

def create_train_loader(mode: str='original', batch_size: int=96, rank: int=0, world_size: int=8, seed: int=1, data_root: Path | str=DEFAULT_DATA_ROOT, selection_path: Path | str=DEFAULT_SELECTION_PATH, repeat: int=25, num_workers: int=1) -> tuple[DataLoader, dict]:
    if mode not in ('original', 'mixed'):
        raise ValueError("mode must be 'original' or 'mixed'")
    if batch_size * world_size != GLOBAL_BATCH_SIZE or not 0 <= rank < world_size:
        raise ValueError('Need valid rank and global training batch size 768')
    if repeat != 25:
        raise ValueError('ABCD protocol fixes repeat=25')
    data_root = Path(data_root)
    selection_path = Path(selection_path) if selection_path is not None else analysis_directory(__file__) / 'data_selection.json'
    selection = prepare_selection(data_root, selection_path)[mode]
    rows, sizes = ([], [])
    for source in selection['sources']:
        source_rows, _ = _read_verified_csv(data_root, source['path'])
        rows.extend((source_rows[index] for index in source['row_indices']))
        sizes.extend([source['size']] * source['count'])
    dataset = RepeatedMazeDataset(rows, sizes, repeat)
    if mode == 'original':
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True, drop_last=True, seed=seed)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, **_loader_kwargs(num_workers))
    else:
        sampler = SizeBucketBatchSampler(dataset, batch_size, rank, world_size, seed)
        loader = DataLoader(dataset, batch_sampler=sampler, **_loader_kwargs(num_workers))
    metadata = {'vocab_size': len(VOCAB), 'seq_len': max(sizes) ** 2 + 1, 'is_causal': False, 'training_examples': len(rows), 'repeat': repeat, 'batches_per_epoch': len(loader), 'sizes': sorted(set(sizes)), 'training_examples_by_size': {str(size): sizes.count(size) for size in sorted(set(sizes))}, 'global_batch_size': GLOBAL_BATCH_SIZE, 'examples_consumed_per_epoch': len(loader) * GLOBAL_BATCH_SIZE, 'selection_sha256': sha256_file(Path(selection_path)), 'fixed_order_across_epochs': True}
    if len(rows) != 1000 or len(loader) != 32:
        raise ValueError('ABCD protocol requires 1,000 unique rows and 32 batches per epoch')
    return (loader, metadata)

def create_eval_loader(name: str, batch_size: int=96, rank: int=0, world_size: int=8, data_root: Path | str=DEFAULT_DATA_ROOT, num_workers: int=1) -> tuple[DataLoader, dict]:
    if name not in EVAL_DATASETS:
        raise ValueError(f'Unknown evaluation dataset: {name}')
    if batch_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Need positive batch size and a valid evaluation rank')
    rows, entry = _read_verified_csv(Path(data_root), EVAL_DATASETS[name])
    dataset = RepeatedMazeDataset(rows, [entry['size']] * len(rows))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=range(rank, len(dataset), world_size), drop_last=False, **_loader_kwargs(num_workers))
    return (loader, {'vocab_size': len(VOCAB), 'seq_len': entry['size'] ** 2 + 1, 'is_causal': False, 'size': entry['size'], 'name': name, 'examples': len(rows), 'batches': len(loader), 'source_path': entry['path'], 'source_sha256': entry['sha256']})
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create/verify the fixed ABCD training selection (CPU only)')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--selection-path', type=Path, default=DEFAULT_SELECTION_PATH)
    arguments = parser.parse_args()
    result = prepare_selection(arguments.data_root, arguments.selection_path)
    print(json.dumps({'selection_path': str((arguments.selection_path or analysis_directory(__file__) / 'data_selection.json').resolve()), 'original_examples': result['original']['training_examples'], 'mixed_examples': result['mixed']['training_examples'], 'mixed_sizes': {str(s['size']): s['count'] for s in result['mixed']['sources']}}, indent=2))
