"""Published mixed-size Maze inputs, without historical selection manifests."""
import argparse
import json
from pathlib import Path
from collections import Counter
from hrm import hub


def load_rows(split='train', path=None):
    """Explicit Parquet inputs take precedence over the pinned HF dataset."""
    if path is None:
        rows = hub.dataset('maze', split)
    else:
        import pyarrow.parquet as pq
        rows = pq.read_table(hub.local_file(path, split+'.parquet')).to_pylist()
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Maze split must be nonempty with unique IDs')
    for row in rows:
        size = len(row['maze'])
        if (len(row['solution']) != size or any(len(line) != size for line in row['maze']+row['solution'])
                or set(''.join(row['maze'])) - set('# SG')
                or set(''.join(row['solution'])) - set('# SGo')):
            raise ValueError(f"Invalid Maze board: {row['id']}")
    counts = Counter(len(row['maze']) for row in rows)
    if split == 'train' and counts != {15:250,20:250,25:250,30:250}:
        raise ValueError(f'E41 requires 250 training mazes per size 15/20/25/30; got {dict(counts)}')
    return rows


def create_train_loader(*, seed, path=None, batch_size=96, rank=0, world_size=8, num_workers=1):
    from torch.utils.data import DataLoader
    from hrm.analyse.maze.extrapolation.maze_dataset import RepeatedMazeDataset, SizeBucketBatchSampler, _loader_kwargs
    rows = load_rows('train', path)
    sizes = [len(row['maze']) for row in rows]
    examples = [{'question': ''.join(row['maze']), 'answer': ''.join(row['solution'])} for row in rows]
    data = RepeatedMazeDataset(examples, sizes, repeat=25)
    sampler = SizeBucketBatchSampler(data, batch_size, rank, world_size, seed)
    loader = DataLoader(data, batch_sampler=sampler, **_loader_kwargs(num_workers))
    return loader, dict(training_examples=len(rows), sizes=dict(Counter(sizes)),
                        batches_per_epoch=len(loader), global_batch_size=768, repeat=25,
                        seed=seed, fixed_order_across_epochs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path)
    parser.add_argument('--split', choices=('train','test','test_control'), default='train')
    args = parser.parse_args()
    rows = load_rows(args.split, args.dataset_path)
    print(json.dumps(dict(split=args.split, examples=len(rows), sizes=dict(Counter(len(r['maze']) for r in rows)),
                         source=str(args.dataset_path) if args.dataset_path else 'cl-agi/maze',
                         revision=None if args.dataset_path else hub.RELEASES['maze'][1])))


if __name__ == '__main__':
    main()
