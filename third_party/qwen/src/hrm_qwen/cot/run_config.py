"""Freeze teacher-CoT inputs before any distributed training or model download."""
import hashlib
import json
import os
from pathlib import Path
import time


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze_run(args):
    root = Path(args.output_dir).resolve()
    path = root / 'resolved_config.json'
    saved = json.loads(path.read_text()) if path.exists() else None
    if args.seed is None:
        args.seed = saved['seed'] if saved else int(time.time())
    config = {k: v for k, v in vars(args).items() if k != 'prepare_only'}
    config['output_dir'] = str(root)
    config['traces'] = [str(Path(p).resolve(strict=True)) for p in args.traces]
    config['trace_sha256'] = [file_hash(p) for p in config['traces']]
    config['code_sha256'] = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    if args.epochs < 1 or args.grad_accum < 1 or args.max_len < 1:
        raise ValueError('epochs, grad-accum and max-len must be positive')
    if saved is not None and config != saved:
        raise ValueError('Configuration, source or traces changed; use a new output directory')
    if saved is None:
        if not args.prepare_only:
            raise ValueError('Run the same command with --prepare-only before training')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('Prepare once before torchrun; ranks must share the saved config')
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f'Output directory is not empty: {root}')
        root.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(config, stream, indent=2)
    if not args.prepare_only and int(os.environ.get('RANK', '0')) == 0:
        # This upstream trainer does not implement optimizer-state resume.
        with (root / 'training_started.json').open('x') as stream:
            json.dump({'seed': args.seed, 'world_size': int(os.environ.get('WORLD_SIZE', '1'))}, stream)
    return config
