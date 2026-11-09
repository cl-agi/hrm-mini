"""Freeze sampling/training inputs before constructing datasets or models."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def prepare(config):
    keys = ('model', 'task', 'learning_rate', 'warmup', 'train_steps', 'ckpt_count', 'eager')
    if config.restore:
        root = Path(config.restore)
        path = root / 'resolved_config.json'
        if not path.is_file():
            raise FileNotFoundError(f'Restore requires saved configuration: {path}')
        saved = json.loads(path.read_text())
        if config.seed is not None and config.seed != saved['seed']:
            raise ValueError('Restore seed differs from the saved seed')
        if config.data_dir is not None and str(Path(config.data_dir).resolve()) != saved['data_dir']:
            raise ValueError('Restore data directory differs from the saved directory')
        if config.force_data_gen:
            raise ValueError('Do not regenerate data in a restored run; create a new run')
        for key in keys:
            setattr(config, key, saved[key])
    else:
        seed = int(time.time()) if config.seed is None else config.seed
        if seed < 0 or seed > 2**31 - 1:
            raise ValueError('seed must be between 0 and 2**31-1')
        label = config.label
        if label and (Path(label).name != label or label in ('.', '..')):
            raise ValueError('label must be a single directory-name component')
        root = Path(config.train_dir) / f'{config.task}_{seed}{"_" + label if label else ""}'
        if config.data_dir is not None and not Path(config.data_dir).is_dir():
            raise FileNotFoundError(f"Explicit data directory does not exist: {config.data_dir}")
        root.mkdir(parents=True, exist_ok=False)
        data = Path(config.data_dir).resolve() if config.data_dir else (root / 'data').resolve()
        saved = {key: getattr(config, key) for key in keys}
        saved.update(seed=seed, data_dir=str(data),
                     data_source='explicit_directory' if config.data_dir else 'run_local_synthetic_generator',
                     generator=config.task, force_data_gen=config.force_data_gen,
                     resume_limit='Model/optimizer restore; random stream and dataset cursor are not restored.')
        package = Path(__file__).parent
        saved['source_sha256'] = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(package.rglob('*.py'))}
        saved['versions'] = {name: importlib.metadata.version(name)
                             for name in ('hrm-querysat', 'tensorflow', 'numpy', 'CNFgen', 'python-sat')}
        with (root / 'resolved_config.json').open('x') as stream:
            json.dump(saved, stream, indent=2)
            stream.write('\n')
    config.seed = saved['seed']
    config.data_dir = saved['data_dir']
    config.train_dir = str(root)
    return saved
