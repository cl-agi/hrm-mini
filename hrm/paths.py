"""Locate supplied assets independently of the current working directory."""
import os
from pathlib import Path

def repository_root():
    override = os.environ.get('HRM_REPOSITORY')
    if override:
        return Path(override).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'hrm').is_dir():
            return parent
    raise RuntimeError('Set HRM_REPOSITORY to the HRM repository root')

def repo_path(relative):
    path = Path(relative)
    records = os.environ.get("HRM_ANALYSIS_RECORDS")
    if records and not path.is_absolute() and path.parts and path.parts[0] in ("data", "checkpoint", "experiment"):
        return Path(records) / path
    return repository_root() / relative

def work_path(relative):
    """Intermediate outputs are regenerated outside the distribution by default."""
    root = Path(os.environ.get("HRM_ANALYSIS_RECORDS") or os.environ.get('HRM_WORKDIR', str(repository_root().parent / 'hrm-work')))
    return root / relative

def checkpoint_config(checkpoint):
    """Prefer a checkpoint's saved config; archived bindings are optional."""
    import json
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    for name in ('model_config.json','resolved_config.json'):
        adjacent=checkpoint.parent/name
        if adjacent.is_file():return adjacent
    manifest=repo_path('data/checkpoint_configs.json')
    if manifest.is_file():
        mapping=json.loads(manifest.read_text())
        try:relative=str(checkpoint.relative_to(repository_root()))
        except ValueError:relative=None
        if relative in mapping:
            target=repo_path(mapping[relative])
            if target.is_file():return target
    raise FileNotFoundError(f'No configuration next to {checkpoint}; supply --config or an adjacent model_config.json')


def dataset_path(name):
    """Use explicit local inputs first, then optional archived bindings or HF."""
    import json
    name=str(name)
    path=Path(name).expanduser()
    if path.exists():return path.resolve()
    if name.startswith('data/'):
        path=repo_path(name)
        if path.exists():return path.resolve()
    bindings=repo_path('data/validation/dataset_binding.json')
    if bindings.is_file():
        mapping={x['source']:x['target'] for x in json.loads(bindings.read_text()) if x['status']=='verified'}
        if name in mapping:
            target=repo_path(mapping[name])
            if not target.exists():raise FileNotFoundError(target)
            return target.resolve()
    if name=='sapientinc/sudoku-extreme-1k' or name=='./downloaded-datasets/sudoku-extreme-1k':
        from hrm.training.data import puzzle_directory
        return puzzle_directory('sudoku')
    if name in ('./downloaded-datasets/sudoku-extreme','sapientinc/sudoku-extreme'):
        from hrm.training.data import public_sudoku_directory
        return public_sudoku_directory()
    raise FileNotFoundError(f'Dataset not found: {name!r}; specify an existing dataset path')


def analysis_directory(source_file):
    """Location of a module's records, preserving its relative directory layout."""
    from hrm.experiments import records_directory, experiments
    source = Path(source_file).resolve()
    relative = source.parent.relative_to(repository_root())
    root = os.environ.get('HRM_ANALYSIS_RECORDS')
    if root is None:
        module = '.'.join(source.relative_to(repository_root()).with_suffix('').parts)
        choices = [e['id'] for e in experiments() if module in e.get('modules', [])]
        if len(choices) != 1:
            raise ValueError('Run this analysis via hrm analyse --experiment NAME --records PATH')
        root = records_directory(choices[0])
    return Path(root) / relative
