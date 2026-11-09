"""Experiment discovery and an explicit records scope for analysis execution."""
import contextlib
import json
import os
from pathlib import Path
from hrm.paths import repository_root


def experiments():
    root = repository_root() / 'exp'
    order = json.loads((root / 'index.json').read_text())
    return [json.loads((root / name / 'experiment.json').read_text()) for name in order]


def experiment(identifier):
    for spec in experiments():
        if spec['id'] == identifier: return spec
    legacy = compatibility_experiments()
    if identifier in legacy:
        return legacy[identifier]
    raise ValueError(f'Unknown experiment {identifier!r}; run hrm list')


def compatibility_experiments():
    return json.loads((repository_root() / 'exp/compatibility.json').read_text())


def records_directory(identifier=None, path=None):
    if path is not None:
        root = Path(path).expanduser().resolve()
    else:
        if identifier is None: raise ValueError('Specify --experiment NAME or --records PATH')
        spec = experiment(identifier)
        root = repository_root() / spec['records_root']
        # Existing legacy runs remain usable without moving their records.
        if identifier in compatibility_experiments():
            previous = repository_root() / 'exp' / identifier / 'records'
            if previous.is_dir() and any(p.is_file() for p in previous.rglob('*')):
                root = previous
    if not root.is_dir() or not any(p.is_file() for p in root.rglob('*')):
        raise FileNotFoundError(f'Analysis records are missing or empty: {root}. Supply --records PATH or generate this experiment\'s records first.')
    return root


@contextlib.contextmanager
def analysis_scope(module, identifier=None, records=None):
    if identifier is None and records is None:
        matches = [e['id'] for e in experiments() if module in e.get('modules', [])]
        if len(matches)!=1:
            raise ValueError(f'Analysis belongs to {matches or "no indexed experiment"}; specify --experiment or --records')
        identifier = matches[0]
    if identifier is not None:
        spec=experiment(identifier)
        if module not in spec.get('modules',[]):
            raise ValueError(f'{module} is not indexed under {identifier}')
    if records is None and identifier not in compatibility_experiments() and module != 'hrm.analyse.training_summary':
        scopes = {item['records_root'] for item in compatibility_experiments().values()
                  if item['group'] == identifier and module in item.get('modules', [])}
        if len(scopes) == 1:
            records = repository_root() / scopes.pop()
        elif len(scopes) > 1:
            raise ValueError(f'{module} has multiple record scopes; specify --records PATH')
    root=records_directory(identifier,records)
    old=os.environ.get('HRM_ANALYSIS_RECORDS');cwd=Path.cwd()
    os.environ['HRM_ANALYSIS_RECORDS']=str(root)
    try:
        os.chdir(root)
        yield root
    finally:
        os.chdir(cwd)
        if old is None:os.environ.pop('HRM_ANALYSIS_RECORDS',None)
        else:os.environ['HRM_ANALYSIS_RECORDS']=old


def recipes(identifier=None):
    entries=json.loads((repository_root()/'exp/recipes.json').read_text())['recipes']
    if identifier is None:return entries
    keys=experiment(identifier)['recipes']
    return {key:entries[key] for key in keys}


def recipe_config(recipe, fresh_seed=False):
    """Unknown historical seeds may be replaced explicitly, never invented."""
    import copy
    if recipe.get('status')!='configuration_validated':
        if not (fresh_seed and recipe.get('unresolved_fields')==['seeds']):
            raise ValueError('Recipe requires resolution: '+'; '.join(recipe.get('issues',[]))+'; use --fresh-seed only when seeds are the sole unresolved field')
    config=copy.deepcopy(recipe['config'])
    if fresh_seed:config.pop('seeds',None)
    return config
