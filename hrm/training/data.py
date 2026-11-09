"""Resolve explicit inputs, published HF splits, or recorded resampling policies."""
import copy
import csv
import json
from pathlib import Path
from hrm import hub


def local_directory(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {path}')
    return path


def puzzle_directory(task='sudoku', variant='standard', size=30):
    import numpy as np
    if variant == 'unpublished':
        raise FileNotFoundError('This exact training dataset is not published on HF. Supply --dataset-path; refusing to substitute a different cohort.')
    root = hub.cache_path(task, f'training_{variant}_{size}/placeholder').parent
    if variant == 'easy':
        groups = json.loads(hub.data_file('sudoku', 'evaluation_groups.json').read_text())
        wanted = {'train': 'train_easy', 'validation_easy': 'validation_easy', **{k: k for k in groups}}
        by_id = None
        for name, split in wanted.items():
            dest = root / (name + '.npz')
            if dest.exists(): continue
            if name in groups:
                if by_id is None:
                    by_id = {r['id']: r for s in ('test', 'test_transfer') for r in hub.dataset('sudoku', s)}
                rows = [by_id[i] for i in groups[name]]
            else:
                rows = hub.dataset('sudoku', split)
            arrays = {key: np.asarray([[int(c) for c in r[field]] for r in rows], dtype=np.uint8)
                      for key, field in [('puzzles','puzzle'),('solutions','solution')]}
            def write(tmp):
                with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
            hub._write(dest, write)
        return root
    if variant != 'standard': raise ValueError(f'Unknown dataset variant: {variant}')
    for split, source in [('train','train'),('test','test'),('test_hard','test')]:
        dest = root / (split + '.csv')
        if not dest.exists():
            original = hub.puzzle_data(task, split=source, size=size)
            hub._write(dest, lambda tmp: tmp.write_bytes(original.read_bytes()))
    return root


def resolve_config(config, defaults=None, dataset_path=None, eval_dataset_path=None):
    config = copy.deepcopy(config)
    data = config.setdefault('data', {})
    defaults = defaults or {}
    explicit = dataset_path if dataset_path is not None else data.get('dataset_name')
    if explicit is not None:
        if str(explicit) in ('sapientinc/sudoku-extreme','sapientinc/sudoku-extreme-1k') or str(explicit).startswith('hf://'):
            data['dataset_name'] = str(explicit)
        else:
            data['dataset_name'] = str(local_directory(explicit))
    elif data.get('prepared_data_path'):
        data['prepared_data_path'] = str(local_directory(data['prepared_data_path']))
    else:
        module = data.get('name', '')
        task = defaults.get('task') or ('maze' if 'maze' in module else 'sudoku' if 'sudoku' in module else None)
        if task is None: raise ValueError('No HF training dataset registered for this loader; specify --dataset-path')
        variant = defaults.get('variant', 'easy' if 'blank_difficulty' in module else 'standard')
        if variant == 'easy_resampled':
            data['dataset_name'] = str(resampled_easy_directory(defaults['sampling_seed'],defaults.get('train_size',80000)))
        elif variant == 'upstream_mixer':
            data['dataset_name'] = SUDOKU_UPSTREAM
        elif variant in ('upstream_full','upstream_stratified'):
            data['dataset_name'] = str(public_sudoku_directory(defaults.get('train_size'),defaults.get('ordered_sha256')))
        else:
            data['dataset_name'] = str(puzzle_directory(task, variant, defaults.get('size', 30)))
    if dataset_path is not None and defaults.get('historical_prepared_source'):
        data['prepared_data_path'] = str(local_directory(dataset_path))
        data['dataset_name'] = defaults['historical_source']
    if eval_dataset_path is not None:
        config['eval_dataset_name'] = str(local_directory(eval_dataset_path))
    elif config.get('eval_dataset_name') is not None:
        config['eval_dataset_name'] = str(local_directory(config['eval_dataset_name']))
    elif defaults.get('historical_eval_source'):
        config['eval_dataset_name'] = str(puzzle_directory('sudoku'))
    return config


def sat_directory(path=None, n_vars=1024):
    if path is not None:
        root = local_directory(path)
        for split in ('train','valid'): hub.local_file(root/'data'/f'{split}_n{n_vars}.npz')
        return root
    import numpy as np
    root = hub.cache_path('sat', f'training_n{n_vars}/data/placeholder').parent.parent
    for target, split in [('train','train'),('valid','validation'),('test','test')]:
        dest = root/'data'/f'{target}_n{n_vars}.npz'
        if dest.exists(): continue
        rows = [r for r in hub.dataset('sat', split, 'cnf') if r['num_variables']==n_vars]
        if not rows: raise FileNotFoundError(f'HF SAT {split} has no n={n_vars} training cohort. Specify --data-root.')
        if any(r['solution'] is None for r in rows): raise ValueError(f'Missing SAT training labels: {split}')
        arrays = dict(clauses=np.asarray([r['clauses'] for r in rows],np.int32),answers=np.asarray([r['solution'] for r in rows],np.int8))
        def write(tmp):
            with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
        hub._write(dest,write)
    return root


SUDOKU_UPSTREAM = 'sapientinc/sudoku-extreme'
SUDOKU_UPSTREAM_REVISION = '58942f96baeb572ca3127e2a9e9c70f330783d6b'


def public_sudoku_directory(size=None, expected_sha256=None):
    """Original public corpus, optionally using the archived source-stratified sampler."""
    from huggingface_hub import hf_hub_download
    import hashlib
    import numpy as np
    import pyarrow as pa
    import pyarrow.csv as pacsv
    root = hub.cache_path('sudoku', f'upstream_{SUDOKU_UPSTREAM_REVISION}_{size or "full"}/placeholder').parent
    train = root/'train.csv'
    if not train.exists():
        source = Path(hf_hub_download(SUDOKU_UPSTREAM, 'train.csv', repo_type='dataset', revision=SUDOKU_UPSTREAM_REVISION))
        if size is None:
            def link(tmp):
                tmp.unlink();tmp.symlink_to(source.resolve())
            hub._write(train,link)
        else:
            table = pacsv.read_csv(source,convert_options=pacsv.ConvertOptions(column_types={'question':pa.string(),'answer':pa.string(),'source':pa.string()}))
            if len(table)!=3831994:raise ValueError('Unexpected public Sudoku training population')
            sources=table['source'].to_numpy();names=sorted(set(sources))
            pool={s:np.flatnonzero(sources==s) for s in names};rng=np.random.default_rng(42)
            for ids in pool.values():rng.shuffle(ids)
            quotas={s:size*len(pool[s])//len(table) for s in names}
            order=sorted(names,key=lambda s:(-(size*len(pool[s])%len(table)),s))
            for s in order[:size-sum(quotas.values())]:quotas[s]+=1
            indices=np.concatenate([pool[s][:quotas[s]] for s in names]);np.random.default_rng(42+size).shuffle(indices)
            selected=table.select(['question','answer']).take(pa.array(indices))
            hub._write(train,lambda tmp:pacsv.write_csv(selected,tmp))
    if expected_sha256:
        digest=hashlib.sha256()
        with train.open() as f:
            for row in csv.DictReader(f):digest.update((row['question'].replace('.','0')+'|'+row['answer']+'\n').encode())
        metadata={'source':SUDOKU_UPSTREAM,'revision':SUDOKU_UPSTREAM_REVISION,'ordered_sha256':digest.hexdigest(),'historical_ordered_sha256':expected_sha256,'historical_sample_equal':digest.hexdigest()==expected_sha256,'source_shuffle_seed':42,'subset_shuffle_seed':42+size if size else None,'train_size':size}
        hub._write(root/'sampling.json',lambda tmp:tmp.write_text(json.dumps(metadata,indent=2)+'\n'))
    for alias in ['test_hard','test']:
        dest=root/(alias+'.csv')
        if not dest.exists():
            source=hub.puzzle_data('sudoku',split='test')
            hub._write(dest,lambda tmp:tmp.write_bytes(source.read_bytes()))
    return root


def _freeze_seed(output, filename, requested=None, metadata=None):
    """Rank zero writes once; every worker reads the same saved seed."""
    import os,time
    path=Path(output)/filename
    path.parent.mkdir(parents=True,exist_ok=True)
    if int(os.environ.get('RANK','0'))==0 and not path.exists():
        seed=int(time.time()) if requested is None else int(requested)
        payload=dict(metadata or {},seed=seed)
        hub._write(path,lambda tmp:tmp.write_text(json.dumps(payload,indent=2)+'\n'))
    deadline=time.monotonic()+60
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(f'Waiting for rank-zero seed: {path}')
        time.sleep(.1)
    seed=int(json.loads(path.read_text())['seed'])
    if requested is not None and seed!=requested:raise ValueError('Saved data seed differs; use a new output directory for a new sample')
    return seed


def freeze_sampling_seed(output, requested=None):
    return _freeze_seed(output,'dataset_sampling.json',requested,{
        'source':SUDOKU_UPSTREAM,'revision':SUDOKU_UPSTREAM_REVISION,
        'policy':'Resample 80000 training puzzles with <=50 blanks, excluding published validation inputs',
        'historical_sample_reused':False})


def freeze_training_seed(output):
    return _freeze_seed(output,'training_seed.json')


def resampled_easy_directory(seed, count=80000):
    from huggingface_hub import hf_hub_download
    import numpy as np
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.compute as pc
    base=puzzle_directory('sudoku','easy')
    root=hub.cache_path('sudoku',f'easy_resampled_{SUDOKU_UPSTREAM_REVISION}_{seed}_{count}/placeholder').parent
    for source in base.glob('*.npz'):
        if source.name=='train.npz':continue
        dest=root/source.name
        if not dest.exists():
            def link(tmp):tmp.unlink();tmp.symlink_to(source.resolve())
            hub._write(dest,link)
    dest=root/'train.npz'
    if not dest.exists():
        source=hf_hub_download(SUDOKU_UPSTREAM,'train.csv',repo_type='dataset',revision=SUDOKU_UPSTREAM_REVISION)
        table=pacsv.read_csv(source,convert_options=pacsv.ConvertOptions(column_types={'question':pa.string(),'answer':pa.string(),'source':pa.string()}))
        q=pc.replace_substring(table['question'],pattern='.',replacement='0')
        validation=hub.dataset('sudoku','validation_easy')
        eligible=pc.and_(pc.less_equal(pc.count_substring(q,pattern='0'),50),pc.invert(pc.is_in(q,value_set=pa.array([r['puzzle'] for r in validation]))))
        indices=np.flatnonzero(eligible.to_numpy())
        if len(indices)<count:raise ValueError('Not enough disjoint easy puzzles in the public source')
        selected=np.random.default_rng(seed).choice(indices,count,replace=False)
        rows=table.select(['question','answer']).take(pa.array(selected)).to_pylist()
        arrays={k:np.asarray([[int(c) for c in row[f].replace('.','0')] for row in rows],np.uint8) for k,f in [('puzzles','question'),('solutions','answer')]}
        def write(tmp):
            with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
        hub._write(dest,write)
        info={'source':SUDOKU_UPSTREAM,'revision':SUDOKU_UPSTREAM_REVISION,'sampling_seed':seed,'count':count,'validation_overlap':0,'historical_sample_reused':False}
        hub._write(root/'sampling.json',lambda tmp:tmp.write_text(json.dumps(info,indent=2)+'\n'))
    return root
