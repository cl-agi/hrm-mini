"""Pinned public assets; explicit local paths never fall back to network downloads."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

RELEASES = {
    'sudoku': ('06eb8ab49491a78eca9d13b0f12bb9a6817cf549', 'b22e1e12d5f38a7129bb53a26ec8aa1254fca502', 'f3b8577af92d563666c401ba2702614887940b88f2fe553183e0a423712223dc'),
    'sat': ('8ebf02d90d02c9060c903495c2cf8f683a0d1839', 'da71766001566bc32a20eb352ed35ced4fbc3d96', '09146cffba54e07921b485165c03850ea7f573b4c7be9031aaa58902806160f9'),
    'maze': ('d70db315176c15c1f504e844d9f2ba0f0928ecaf', '955024f1beb6797b91dafc721397c1881fb0b571', '35ec4d23ffcfb5b0907d9a3652401e30784512551f601f23ac8b2156d0731b2d'),
    'cremp': ('39143476c991de8ae72f40fc20612b8cac192da9', 'fa23ab1c4e729a8d38cb35e1864da924f67ac17c', '618646abc89247e05705c078c9e922b0544fee9695e0c362f0331b29e59b0bc5'),
}

def local_file(path, filename=None):
    path = Path(path).expanduser()
    if path.is_dir():
        if filename is None:
            raise ValueError(f'Specify a file within {path}')
        path = path / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()

def checkpoint(task, path=None):
    if path is not None:
        return local_file(path, 'model.pt')
    from huggingface_hub import hf_hub_download
    revision, _, digest = RELEASES[task]
    path = Path(hf_hub_download(f'cl-agi/{task}', 'model.pt', revision=revision))
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Published {task} checkpoint hash mismatch')
    return path

def data_file(task, filename):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(f'cl-agi/{task}', filename, repo_type='dataset', revision=RELEASES[task][1]))

def dataset(task, split, config='default', columns=None):
    """Do not infer configs from model cards: the repositories contain data only."""
    import pyarrow.parquet as pq
    return pq.read_table(data_file(task, f'data/{config}/{split}.parquet'), columns=columns).to_pylist()

def cache_path(task, name):
    from huggingface_hub.constants import HF_HOME
    root = Path(os.environ.get('HRM_ASSET_CACHE', str(Path(HF_HOME) / 'hrm-inputs-v1')))
    path = root / task / RELEASES[task][1] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _write(path, writer):
    # Separate temporary files keep simultaneous readers from seeing partial caches.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path

def puzzle_data(task, path=None, *, split='test', size=30):
    if path is not None:
        path = Path(path).expanduser()
        if task == 'sudoku' and path.is_dir() and split in ('test', 'test_hard'):
            if not (path / (split+'.csv')).exists():
                alias = 'test_hard' if split == 'test' else 'test'
                return local_file(path, alias+'.csv')
        return local_file(path, split + '.csv')
    if task not in ('sudoku', 'maze'):
        raise ValueError(task)
    split = 'test' if task == 'sudoku' and split == 'test_hard' else split
    dest = cache_path(task, f'{split}' + (f'_n{size}' if task == 'maze' else '') + '.csv')
    if dest.exists():
        return dest
    rows = dataset(task, split)
    if task == 'sudoku':
        rows = [(r['puzzle'], r['solution']) for r in rows]
    else:
        rows = [(''.join(r['maze']), ''.join(r['solution'])) for r in rows if len(r['maze']) == size]
    if not rows:
        raise ValueError(f'No rows for {task}/{split}, size={size}')
    def write(tmp):
        with tmp.open('w', newline='') as f:
            out = csv.writer(f); out.writerow(['question', 'answer']); out.writerows(rows)
    return _write(dest, write)

def sat_public_data(path=None, *, n_vars=1024):
    if path is not None:
        root = Path(path).expanduser().resolve()
        if not root.is_dir(): raise NotADirectoryError(root)
        for density in ('4.2', '4.3'): local_file(root / f'data_{density}.npz')
        return root
    import numpy as np
    root = cache_path('sat', f'public_n{n_vars}/placeholder').parent
    if all((root / f'data_{d}.npz').is_file() for d in ('4.2', '4.3')):
        return root
    rows = dataset('sat', 'test_public', 'cnf')
    for density in ('4.2', '4.3'):
        selected = [r for r in rows if r['id'].startswith(f'public:n{n_vars}:a{density}:')]
        if not selected: raise ValueError(f'No public SAT subset n{n_vars}/{density}')
        if any(r['solution'] is not None for r in selected): raise ValueError('Public SAT must have no answer cues')
        arrays = dict(clauses=np.asarray([r['clauses'] for r in selected], dtype=np.int32),
                      answers=np.zeros((len(selected), n_vars), dtype=np.int8),
                      official_ids=np.asarray([int(r['id'].rsplit(':', 1)[1]) for r in selected], dtype=np.int64))
        def write(tmp):
            with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
        _write(root / f'data_{density}.npz', write)
    return root

def cremp_data(path=None, *, split='validation'):
    if path is not None:
        return local_file(path, f'{split}.npz')
    import numpy as np
    from hrm_cremp.ringer_support import conditioning, graph_from_smiles
    dest = cache_path('cremp', split + '_angles.npz')
    if dest.exists():
        if split=='train': _bind_cremp_training_geometry(dest)
        return dest
    rows = dataset('cremp', split, columns=['id','smiles','angles'])
    graphs, masks, lengths, angles, owners = [], [], [], [], []
    for i, row in enumerate(rows):
        graph, mask, ring = conditioning(graph_from_smiles(row['smiles']))
        graphs.append(np.pad(graph, ((0, 18-len(ring)), (0, 0))))
        masks.append(np.pad(mask, ((0, 18-len(ring)), (0, 0))))
        lengths.append(len(ring)); angles.extend(row['angles']); owners.extend([i]*len(row['angles']))
    arrays = dict(ids=np.asarray([r['id'] for r in rows]), graph_smiles=np.asarray([r['smiles'] for r in rows]),
                  graph_features=np.asarray(graphs, np.float32), field_mask=np.asarray(masks),
                  lengths=np.asarray(lengths), angles=np.asarray(angles, np.float32), conf_molecule=np.asarray(owners, np.int64))
    def write(tmp):
        with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    _write(dest, write)
    if split=='train': _bind_cremp_training_geometry(dest)
    return dest

def _bind_cremp_training_geometry(cache):
    geometry=cremp_geometry(data_file('cremp','geometry.json'))
    geometry['published_source_cache_sha256']=geometry['source_cache_sha256']
    geometry['source_cache_sha256']=hashlib.sha256(cache.read_bytes()).hexdigest()
    geometry['dataset_revision']=RELEASES['cremp'][1]
    geometry['binding']='Published train split converted to the training NPZ representation'
    _write(cache.with_suffix('.geometry.json'),lambda tmp:tmp.write_text(json.dumps(geometry,indent=2)+'\n'))

def cremp_geometry(path):
    """Restore provenance only for the byte-identified published geometry asset."""
    path=local_file(path)
    content=path.read_bytes()
    statistics=json.loads(content)
    if hashlib.sha256(content).hexdigest()=='4697e22104eac4dc5fb931a0f4fba8aa5606269eed6ea4df46556a81bfed023b':
        statistics.update(source_split='train',source_cache_sha256='9f7a5ca11dfcf23ade41894653d4bfa0c2249d40bd358205df740e542947e698')
    if statistics.get('source_split')!='train' or not statistics.get('source_cache_sha256'):
        raise ValueError('Geometry must identify its training split and source cache; unrecognized data-only geometry cannot be bound to a checkpoint')
    return statistics


def model_config(task):
    """Inference architecture stored with code, since model repositories contain weights only."""
    arch = dict(name='hrm@HRM', num_layers=2, hidden_size=512, intermediate_size=2048,
                head_dim=64, norm_eps=1e-6, rope_theta=10000., H_cycles=2, L_cycles=6,
                bptt=True, forward_dtype='bfloat16', is_causal=False)
    if task == 'maze':
        arch.update(rope_type='2d', allowed_sizes=[15,20,25,30,32,35,40,45,50,60,80], is_mlp_mixer=False)
    if task == 'sat':
        arch.update(name='hrm_flow_bias@HRM', seq_len=6411, vocab_size=4, flow_lambda=1.,
                    mixture_logit_std=1., preserve_first_zero=True, normalize_mixture_variance=True, use_rope=False)
    return arch

def sat_training_config():
    return dict(arch=model_config('sat'), seed=2, model_family='hrm',
                deletion_max_fraction=0., noise_std=1., p_max=1.5, leak_amplitude_clip=1.)

def sat_eval_config(n_vars=1024):
    if n_vars not in (512, 1024): raise ValueError('Public SAT supports 512 or 1024 variables')
    return dict(n_vars=n_vars, model_family='hrm', microbatch=16,
                seed={512:1793005006,1024:1792906882}[n_vars], data_hashes={'4.2':None,'4.3':None})
