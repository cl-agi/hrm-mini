"""Prepare E13 Maze-Hard independently from unrelated SAT archives."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from hrm.paths import repository_root
from .metrics import geometry,cpu_valid

SOURCE='sapientinc/maze-30x30-hard-1k'
REVISION='549754de3d67eebe57b6b22c7d226b9412d55b07'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(path):
    vocab={c:i for i,c in enumerate('~# SGo')}
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=1000:raise ValueError('E13 requires exactly 1000 rows in each source split')
    if any(len(r['question'])!=900 or len(r['answer'])!=900 for r in rows):
        raise ValueError('E13 requires 30x30 mazes')
    arrays={key:np.asarray([[0]+[vocab[c] for c in row[field]] for row in rows],dtype=np.int16)
            for key,field in [('x','question'),('y','answer')]}
    arrays['ids']=np.arange(len(rows));arrays['ds'],arrays['dg']=geometry(arrays['x'])
    if len(set(map(bytes,arrays['x'])))!=1000:raise ValueError('Duplicate maze inputs')
    if not all(cpu_valid(y,x) for x,y in zip(arrays['x'],arrays['y'])):
        raise ValueError('A source answer is not a valid shortest path')
    return arrays


def prepare(output=None,dataset_path=None,seed=None):
    output=Path(output) if output is not None else repository_root()/'exp/maze/records/continuous/data'
    manifest=output/'manifest.json'
    source_name=str(Path(dataset_path).resolve()) if dataset_path is not None else SOURCE
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record['source']!=source_name or (seed is not None and record['seed']!=seed):
            raise ValueError('Existing data source/seed differs; choose a new output directory')
        if dataset_path is not None:
            for split,sha in record['input_sha256'].items():
                if digest(Path(dataset_path)/(split+'.csv'))!=sha:
                    raise ValueError('Local source changed: '+split)
        for name,sha in record['files'].items():
            if digest(output/name)!=sha:raise ValueError('Prepared data changed: '+name)
        return output
    if output.exists() and any(output.iterdir()):raise FileExistsError('Incomplete/nonempty preparation directory; choose a new output')
    output.mkdir(parents=True,exist_ok=True)
    seed=int(time.time()) if seed is None else int(seed)
    record=dict(source=source_name,revision=REVISION if dataset_path is None else None,seed=seed,
                policy='Keep 1000 train mazes; shuffle 1000 held-out IDs, first200 validation and remaining800 test')
    (output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    if dataset_path is None:
        from huggingface_hub import hf_hub_download
        paths={s:Path(hf_hub_download(SOURCE,s+'.csv',repo_type='dataset',revision=REVISION)) for s in ('train','test')}
    else:
        paths={s:Path(dataset_path)/(s+'.csv') for s in ('train','test')}
    arrays={s:encode(path) for s,path in paths.items()}
    if set(map(bytes,arrays['train']['x'])) & set(map(bytes,arrays['test']['x'])):
        raise ValueError('Maze train and held-out inputs overlap')
    order=np.random.default_rng(seed).permutation(1000)
    arrays['val']={k:v[order[:200]] for k,v in arrays['test'].items()}
    arrays['test']={k:v[order[200:]] for k,v in arrays['test'].items()}
    record['input_sha256']={s:digest(p) for s,p in paths.items()};record['files']={};record['splits']={}
    for split,values in arrays.items():
        name=f'maze_{split}.npz';np.savez_compressed(output/name,**values)
        record['files'][name]=digest(output/name);record['splits'][split]=len(values['ids'])
    record['checks']=dict(train_heldout_disjoint=True,all_targets_valid_shortest_paths=True,heldout_partition_complete=True)
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--dataset-path',type=Path);p.add_argument('--seed',type=int)
    a=p.parse_args(argv);print(prepare(a.output,a.dataset_path,a.seed))

if __name__=='__main__':main()
