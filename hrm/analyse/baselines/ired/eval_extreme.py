"""Evaluate best SAT-Net checkpoint on both official extreme-1k test splits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from hrm.training.ired_core import model, evaluate
from hrm.analyse.baselines.ired.train_extreme_aug import Train1k
from hrm import hub
from hrm.paths import repo_path, work_path

OUT=work_path('ired/evaluation')

def data(data_path=None):
    names=('test_hard.csv','test_sudoku_bench.csv')
    if data_path is not None:
        paths=[hub.local_file(Path(data_path)/name) for name in names]
    else:
        from huggingface_hub import hf_hub_download
        paths=[Path(hf_hub_download('sapientinc/sudoku-extreme-1k',name,repo_type='dataset',
                   revision='48941acb15ff8cc57f6fb355dded588ef6c3c2b8')) for name in names]
    rows=[]
    for path in paths:
        with path.open() as stream:rows.append(list(csv.DictReader(stream)))
    assert [len(r) for r in rows] == [20000,100]
    ds=Train1k.__new__(Train1k)
    ds.augment=False
    convert=lambda key: np.array([[int(c) for c in row[key].replace('.','0')]
                                 for split in rows for row in split],dtype=np.int64)
    ds.x,ds.y=convert('question').reshape(-1,9,9),convert('answer').reshape(-1,9,9)
    return ds,paths

def main():
    global OUT, CKPT
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda')
    p.add_argument('--data-path',type=Path)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=8)
    p.add_argument('--merge',action='store_true')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--expected-step',type=int,default=60000)
    p.add_argument('--training-description',default='SAT-Net only; no extreme training or fine-tuning')
    a=p.parse_args();OUT=a.output;CKPT=a.checkpoint;OUT.mkdir(parents=True,exist_ok=True)
    if a.shards<1 or not 0<=a.shard<a.shards:p.error("invalid shard index/count")
    ds,paths=data(a.data_path)
    if a.merge:
        target=ds.y.reshape(-1,81);inputs=ds.x.reshape(-1,81)
        pred=np.zeros_like(target);covered=np.zeros(len(ds),dtype=bool)
        for i in range(a.shards):
            ids=np.load(OUT/f'shard{i}.indices.npy')
            expected=np.arange(i,len(ds),a.shards)
            if not np.array_equal(ids,expected):raise ValueError(f'Shard {i} indices differ from the required ordered partition')
            values=np.load(OUT/f'shard{i}.predictions.npy')
            if values.shape!=(len(ids),81):raise ValueError(f'Shard {i} prediction shape mismatch')
            if covered[ids].any():raise ValueError('Overlapping shards')
            pred[ids]=values;covered[ids]=True
        assert covered.all()
        exact=(pred==target).all(1)
        board=pred.reshape(-1,9,9); digits=np.arange(1,10)
        rows=(np.sort(board,axis=2)==digits).all((1,2))
        cols=(np.sort(board,axis=1)==digits[None,:,None]).all((1,2))
        boxes=board.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        blocks=(np.sort(boxes,axis=2)==digits).all((1,2))
        clues=((inputs==0)|(pred==inputs)).all(1)
        valid=rows & cols & blocks & clues
        out=dict(checkpoint=str(CKPT),checkpoint_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),
                 checkpoint_step=a.expected_step,weights='ema',landscapes=10,inner_steps=20,candidates_per_puzzle=1,
                 trained_on=a.training_description,
                 shards=a.shards,shard_seed_rule='42 + shard index; indices stride by shard count',splits={})
        for name,lo,hi,path in [('test_hard',0,20000,paths[0]),('test_sudoku_bench',20000,len(ds),paths[1])]:
            out['splits'][name]=dict(n=hi-lo,exact_correct=int(exact[lo:hi].sum()),
                 accuracy=float(exact[lo:hi].mean()),strict_valid_and_clues=int(valid[lo:hi].sum()),
                 clues_preserved=int(clues[lo:hi].sum()),correct_indices=np.flatnonzero(exact[lo:hi]).tolist(),
                 source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        np.save(OUT/'all_predictions.npy',pred)
        (OUT/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
        return
    torch.set_num_threads(4);torch.manual_seed(42)
    ckpt=torch.load(CKPT,map_location='cpu',weights_only=False)
    assert ckpt['step']==a.expected_step
    m=model()
    m.load_state_dict({k.removeprefix('ema_model.'):v for k,v in ckpt['ema'].items() if k.startswith('ema_model.')},strict=True)
    m.to(a.device).eval()
    ids=np.arange(a.shard,len(ds),a.shards)
    np.save(OUT/f'shard{a.shard}.indices.npy',ids)
    evaluate(m,Subset(ds,ids.tolist()),a.device,64,42+a.shard,OUT/f'shard{a.shard}.json')

if __name__=='__main__':main()
