"""Re-score saved IRED predictions; no inference or training."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
report={}
for arm in ['scratch_extreme_1k/augmentation/evaluation','extreme_best60k']:
 root=a.root/arm;record=json.loads((root/'results.json').read_text());pred=np.load(root/'all_predictions.npy');assert pred.shape==(20100,81)
 seen=set()
 for i in range(8):
  ids=np.load(root/f'shard{i}.indices.npy');v=np.load(root/f'shard{i}.predictions.npy');assert not seen.intersection(ids.tolist());seen.update(ids.tolist());assert np.array_equal(pred[ids],v)
 assert len(seen)==20100
 report[arm]={}
 for name,lo,hi in [('test_hard',0,20000),('test_sudoku_bench',20000,20100)]:
  f=a.dataset/(name+'.csv');assert hashlib.sha256(f.read_bytes()).hexdigest()==record['splits'][name]['sha256']
  rows=list(csv.DictReader(f.open()));assert len(rows)==hi-lo
  x=np.array([[int(c) for c in r['question'].replace('.','0')] for r in rows]);y=np.array([[int(c) for c in r['answer']] for r in rows]);v=pred[lo:hi];exact=(v==y).all(1)
  valid=[]
  for board,clues in zip(v,x):
   b=board.reshape(9,9);units=[*b,*b.T]+[b[i:i+3,j:j+3].ravel() for i in [0,3,6] for j in [0,3,6]]
   valid.append(all(set(u.tolist())==set(range(1,10)) for u in units) and bool(((clues==0)|(board==clues)).all()))
  assert int(exact.sum())==record['splits'][name]['exact_correct'];assert sum(valid)==record['splits'][name]['strict_valid_and_clues'];assert np.array_equal(np.flatnonzero(exact),record['splits'][name]['correct_indices'])
  report[arm][name]=dict(n=len(rows),exact=int(exact.sum()),strict_valid_and_clues=sum(valid))
print(json.dumps(dict(passed=True,results=report),indent=2))
