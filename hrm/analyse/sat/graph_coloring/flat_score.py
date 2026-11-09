from pathlib import Path
import json,argparse,numpy as np
parser=argparse.ArgumentParser();parser.add_argument('model_dir',type=Path);args=parser.parse_args();reports=[]
for root in [args.model_dir]:
 family=json.loads((root/'config.json').read_text())['model'];cfg=json.loads((root/'config.json').read_text());clauses=np.load(cfg['data_file'])['clauses'];graphs=[]
 for cnf in clauses:
  edges=set()
  for row in cnf:
   u=sorted(set(map(int,row)))
   if len(u)==2 and max(u)<0:
    a,b=sorted(-x-1 for x in u)
    if a//3!=b//3:assert a%3==b%3;edges.add((a//3,b//3))
  assert len(edges)==479;graphs.append(np.asarray(sorted(edges)))
 decoded=np.zeros(100,bool);cnf_ok=np.zeros(100,bool);seen=np.zeros((100,256),bool);complete=True
 for rank in range(8):
  p=root/f'rank{rank:02d}'/'status.json';complete &= p.exists() and json.loads(p.read_text()).get('state')=='complete'
  for p in sorted((root/f'rank{rank:02d}').glob('b*.npz')):
   z=np.load(p);assert not z['preflight'].item();pp=z['predictions'];ids=z['case_ids'];starts=z['starts'];assert np.isin(pp,[1,2]).all() and len(pp)==256
   for j,(i,s) in enumerate(zip(ids,starts)):
    assert not seen[i,s] and i%8==rank;seen[i,s]=True;colors=(pp[:,j].reshape(256,200,3)==2).argmax(-1);edges=graphs[i];valid=(colors[:,edges[:,0]]!=colors[:,edges[:,1]]).all(-1);full=z['success'][:,j];assert np.all(~full|valid);decoded[i]|=valid.any();cnf_ok[i]|=full.any()
 coverage=bool(np.all(cnf_ok|seen.all(1)));reports.append({'model':family,'complete':bool(complete and coverage),'cases_seen':int(seen.any(1).sum()),'decoded_correct':int(decoded.sum()),'cnf_correct':int(cnf_ok.sum()),'denominator':100,'attempts':int(seen.sum()),'scope':'Decoded first-true coloring as formal figure; CNF success separately. Partial runs never labeled full reproduction.'})
(args.model_dir/'decoded_summary.json').write_text(json.dumps(reports,indent=2));print(reports)
