from hrm.paths import analysis_directory
from pathlib import Path
import json,numpy as np
P=analysis_directory(__file__)
result={}
for model in ['rt','hrm']:
 d=P/model;states=[json.loads(x.read_text()) for x in sorted(d.glob('rank*/status.json'))]
 curves=[dict(np.load(x)) for x in sorted(d.glob('rank*/trajectory.npz'))]
 rows=[]
 if len(curves)==4:
  end=min(len(x['solved']) for x in curves)
  for b in [64,128,256,512,1024,2048,4096,8192,16384]:
   if b>end:break
   rows.append(dict(outer=b,cases=sum(len(x['ids']) for x in curves),final_solved=sum(int(x['solved'][b-1].sum()) for x in curves),any_solved=sum(int(x['solved'][:b].any(0).sum()) for x in curves),mean_violated=float(np.concatenate([x['violated'][b-1] for x in curves]).mean())))
 result[model]=dict(ranks=states,budgets=rows)
t=P/'summary.tmp';t.write_text(json.dumps(result,indent=2));t.replace(P/'summary.json');print(json.dumps(result,indent=2))
