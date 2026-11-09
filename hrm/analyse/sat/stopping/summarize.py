from hrm.paths import analysis_directory
from pathlib import Path
import json,sqlite3,time
P=analysis_directory(__file__);c=json.loads((P/'config.json').read_text());db=sqlite3.connect(P/'queue.sqlite',timeout=120)
rows=db.execute('select case_id,lo,hi,status,hit,any_hit,checked_end from tasks').fetchall();states=[]
for i in range(108):
 rr=sorted([r for r in rows if r[0]==i],key=lambda r:r[1]);prefix=0
 for r in rr:
  if r[3]=='done' and r[1]<=prefix:prefix=max(prefix,r[6])
 states.append(dict(case=i,checked_prefix=prefix,final_hit=min([r[4] for r in rr if r[4]>0],default=0),any_hit=min([r[5] for r in rr if r[5]>0],default=0)))
curves={}
for metric in ['final','any']:
 curve=[]
 for k in [2**j for j in range(15)]:
  hit=lambda s:s[metric+'_hit']
  solved=sum(0<hit(s)<=k for s in states);covered=sum(s['checked_prefix']>=k or 0<hit(s)<=k for s in states)
  curve.append(dict(starts=k,solved=solved,total=108,fully_evaluated=covered==108,rate=solved/108 if covered==108 else None))
 curves[metric]=curve
out=dict(model=c['model'],complete=all(s['final_hit']>0 or s['checked_prefix']>=c['max_starts'] for s in states),curves=curves,states=states,task_counts=dict(db.execute('select status,count(*) from tasks group by status')),time=time.time())
t=P/'summary.tmp';t.write_text(json.dumps(out,indent=2));t.replace(P/'summary.json')
