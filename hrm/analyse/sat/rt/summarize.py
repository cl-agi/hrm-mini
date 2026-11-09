from hrm.paths import analysis_directory
from pathlib import Path
import json,sqlite3,time,collections
P=analysis_directory(__file__);cfg=json.loads((P/'config.json').read_text());db=sqlite3.connect(P/'queue.sqlite');groups={};by=collections.defaultdict(list)
for r in db.execute('select density,case_id,lo,hi,status,hit,checked_end from tasks'):by[(r[0],r[1])].append(r)
for tag,total in cfg['counts'].items():
 states=[]
 for i in range(total):
  rr=[r for r in by[(tag+'_o64',i)] if r[2]<cfg['max_starts']];hit=min([r[5] for r in rr if r[5]>0],default=0);prefix=0
  for r in sorted(rr,key=lambda r:r[2]):
   if r[4] in ['done','paused'] and r[2]<=prefix:prefix=max(prefix,r[6])
  states.append(dict(index=i,official_id=cfg['selected_ids'][tag][i],first_hit=hit,checked_prefix=prefix))
 curve=[]
 for k in [2**i for i in range(cfg['max_starts'].bit_length())]:
  solved=sum(0<x['first_hit']<=k for x in states);known=sum(x['checked_prefix']>=k or 0<x['first_hit']<=k for x in states);curve.append(dict(starts=k,solved=solved,total=total,fully_evaluated=known==total,rate=solved/total if known==total else None))
 groups[tag]=dict(complete=all((x['first_hit']>0 and x['checked_prefix']>=x['first_hit']) or x['checked_prefix']>=cfg['max_starts'] for x in states),found=sum(x['first_hit']>0 for x in states),total=total,curve=curve,states=states)
out=dict(complete=all(x['complete'] for x in groups.values()),n_vars=cfg['n_vars'],groups=groups,task_counts=dict(db.execute('select status,count(*) from tasks group by status').fetchall()),time=time.time());t=P/'summary.tmp';t.write_text(json.dumps(out,indent=2));t.replace(P/'summary.json')
