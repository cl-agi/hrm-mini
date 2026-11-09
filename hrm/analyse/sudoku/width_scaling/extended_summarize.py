from hrm.paths import analysis_directory
import json,statistics
from pathlib import Path
P=analysis_directory(__file__);M=json.loads((P/'manifest.json').read_text());B=Path(M['parent_root']);old=json.loads((B/'manifest.json').read_text());mapping={r['parent_run_id']:r for r in M['runs']};rows=[];pending=[]
for r in old['runs']:
 ext=mapping.get(r['run_id']);w=(P/'runs'/ext['run_id']) if ext else B/'runs'/r['run_id'];f=w/'result.json'
 if not f.exists():pending.append({'run_id':ext['run_id'] if ext else r['run_id'],'target_epochs':M['budgets'][r['size']]});continue
 a=json.loads(f.read_text());rows.append({'architecture':r['architecture'],'size':r['size'],'seed':r['seed'],'epochs':M['budgets'][r['size']],'best_percent':100*a['best']['exact_match'],'final_percent':100*a['final']['exact_match'],'best_epoch':a['best']['epoch']+1})
summary=[]
for arch in ['hrm','rt']:
 for size in ['XS','S','base','L','XL']:
  rr=[r for r in rows if r['architecture']==arch and r['size']==size]
  if not rr:continue
  x={'architecture':arch,'size':size,'epochs':M['budgets'][size],'n':len(rr)}
  for metric in ['best_percent','final_percent']:
   v=[r[metric] for r in rr];x[metric+'_mean']=statistics.mean(v);x[metric+'_sd']=statistics.stdev(v) if len(v)>1 else None
  summary.append(x)
result={'completed_target_budget':len(rows),'planned':30,'summary':summary,'per_seed':rows,'pending':pending,'resume_note':M['resume_note']}
(P/'results_snapshot.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
