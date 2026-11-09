from hrm.paths import analysis_directory
import json,statistics,csv
from pathlib import Path
P=analysis_directory(__file__);M=json.loads((P/'manifest.json').read_text());rows=[];states=[]
for r in M['runs']:
 w=P/'runs'/r['run_id'];f=w/'result.json';st=w/'status.json'
 states.append(json.loads(st.read_text()) if st.exists() else {'run_id':r['run_id'],'stage':'not_started'})
 if not f.exists():continue
 a=json.loads(f.read_text());rows.append({'architecture':r['architecture'],'size':r['size'],'seed':r['seed'],'final_percent':100*a['final']['exact_match'],'best_percent':100*a['best']['exact_match'],'best_epoch_1based':a['best']['epoch']+1,'lr':r['config']['lr'],'wd':r['config']['weight_decay']})
summary=[]
for arch in ['hrm','rt']:
 for size in ['XS','S','base','L','XL']:
  rr=[x for x in rows if x['architecture']==arch and x['size']==size]
  if not rr:continue
  s={'architecture':arch,'size':size,'n':len(rr),'seeds':[x['seed'] for x in rr]}
  for metric in ['final_percent','best_percent']:
   v=[x[metric] for x in rr];s[metric+'_mean']=statistics.mean(v);s[metric+'_sd']=statistics.stdev(v) if len(v)>1 else None
  summary.append(s)
result={'completed':len(rows),'planned':30,'states':states,'summary':summary,'runs':rows};(P/'results_snapshot.json').write_text(json.dumps(result,indent=2))
if rows:
 with (P/'per_seed_results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps({'completed':len(rows),'planned':30,'summary':summary},indent=2))
