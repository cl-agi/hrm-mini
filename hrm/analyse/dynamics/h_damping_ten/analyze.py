"""Analyze all predeclared interventions, including harms and null effects."""
from hrm.paths import analysis_directory
from pathlib import Path
import csv,json,hashlib
import numpy as np
b=analysis_directory(__file__);r=b/'results';c=json.loads((b/'config.json').read_text());co=np.load(r/'cohort.npz')
rows=co['row_id'];y=co['labels'];tokens=co['target_token'];truth=y[np.arange(len(y)),tokens];clues=co['input_ids']!=0
base=np.load(r/'baseline_bf16.npz');bl=base['target_logits'][:,63].copy();bl[np.arange(len(rows)),truth]=-np.inf;rival=bl.argmax(-1)
summary={'case_selection':c['selection'],'n':len(rows),'training':False,'variants':{}};table=[];traces=[]
for v in c['variants']:
 a=np.load(r/(v['name']+'.npz'));np.testing.assert_array_equal(a['row_id'],rows)
 exact=(a['predictions']==y[:,None,:]).all(-1);ever=np.maximum.accumulate(exact,1)
 out={'variant':v,'final_correct':int(exact[:,-1].sum()),'ever_correct_then_wrong':int((ever&~exact).any(1).sum()),'per_puzzle':[]}
 h=a['H_target'];old=np.concatenate([np.broadcast_to(a['H_init'],(len(rows),1,h.shape[-1])),h[:,:-1]],1);delta=h-old
 w=a['readout_weight'][truth]-a['readout_weight'][rival]
 proj=np.einsum('btd,bd->bt',delta,w)
 fixed_margin=a['target_logits'][np.arange(len(rows))[:,None],np.arange(exact.shape[1])[None,:],truth[:,None]]-a['target_logits'][np.arange(len(rows))[:,None],np.arange(exact.shape[1])[None,:],rival[:,None]]
 logits=a['target_logits'].copy();true_logit=np.take_along_axis(logits,truth[:,None,None],2).squeeze(2);logits[np.arange(len(rows))[:,None],np.arange(exact.shape[1])[None,:],truth[:,None]]=-np.inf
 margin=true_logit-logits.max(-1)
 for i,row in enumerate(rows):
  bad=np.flatnonzero(ever[i]&~exact[i]);first=np.flatnonzero(exact[i]);cluewrong=((a['predictions'][i]!=y[i])&clues[i]).sum(1)
  rec={'row_id':int(row),'target_token':int(tokens[i]),'truth':int(truth[i]),'fixed_rival':int(rival[i]),'first_correct_H':int(first[0]+1) if len(first) else None,'first_reversal_H':int(bad[0]+1) if len(bad) else None,'correct_H32':bool(exact[i,31]),'correct_H64':bool(exact[i,63]),'correct_H128':bool(exact[i,127]),'correct_final':bool(exact[i,-1]),'final_target_prediction':int(a['predictions'][i,-1,tokens[i]]),'final_margin':float(margin[i,-1]),'max_corrupted_clues':int(cluewrong.max()),'early4_mean_delta_H_rms':float(a['delta_H_rms'][i,:4].mean()),'early4_mean_delta_H_target_rms':float(a['delta_H_target_rms'][i,:4].mean()),'late_mean_delta_H_rms':float(a['delta_H_rms'][i,63:].mean())}
  out['per_puzzle'].append(rec);table.append({'variant':v['name'],**rec})
  for t in range(exact.shape[1]):
   path=a['H_pathlength_after_H4'][i,t];rho=a['H_displacement_from_H4'][i,t]/path if path else 0.
   traces.append({'variant':v['name'],'row_id':int(row),'H':t+1,'exact':int(exact[i,t]),'target_prediction':int(a['predictions'][i,t,tokens[i]]),'best_digit_excluding_blank':int(a['target_logits'][i,t,1:].argmax()+1),'margin':float(margin[i,t]),'fixed_rival_margin':float(fixed_margin[i,t]),'readout_projection_delta':float(proj[i,t]),'fixed_margin_delta':float(fixed_margin[i,t]-fixed_margin[i,t-1]) if t else None,'delta_H_global':float(a['delta_H_rms'][i,t]),'delta_H_target':float(a['delta_H_target_rms'][i,t]),'delta_H_other':float(a['delta_H_other_rms'][i,t]),'target_drift_cosine':float(np.dot(delta[i,t],w[i])/(np.linalg.norm(delta[i,t])*np.linalg.norm(w[i])+1e-30)),'global_adjacent_update_cosine':float(a['cos_delta_H'][i,t]),'displacement_over_path_after_H4':float(rho),'H_rms':float(a['H_rms'][i,t])})
 summary['variants'][v['name']]=out
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name,data in [('outcomes.csv',table),('trajectories.csv',traces)]:
 with (r/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
for name,v in summary['variants'].items():
 print(name,'correct',v['final_correct'],[(a['row_id'],a['first_correct_H'],a['first_reversal_H'],a['final_target_prediction'],round(a['final_margin'],3)) for a in v['per_puzzle'] if a['row_id'] in [41,56]])
