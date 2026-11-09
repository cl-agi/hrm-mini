"""Summarize failure patterns and update sizes without assuming their mechanism."""
from hrm.paths import analysis_directory
from pathlib import Path
import csv,json
import numpy as np
root=analysis_directory(__file__);out=root/'results'
c=json.loads((root/'config.json').read_text());cohort=np.load(out/'cohort.npz');y=cohort['labels']
summary={'n_puzzles':len(y),'H_updates':c['h_updates'],'native_endpoint_H':32,'conditions':{}}
trajectory=[];examples=[]
for ci,condition in enumerate(c['conditions']):
 batches=[np.load(p) for p in sorted(out.glob(condition+'_*.npz'))]
 merged={k:np.concatenate([a[k] for a in batches]) for k in batches[0].files}
 np.testing.assert_array_equal(merged['row_id'],np.arange(len(y)))
 p=merged['predictions'];wrong=(p!=y[:,None,:]);exact=~wrong.any(-1);cell_exact=~wrong[:,:,1:].any(-1)
 clues=cohort['inputs'][ci]!=0;clue_wrong=(wrong & clues[:,None,:]).sum(-1)
 prior=np.concatenate([np.zeros((len(y),1),bool),np.maximum.accumulate(exact,axis=1)[:,:-1]],axis=1)
 reversed_at=prior & ~exact
 transitions=exact[:,:-1] & ~exact[:,1:]
 result={'ever_exact':int(exact.any(1).sum()),'never_exact':int((~exact.any(1)).sum()),'ever_correct_then_wrong':int(reversed_at.any(1).sum()),'at_H':{},'reversal_events':[]}
 for t in [1,2,4,8,16,32,64]:
  result['at_H'][str(t)]={'exact':int(exact[:,t-1].sum()),'cell_exact':int(cell_exact[:,t-1].sum()),'puzzles_with_corrupted_clues':int((clue_wrong[:,t-1]>0).sum()),'mean_wrong_cells':float(wrong[:,t-1,1:].sum(1).mean())}
 for i,t in zip(*np.where(transitions)):
  result['reversal_events'].append({'row_id':int(i),'from_H':int(t+1),'to_H':int(t+2),'delta_H_rms':float(merged['delta_H_rms'][i,t+1]),'delta_L_block_rms':float(merged['delta_L_block_rms'][i,t+1]),'new_wrong_cells':int(wrong[i,t+1,1:].sum()),'clues_corrupted':int(clue_wrong[i,t+1]),'min_true_margin_before':float(merged['true_margin'][i,t,1:].min()),'min_true_margin_after':float(merged['true_margin'][i,t+1,1:].min())})
 for group,mask in [('reversal_transition',np.pad(transitions,((0,0),(1,0)))),('stayed_correct',np.pad(exact[:,:-1]&exact[:,1:],((0,0),(1,0)))),('never_solved_trajectories',np.repeat((~exact.any(1))[:,None],exact.shape[1],axis=1))]:
  result[group]={'observations':int(mask.sum()),**{k:float(merged[k][mask].mean()) if mask.any() else None for k in ['delta_H_rms','delta_L_block_rms','relative_H']}}
 summary['conditions'][condition]=result
 selected=sorted(set(np.flatnonzero(reversed_at.any(1))[:2].tolist()+np.flatnonzero(~exact.any(1))[:2].tolist()))
 for i in selected:
  examples.append({'condition':condition,'row_id':i,'input':cohort['inputs'][ci,i,1:].tolist(),'answer':y[i,1:].tolist(),'first_correct_H':int(np.flatnonzero(exact[i])[0]+1) if exact[i].any() else None,'predictions':p[i,:,1:].tolist()})
 for i in range(len(y)):
  for t in range(exact.shape[1]):
   trajectory.append({'condition':condition,'row_id':i,'H_update':t+1,'exact':int(exact[i,t]),'wrong_cells':int(wrong[i,t,1:].sum()),'corrupted_clues':int(clue_wrong[i,t]),'delta_H_rms':float(merged['delta_H_rms'][i,t]),'delta_L_block_rms':float(merged['delta_L_block_rms'][i,t]),'relative_H':float(merged['relative_H'][i,t]),'min_true_margin':float(merged['true_margin'][i,t,1:].min())})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
with (out/'trajectories.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(trajectory[0]),lineterminator='\n');w.writeheader();w.writerows(trajectory)
print(json.dumps(summary,indent=2))
