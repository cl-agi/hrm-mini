"""Flat600 adapter: reuse the frozen NeuroSAT model and instrumented recurrence."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,hashlib,time
import numpy as np
import torch
from hrm.analyse.sat.graph_coloring import neurosat as ns
from hrm.analyse.sat.graph_coloring.flat_metrics import read_dimacs
from hrm.paths import repo_path,work_path
p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=4);p.add_argument('--output',type=Path);p.add_argument('--save-predictions',action='store_true');a=p.parse_args()
assert a.world_size > 0 and 0 <= a.rank < a.world_size
out=a.output or work_path('sat/flat200/neurosat')/('preflight' if a.preflight else f'rank{a.rank}')
out.mkdir(parents=True,exist_ok=True)
cfg=json.loads(repo_path('data/sat/flat200/hrm.json').read_text());src=repo_path(cfg['data_file']);assert ns.sha256(src)==cfg['data_sha256'];z=np.load(src);clauses=z['clauses'];assert clauses.shape==(100,2237,3)
model_config=json.loads(repo_path('data/sat/graph_coloring/neurosat.json').read_text());seed=model_config['seed_base'];torch.manual_seed(seed);torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ns.N_VARIABLES=600;device=torch.device('cuda:0');ns.check_source_equivalence();model,info=ns.load_model(device, repo_path(model_config["checkpoint"]))
metadata=dict(checkpoint_sha256=ns.CHECKPOINT_SHA256,data_sha256=cfg['data_sha256'],seed=seed,initialization='fixed learned state; original baseline seed retained',starts=1,iterations=2048,batch_size=1,variables=600,vertices=200,edges=479,torch=str(torch.__version__),source_sha256=ns.sha256(Path(ns.__file__)),adapter_sha256=ns.sha256(__file__),scope='Frozen weights; exact original CNFs, first-true decoder without repair; any and final iteration separate',checkpoint=info)
(out/'config.json').write_text(json.dumps(metadata,indent=2));results=[]
for idx in ([0] if a.preflight else range(a.rank,100,a.world_size)):
 c=clauses[idx];cnf=repo_path(cfg['original_cnf_files'][idx]);assert ns.sha256(cnf)==cfg['original_cnf_sha256'][idx];n,original=read_dimacs(cnf);assert n==600 and all(set(x)==set(y) for x,y in zip(original,c))
 edges={};positives=0;same=0
 for row in original:
  if all(v>0 for v in row):
   assert len(row)==3 and {(v-1)//3 for v in row}=={(row[0]-1)//3};positives+=1;continue
  assert len(row)==2 and all(v<0 for v in row)
  u,v=[abs(t)-1 for t in row]
  if u//3==v//3:same+=1;continue
  assert u%3==v%3;edge=tuple(sorted([u//3,v//3]));edges.setdefault(edge,set()).add(u%3)
 assert positives==200 and same==600 and len(edges)==479 and all(v=={0,1,2} for v in edges.values());e=np.array(sorted(edges))
 keep=np.ones(c.shape,dtype=bool)
 for j in range(1,3):keep[:,j]=(c[:,j,None]!=c[:,:j]).all(-1)
 lit=2*(np.abs(c)-1)+(c<0);clause=np.broadcast_to(np.arange(len(c))[:,None],c.shape)
 graph=SimpleNamespace(num_graphs=1,l_size=torch.tensor([1200],device=device),c_size=torch.tensor([2237],device=device),l_edge_index=torch.as_tensor(lit[keep],device=device),c_edge_index=torch.as_tensor(clause[keep],device=device))
 parity=[]
 with torch.inference_mode():
  if a.preflight:
   for steps in [1,3,8]:
    model.opts.n_iterations=steps;expected=model(graph);pred,actual=ns.trajectory(model,graph,steps);torch.testing.assert_close(actual.reshape(-1),expected,rtol=1e-5,atol=1e-6);assert torch.equal(pred[-1].reshape(-1),expected>0.5);parity.append(dict(iterations=steps,max_abs_error=float((actual.reshape(-1)-expected).abs().max())))
  model.opts.n_iterations=2048;start=time.time();pred,_=ns.trajectory(model,graph,2048);v=pred[:,0].cpu().numpy()
 full=[];legal=[]
 for lo in range(0,2048,64):
  b=v[lo:lo+64];full.extend((b[:,np.abs(c)-1]==(c>0)[None]).any(-1).all(-1));color=b.reshape(-1,200,3).argmax(-1);legal.extend((color[:,e[:,0]]!=color[:,e[:,1]]).all(-1))
 full=np.array(full);legal=np.array(legal);assert np.all(~full|legal)
 if a.save_predictions: np.savez_compressed(out/f'case{idx:03d}.npz',case_id=idx,predictions=np.packbits(v,axis=-1,bitorder='little'),full=full,legal=legal,edges=e)
 r=dict(case_id=idx,cnf_sha256=ns.sha256(cnf),any_full=bool(full.any()),any_legal=bool(legal.any()),final_full=bool(full[-1]),final_legal=bool(legal[-1]),seconds=time.time()-start,endpoint_parity=parity);results.append(r);print(json.dumps(r),flush=True)
totals={k:sum(int(r[k]) for r in results) for k in ['any_full','any_legal','final_full','final_legal']}
(out/'complete.json').write_text(json.dumps(dict(metadata=metadata,results=results,total=len(results),totals=totals,preflight=a.preflight),indent=2))
