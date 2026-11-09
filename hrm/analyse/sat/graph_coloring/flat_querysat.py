"""Flat600 QuerySAT: frozen selected model, 4 starts, selected head, 2048 rounds."""
from pathlib import Path
import argparse,json,os,time
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','4')
import numpy as np
import tensorflow as tf
from hrm_querysat.optimization.AdaBelief import AdaBeliefOptimizer
from hrm_querysat.trajectory_model import ObservedQuerySAT
from hrm.analyse.sat.graph_coloring import querysat as q
from hrm.analyse.sat.graph_coloring.flat_metrics import read_dimacs
from hrm.paths import repo_path,work_path

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=repo_path('data/sat/flat200/querysat.json'));p.add_argument('--preflight',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--output',type=Path);p.add_argument('--world-size',type=int);p.add_argument('--save-predictions',action='store_true');a=p.parse_args();cfg=json.loads(a.config.read_text());base=cfg['seed'];out=a.output or work_path('sat/flat200/querysat')/('preflight' if a.preflight else f'rank{a.rank}');out.mkdir(parents=True,exist_ok=True)
shards=a.world_size or cfg['shards'];assert shards>0 and 0<=a.rank<shards;assert cfg['starts']==4 and cfg['rounds']==2048 and cfg['checkpoint_label']=='batch4_172500'
q.tf=tf;q.np=np;q.AdaBeliefOptimizer=AdaBeliefOptimizer;q.HERE=out
flat=json.loads(repo_path('data/sat/flat200/hrm.json').read_text());data=repo_path(flat['data_file']);assert q.digest(data)==flat['data_sha256'];clauses=np.load(data)['clauses'];assert clauses.shape==(100,2237,3)
mc=json.loads(repo_path('data/sat/graph_coloring/querysat.json').read_text())['models']['batch4_172500'];ckpt=repo_path(mc['checkpoint'])
for suffix,key in [('.index','index_sha'),('.data-00000-of-00001','data_sha')]:assert q.digest(str(ckpt)+suffix)==mc[key]
gpus=tf.config.list_physical_devices('GPU');assert len(gpus)==1
for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
tf.config.experimental.enable_tensor_float_32_execution(False);tf.keras.utils.set_random_seed(base)
def sample(c):
 rows=[list(dict.fromkeys(map(int,row))) for row in c];pairs=[[abs(v)-1+(600 if v<0 else 0),i] for i,row in enumerate(rows) for v in row]
 return dict(adj_matrix=q.sparse(pairs,[1200,2237]),clauses_graph=q.sparse([[0,i] for i in range(2237)],[1,2237]),variables_graph=q.sparse([[0,i] for i in range(600)],[1,600]),solutions=tf.ragged.constant([[0]*600],dtype=tf.int32,row_splits_dtype=tf.int32))
def build(inp,audit_logits=False):
 net=ObservedQuerySAT(AdaBeliefOptimizer(.0002,beta_1=.6,clip_gradients=True),test_rounds=1);net.audit_logits=audit_logits;net.call(**{k:inp[k] for k in ['adj_matrix','clauses_graph','variables_graph']},training=False);restore=tf.train.Checkpoint(model=net).restore(str(ckpt));restore.assert_existing_objects_matched();restore.expect_partial();return net
inp=sample(clauses[0]);meta=dict(config=cfg,checkpoint=mc,data_sha256=flat['data_sha256'],observer_sha256=q.digest(q.SOURCE/'trajectory_model.py'),adapter_sha256=q.digest(__file__),tensorflow=tf.__version__,seed_rule='base + 100000*start + original_case_index; stateless per-round queries',heads='Original lowest full-CNF softplus loss; no success-based head choice',protocol='600 variables, 4 starts, 2048 rounds, first-true colors; stop later starts only after selected-head full-CNF success',new_noise_stream=True)
if a.preflight:
 with tf.device('/CPU:0'):
  ci=sample(clauses[0]);audit=build(ci,True);audit.audit_logits=True;audit.test_rounds=64;meta['cpu_equivalence']=q.original_equivalence(audit,ci,base,64,ckpt)
net=build(inp);net.test_rounds=2048;net.predict_step.get_concrete_function();(out/'config.json').write_text(json.dumps(meta,indent=2));records=[]
for idx in ([0] if a.preflight else range(a.rank,100,shards)):
 c=clauses[idx];path=repo_path(flat['original_cnf_files'][idx]);assert q.digest(path)==flat['original_cnf_sha256'][idx];n,raw=read_dimacs(path);assert n==600 and all(set(x)==set(y) for x,y in zip(raw,c));edges={}
 for row in raw:
  if len(row)==2 and all(v<0 for v in row):
   u,v=[abs(t)-1 for t in row]
   if u//3!=v//3:assert u%3==v%3;edges.setdefault(tuple(sorted([u//3,v//3])),set()).add(u%3)
 assert len(edges)==479 and all(x=={0,1,2} for x in edges.values());e=np.array(sorted(edges));hit_full=False;hit_legal=False;starts=[]
 for start in range(4):
  seed=base+100000*start+idx;tf.keras.utils.set_random_seed(seed);t=time.time();res=net.predict_step(**sample(c),query_seed=tf.constant(seed,tf.int32));bits=res['trajectory_bits'].numpy().transpose(0,2,1);heads=res['trajectory_heads'].numpy();loss=res['trajectory_losses'].numpy();assert bits.shape[1:]==(8,600);assert len(bits)==int(res['steps_taken'].numpy())+1
  assert np.array_equal(heads,loss.argmin(-1));selected=bits[np.arange(len(bits)),heads];full=[];legal=[]
  for lo in range(0,len(bits),64):
   b=selected[lo:lo+64];full.extend((b[:,np.abs(c)-1]==(c>0)[None]).any(-1).all(-1));color=b.reshape(-1,200,3).argmax(-1);legal.extend((color[:,e[:,0]]!=color[:,e[:,1]]).all(-1))
  full=np.array(full);legal=np.array(legal);assert np.all(~full|legal);first=np.flatnonzero(full);assert int(res['first_hit'].numpy()[0])==(int(first[0])+1 if len(first) else 0)
  assert len(bits)==2048 or (full[-1] and len(first) and first[0]==len(bits)-1)
  if a.save_predictions: np.savez_compressed(out/f'case{idx:03d}_start{start}.npz',case_id=idx,start=start,seed=seed,bits_packed=np.packbits(bits,axis=-1,bitorder='big'),selected_head=heads,head_selection_loss=loss,full=full,legal=legal,edges=e)
  hit_full|=bool(full.any());hit_legal|=bool(legal.any());starts.append(dict(start=start,seed=seed,rounds=len(bits),full=bool(full.any()),legal=bool(legal.any()),seconds=time.time()-t))
  if hit_full:break
 row=dict(case_id=idx,any_full=hit_full,any_legal=hit_legal,starts=starts);records.append(row);print(json.dumps(row),flush=True)
totals={k:sum(int(r[k]) for r in records) for k in ['any_full','any_legal']}
(out/'complete.json').write_text(json.dumps(dict(metadata=meta,results=records,total=len(records),totals=totals,preflight=a.preflight),indent=2))
