from hrm.paths import repository_root
import os
from hrm.paths import repo_path
"""Frozen task encodings, semantic augmentation, and target-free validators."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(os.environ.get("HRM_ANALYSIS_RECORDS",repository_root()/"exp/maze/records/continuous"))
SEED=20260908

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def augment_sat(c, a, generator, polarity=True):
    """old variable i -> perm[i]; new truth = old truth XOR flip[i]."""
    b,m,k=c.shape; n=a.shape[1]; dev=c.device
    perm=torch.rand(b,n,generator=generator,device=dev).argsort(1)
    flip=torch.randint(0,2,(b,n),generator=generator,device=dev).bool() if polarity else torch.zeros_like(a,dtype=torch.bool)
    old=c.abs().long()-1
    new=perm.gather(1,old.flatten(1)).view_as(c)+1
    signs=c.sign()*(1-2*flip.gather(1,old.flatten(1)).view_as(c).long())
    result=new*signs
    answer=torch.empty_like(a).scatter_(1,perm,a.long()^flip.long())
    literal_order=torch.rand(b,m,k,generator=generator,device=dev).argsort(2)
    result=result.gather(2,literal_order)
    clause_order=torch.rand(b,m,generator=generator,device=dev).argsort(1)
    result=result.gather(1,clause_order[:,:,None].expand(-1,-1,k))
    return result,answer

def sat_satisfies(c,a):
    v=a.bool().gather(1,(c.abs().long()-1).flatten(1)).view_as(c)
    return torch.where(c>0,v,~v).any(-1).all(-1)

def sat_batch(c,a,generator,augment=True):
    if augment: c,a=augment_sat(c,a,generator)
    b,m,k=c.shape; n=a.shape[1]; s=2*n+m+1; dev=c.device
    r=torch.randint(0,2,(b,n),generator=generator,device=dev).bool()
    x=torch.full((b,s),3,dtype=torch.long,device=dev); x[:,0]=0
    x[:,1:2*n+1]=torch.stack((r,~r),-1).flatten(1).long()+1
    y=x.clone(); y[:,1:2*n+1]=torch.stack((a.bool(),~a.bool()),-1).flatten(1).long()+1
    # Match upstream SAT encoding, including its polarity-to-node convention.
    # Positive literal i connects node 2*i, negative connects 2*i-1.
    # The output truth is decoded from node 2*i-1, as in upstream training.
    vx=torch.where(c>0,2*c,-2*c-1).flatten(1).long()
    cy=(torch.arange(m,device=dev)+2*n+1).repeat_interleave(k)[None].expand(b,-1)
    mask=torch.zeros((b,s*s),dtype=torch.bool,device=dev)
    mask.scatter_(1,vx*s+cy,True); mask.scatter_(1,cy*s+vx,True)
    mask=mask.view(b,s,s)
    ii=torch.arange(2*n+1,device=dev); mask[:,ii,ii]=True
    ii=torch.arange(n,device=dev)*2+2; mask[:,ii,ii-1]=True; mask[:,ii-1,ii]=True
    mask[:,:,0]=True; mask[:,0,:]=True
    return dict(x=x,y=y,mask=mask[:,None],clauses=c,n=n)

def sat_metrics(pred,batch):
    n=batch['n']; pos=pred[:,1:2*n+1:2]; neg=pred[:,2:2*n+1:2]
    fmt=((pos==1)|(pos==2)).all(1)&(neg==3-pos).all(1)&(pred[:,0]==0)
    valid=sat_satisfies(batch['clauses'],pos==2)&((pos==1)|(pos==2)).all(1)
    exact=(pred[:,:2*n+1]==batch['y'][:,:2*n+1]).all(1)
    return valid,exact

def load_arrays(task,split,root=None):
    directory=Path(root) if root is not None else HERE/'data'
    with np.load(directory/f'{task}_{split}.npz',allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

def batch(task,arrays,ids,seed,augment=True,device='cuda'):
    gen=torch.Generator(device=device).manual_seed(seed)
    if task=='sat':
        return sat_batch(torch.as_tensor(arrays['clauses'][ids],device=device).long(),torch.as_tensor(arrays['answers'][ids],device=device).long(),gen,augment)
    return {k:torch.as_tensor(arrays[k][ids],device=device).long() for k in ['x','y','ds','dg']}|{'mask':None}

def model_config(task,kind):
    common=dict(seq_len=901 if task=='maze' else 1604,vocab_size=6 if task=='maze' else 4,
        hidden_size=512,intermediate_size=2048,head_dim=64,num_layers=4 if kind=='rt' else 2,
        is_causal=False,enable_pe=True,norm_eps=1e-6,rope_theta=10000.,bptt=True,forward_dtype='bfloat16')
    if kind=='rt':common.update(cycles=7)
    else:common.update(H_cycles=2,L_cycles=6)
    if kind=='trm':common.update(dual_module=False,hh_link=True,sym_io=False)
    return common

def make_model(task,kind,device='cuda'):
    from hrm.registry import load_class
    identifier={'hrm':'hrm.models.hrm@HRM','trm':'hrm.models.ablations@HRM',
                'rt':'hrm.models.rt@RecurrentTransformer'}[kind]
    cls=load_class(identifier)
    with torch.device(device): return cls(model_config(task,kind))

if __name__=='__main__':
    torch.set_num_threads(2)
    from datasets import load_from_disk
    out=HERE/'data';out.mkdir(exist_ok=True)
    report={'seed':SEED,'sources':{},'splits':{},'checks':{}}
    vocab={c:i for i,c in enumerate('~# SGo')}
    maze={}
    for split in ['train','test']:
        path=repo_path('data/maze/data/maze-30x30-hard-1k')/(split+'.csv')
        rows=list(csv.DictReader(path.open())); report['sources'][str(path)]=digest(path)
        maze[split]={'x':np.array([[0]+[vocab[c] for c in r['question']] for r in rows],np.int16),'y':np.array([[0]+[vocab[c] for c in r['answer']] for r in rows],np.int16),'ids':np.arange(len(rows))}
    from hrm.analyse.maze.continuous_inference.metrics import geometry
    for a in maze.values():a['ds'],a['dg']=geometry(a['x'])
    p=np.random.default_rng(SEED).permutation(len(maze['test']['ids']))
    maze['val']={k:v[p[:200]] for k,v in maze['test'].items()};maze['test']={k:v[p[200:]] for k,v in maze['test'].items()}
    for s,a in maze.items():np.savez_compressed(out/f'maze_{s}.npz',**a)
    report['splits']['maze']={s:len(a['ids']) for s,a in maze.items()}
    assert not set(map(bytes,maze['train']['x'])) & set(map(bytes,maze['test']['x']))
    ds=load_from_disk(str(repo_path('data/sat/SAT_256_datasets_1k')));sat={}
    for s,d in ds.items():
        rows=list(d);assert all(r['assignment'] is not None and r['n']==256 and len(r['clauses'])==1091 for r in rows)
        sat[s]={'clauses':np.array([r['clauses'] for r in rows],np.int16),'answers':np.array([r['assignment'] for r in rows],np.int8),'ids':np.array([r['id'] for r in rows])}
    # Preserve the upstream held-out set, deduplicate it, and remove all matching
    # formulas from training before creating validation. Never move test into train.
    def key(row):return tuple(sorted(tuple(sorted(c)) for c in row.tolist()))
    test_seen=set(); keep=[]
    for i,row in enumerate(sat['test']['clauses']):
        k=key(row)
        if k not in test_seen:keep.append(i);test_seen.add(k)
    sat['test']={k:v[keep] for k,v in sat['test'].items()}
    train_seen=set();keep=[]
    for i,row in enumerate(sat['train']['clauses']):
        k=key(row)
        if k not in test_seen and k not in train_seen:keep.append(i);train_seen.add(k)
    sat['train']={k:v[keep] for k,v in sat['train'].items()}
    p=np.random.default_rng(SEED).permutation(len(sat['train']['ids']))
    sat['val']={k:v[p[:128]] for k,v in sat['train'].items()};sat['train']={k:v[p[128:]] for k,v in sat['train'].items()}
    canonical={}
    for s,a in sat.items():
        canonical[s]={tuple(sorted(tuple(sorted(cl)) for cl in row.tolist())) for row in a['clauses']}
        assert len(canonical[s])==len(a['ids'])
        assert sat_satisfies(torch.tensor(a['clauses']).long(),torch.tensor(a['answers'])).all()
        np.savez_compressed(out/f'sat_{s}.npz',**a)
    for s,t in [('train','val'),('train','test'),('val','test')]:assert not canonical[s]&canonical[t]
    report['splits']['sat']={s:len(a['ids']) for s,a in sat.items()}
    report['checks'].update(sat_all_targets_satisfy=True,sat_no_order_invariant_duplicates=True,maze_train_test_disjoint=True)
    for f in out.glob('*.npz'):report['sources'][str(f)]=digest(f)
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['splits']))
