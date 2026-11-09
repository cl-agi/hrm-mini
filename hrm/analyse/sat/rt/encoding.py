"""Frozen task encodings, semantic augmentation, and target-free validators."""
from hrm.paths import analysis_directory
from pathlib import Path
import csv, hashlib, json
import numpy as np
import torch
import torch.nn.functional as F

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

def sat_batch(c,a,generator,augment=True,delete_fraction=0.):
    if augment: c,a=augment_sat(c,a,generator)
    b,m,k=c.shape; n=a.shape[1]; s=2*n+m+1; dev=c.device
    r=torch.randint(0,2,(b,n),generator=generator,device=dev).bool()
    x=torch.full((b,s),3,dtype=torch.long,device=dev); x[:,0]=0
    x[:,1:2*n+1]=torch.stack((r,~r),-1).flatten(1).long()+1
    y=x.clone(); y[:,1:2*n+1]=torch.stack((a.bool(),~a.bool()),-1).flatten(1).long()+1
    # Direct polarity: positive literal connects positive output node.
    vx=(1+2*(c.abs()-1)+(c<0).long()).flatten(1).long()
    cy=(torch.arange(m,device=dev)+2*n+1).repeat_interleave(k)[None].expand(b,-1)
    mask=torch.zeros((b,s*s),dtype=torch.bool,device=dev)
    mask.scatter_(1,vx*s+cy,True); mask.scatter_(1,cy*s+vx,True)
    mask=mask.view(b,s,s)
    ii=torch.arange(s,device=dev); mask[:,ii,ii]=True
    ii=torch.arange(n,device=dev)*2+2; mask[:,ii,ii-1]=True; mask[:,ii-1,ii]=True
    mask[:,:,0]=True; mask[:,0,:]=True
    keep_var=torch.ones((b,n),device=dev,dtype=torch.bool)
    if augment and delete_fraction>0:
        deleted=torch.randint(0,int(n*delete_fraction)+1,(b,1),generator=generator,device=dev)
        order=torch.rand((b,n),generator=generator,device=dev).argsort(1).argsort(1)
        keep_var=order>=deleted
    keep_clause=keep_var.gather(1,(c.abs().long()-1).flatten(1)).view(b,m,k).all(-1)
    active=torch.cat((torch.ones((b,1),device=dev,dtype=torch.bool),keep_var.repeat_interleave(2,1),keep_clause),1)
    # Inactive padded nodes cannot send or receive any signal; self edges avoid empty softmax.
    mask &= active[:,:,None]
    mask &= active[:,None,:]
    ii=torch.arange(s,device=dev);mask[:,ii,ii]=True
    return dict(x=x,y=y,mask=mask[:,None],clauses=c,n=n,keep_var=keep_var,
                keep_clause=keep_clause,active=active,loss_mask=active[:,:2*n+1],
                effective_n=keep_var.sum(1),effective_m=keep_clause.sum(1))

def sat_metrics(pred,batch):
    n=batch['n']; pos=pred[:,1:2*n+1:2]; neg=pred[:,2:2*n+1:2]
    fmt=((pos==1)|(pos==2)).all(1)&(neg==3-pos).all(1)&(pred[:,0]==0)
    valid=sat_satisfies(batch['clauses'],pos==2)&((pos==1)|(pos==2)).all(1)
    exact=(pred[:,:2*n+1]==batch['y'][:,:2*n+1]).all(1)
    return valid,exact

def load_arrays(task,split):
    f=np.load(analysis_directory(__file__)/'data'/f'{task}_{split}.npz')
    return {k:f[k] for k in f.files}

def batch(task,arrays,ids,seed,augment=True,device='cuda'):
    gen=torch.Generator(device=device).manual_seed(seed)
    if task=='sat':
        return sat_batch(torch.as_tensor(arrays['clauses'][ids],device=device).long(),torch.as_tensor(arrays['answers'][ids],device=device).long(),gen,augment)
    return {k:torch.as_tensor(arrays[k][ids],device=device).long() for k in ['x','y','ds','dg']}|{'mask':None}
