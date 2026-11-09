"""Puzzle-disjoint old-H reconstruction probes; frozen protocol in experiment/."""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
import torch
from torch import nn
from hrm.analyse.sudoku.core.no_hh_mediation import load_model, read_data, initial_state, low_update, commit, digest
STEPS=(1,8,24)
NTRAIN,NVAL,NTEST,TOKENS=1024,256,512,16

@torch.no_grad()
def extract(m, inputs, indices, positions, out, steps=(1,8,24), device='cuda', batch_size=32):
    chunks={};starttime=time.time()
    def collect(k,v,pos):
        q=torch.arange(len(v),device=v.device)[:,None]
        chunks.setdefault(k,[]).append(v[q,pos].cpu())
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size];tok=torch.tensor(inputs[ix],device=device);pos=torch.tensor(positions[start:start+batch_size],device=device)
        x=m.embed(tok);h,l=initial_state(m,tok)
        collect('x',x,pos)
        for step in range(max(steps)+1):
            nl=low_update(m,h,l,x);nh=commit(m,h,nl)
            if step in steps:
                for key,v in [('h',h),('l',l),('next_l',nl),('next_h',nh),('bare_next_h',m.H_level(nl)),('bare_old_h',m.H_level(l))]:
                    collect(f'{step}/{key}',v,pos)
            h,l=nh,nl
        if start%256==0:print('extract',start+len(ix),'seconds',round(time.time()-starttime,1),flush=True)
    saved={k:torch.cat(v) for k,v in chunks.items()}
    saved['positions']=torch.tensor(positions-1)
    saved['is_blank']=torch.tensor(inputs[indices[:,None],positions]==0)
    saved['indices']=torch.tensor(indices)
    torch.save(saved,out/'features.pt')
    return saved


def position_mean(v,pos):
    mean=torch.zeros(81,v.shape[-1],device=v.device,dtype=v.dtype)
    mean.index_add_(0,pos,v)
    count=torch.bincount(pos,minlength=81).clamp_min(1).to(v.dtype)
    return mean/count[:,None]


def standardize(x,pos,ntrain):
    mu=position_mean(x[:ntrain],pos[:ntrain]);center=x-mu[pos]
    scale=center[:ntrain].square().mean(0).sqrt().clamp_min(1e-3)
    return center/scale,mu,scale


def ridge(x,y,ntrain,nval,validation_y=None):
    a=x[:ntrain].double();b=y[:ntrain].double()
    gram=a.T@a/ntrain;cross=a.T@b/ntrain
    eig,vec=torch.linalg.eigh(gram);eig=eig.clamp_min(0)
    factor=float(gram.diag().mean());rhs=vec.T@cross
    target=y[ntrain:ntrain+nval] if validation_y is None else validation_y
    best=None;history=[]
    for alpha in (1e-6,1e-5,1e-4,1e-3,.01,.1,1.,10.):
        w=(vec@(rhs/(eig[:,None]+max(alpha*factor,1e-12)))).float()
        mse=float((x[ntrain:ntrain+nval]@w-target).square().mean())
        history.append(dict(alpha=alpha,validation_mse=mse))
        if best is None or mse<best[0]:best=(mse,w,alpha)
    return best[1],dict(alpha=best[2],validation_mse=best[0],grid=history)


def shuffle_within_position(y,pos,seed):
    g=torch.Generator(device=y.device).manual_seed(seed);perm=torch.arange(len(y),device=y.device)
    for p in range(81):
        ix=torch.where(pos==p)[0];perm[ix]=ix[torch.randperm(len(ix),generator=g,device=y.device)]
    return y[perm]


def fit_mlp(x,y,w,ntrain,nval,seed,epochs=30):
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(x.shape[-1],256),nn.GELU(),nn.Linear(256,y.shape[-1])).to(x.device)
    nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    base=x@w
    with torch.no_grad():bestloss=float((base[ntrain:ntrain+nval]-y[ntrain:ntrain+nval]).square().mean())
    best=copy.deepcopy(net.state_dict());bestep=0;history=[dict(epoch=0,validation_mse=bestloss)]
    g=torch.Generator(device=x.device).manual_seed(seed+1)
    for epoch in range(1,epochs+1):
        net.train();perm=torch.randperm(ntrain,device=x.device,generator=g)
        for start in range(0,ntrain,2048):
            ix=perm[start:start+2048];pred=base[ix]+net(x[ix]);loss=(pred-y[ix]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():v=float((base[ntrain:ntrain+nval]+net(x[ntrain:ntrain+nval])-y[ntrain:ntrain+nval]).square().mean())
        history.append(dict(epoch=epoch,validation_mse=v))
        if v<bestloss:bestloss=v;best=copy.deepcopy(net.state_dict());bestep=epoch
    net.load_state_dict(best)
    with torch.no_grad():pred=base[ntrain+nval:]+net(x[ntrain+nval:])
    return pred,{k:v.cpu() for k,v in best.items()},dict(epoch=bestep,validation_mse=bestloss,history=history)


def metrics(pred,y,mu,blank,ntest,tokens,seed,repeats=1000):
    err=(pred-y).square().mean(-1);den=(y-mu).square().mean(-1);energy=y.square().mean(-1)
    absolute_error=(pred-y).abs().mean(-1);target_abs=y.abs().mean(-1)
    cosine=torch.nn.functional.cosine_similarity(pred,y,dim=-1)
    centered=torch.nn.functional.cosine_similarity(pred-mu,y-mu,dim=-1)
    vals={k:v.reshape(ntest,tokens).mean(-1).cpu().numpy() for k,v in dict(mse=err,baseline_mse=den,target_energy=energy,mae=absolute_error,target_abs=target_abs,cosine=cosine,centered_cosine=centered).items()}
    ratio=lambda numerator,denominator:float(numerator/denominator) if denominator>0 else None
    r2=lambda error,baseline:1-float(error/baseline) if baseline>0 else None
    s=dict(mse=float(err.mean()),baseline_mse=float(den.mean()),r2=r2(err.sum(),den.sum()),relative_energy_error=ratio(err.sum(),energy.sum()),cosine=float(cosine.mean()),centered_cosine=float(centered.mean()))
    s.update(mae=float(absolute_error.mean()),target_abs=float(target_abs.mean()),relative_l1=ratio(absolute_error.sum(),target_abs.sum()),relative_l2=float((err.sum()/energy.sum()).sqrt()) if energy.sum()>0 else None)
    for label,mask in [('blank',blank),('given',~blank)]:
        s[label+'_r2']=r2(err[mask].sum(),den[mask].sum())
        s[label+'_cells']=int(mask.sum())
    rng=np.random.default_rng(seed);ix=rng.integers(ntest,size=(repeats,ntest))
    denominator=vals['baseline_mse'][ix].mean(1);valid=denominator>0
    cis=1-vals['mse'][ix].mean(1)[valid]/denominator[valid]
    s['conditional_puzzle_r2_ci95']=np.quantile(cis,[.025,.975]).tolist() if len(cis) else None
    s['defined_bootstrap_replicates']=int(valid.sum())
    return s,vals


def fit(name,out,features,counts=(1024,256,512),tokens=16,steps=(1,8,24),device='cuda',seed=None,epochs=30,repeats=1000):
    if seed is None:raise ValueError('A frozen experiment seed is required')
    ntest=counts[2]
    pos=features['positions'].flatten().to(device);ntrain=counts[0]*tokens;nval=counts[1]*tokens;test=slice(ntrain+nval,None)
    blank=features['is_blank'].flatten().to(device)[test]
    xembed=features['x'].flatten(0,1).float().to(device)
    rows=[];perpuzzle={}
    def record(step,label,pred,y,mu,extra=None):
        stat,arrays=metrics(pred,y,mu,blank,ntest,tokens,seed,repeats)
        rows.append(dict(case=name,step=step,probe=label,**stat,**(extra or {})))
        for k,v in arrays.items():perpuzzle[f'{step}/{label}/{k}']=v
        print(name,step,label,'R2',round(stat['r2'],5) if stat['r2'] is not None else None,flush=True)
    for step_index,step in enumerate(steps):
        target=features[f'{step}/h'].flatten(0,1).float().to(device)
        ymean=position_mean(target[:ntrain],pos[:ntrain]);ymu=ymean[pos];y=target-ymu
        record(step,'position_mean',ymu[test],target[test],ymu[test])
        for key in ('next_h','next_l','bare_next_h','bare_old_h'):
            pred=features[f'{step}/{key}'].flatten(0,1).float().to(device)[test]
            record(step,'raw_'+key,pred,target[test],ymu[test])
        for j,input_name in enumerate(('next_l','old_l','x','old_l_x')):
            if input_name=='x':raw=xembed
            elif input_name=='old_l_x':raw=torch.cat((features[f'{step}/l'].flatten(0,1).float().to(device),xembed),-1)
            else:raw=features[f'{step}/'+('l' if input_name=='old_l' else 'next_l')].flatten(0,1).float().to(device)
            xx,xmean,xscale=standardize(raw,pos,ntrain)
            w,info=ridge(xx,y,ntrain,nval)
            with torch.no_grad():pred=xx[test]@w+ymu[test]
            record(step,input_name+'_linear',pred,target[test],ymu[test],dict(selection=info))
            mlp_seed=seed+100+step_index*10+j
            mp,weights,mi=fit_mlp(xx,y,w,ntrain,nval,mlp_seed,epochs)
            record(step,input_name+'_mlp',mp+ymu[test],target[test],ymu[test],dict(selection=mi))
            state=dict(mlp_seed=mlp_seed,sampling_seed=mlp_seed+1,ridge=w.cpu(),xmean=xmean.cpu(),xscale=xscale.cpu(),ymean=ymean.cpu(),mlp=weights,linear_selection=info,mlp_selection=mi)
            torch.save(state,out/f'probe_{step}_{input_name}.pt')
            if input_name=='next_l':
                sh=y.clone();sh[:ntrain]=shuffle_within_position(y[:ntrain],pos[:ntrain],seed+10000)
                sh[ntrain:ntrain+nval]=shuffle_within_position(y[ntrain:ntrain+nval],pos[ntrain:ntrain+nval],seed+10001)
                sw,si=ridge(xx,sh,ntrain,nval)
                torch.save(dict(ridge=sw.cpu(),xmean=xmean.cpu(),xscale=xscale.cpu(),ymean=ymean.cpu(),linear_selection=si,shuffle_train_seed=seed+10000,shuffle_validation_seed=seed+10001),out/f'probe_{step}_next_l_shuffled.pt')
                record(step,'next_l_shuffled_linear',xx[test]@sw+ymu[test],target[test],ymu[test],dict(selection=si))
        (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(out/'per_puzzle.npz',**perpuzzle)
    return rows


if __name__=='__main__':
    from hrm.analyse.dynamics.old_h_recovery.run import main
    raise SystemExit(main())
