"""Same-checkpoint R1/R2 denoising on one shared fixed-noise validation stream."""
import argparse
import json
from pathlib import Path
import time
import hrm.analyse.cremp.evaluation.bootstrap as bootstrap
import numpy as np
import torch
from torch.utils.data import DataLoader
from hrm_cremp.angle_data import AngleDataset, sha
from hrm_cremp.angle_diffusion import noised_dataset
from hrm.analyse.cremp.evaluation.outer_model import OuterDenoiser


def terms(prediction, target):
    delta=(target-prediction+torch.pi)%(2*torch.pi)-torch.pi
    absolute=delta.abs();beta=torch.pi/10
    return torch.where(absolute<beta,.5*delta.square()/beta,absolute-.5*beta)


def masks(batch):
    valid=batch['attn_mask'].bool().unsqueeze(-1).expand_as(batch['feat_mask'])
    backbone=torch.zeros_like(valid);backbone[...,:2]=valid[...,:2]
    return dict(backbone=backbone,
                existing_sidechain=valid & batch['feat_mask'].bool() & ~backbone,
                missing_sidechain=valid & ~batch['feat_mask'].bool() & ~backbone)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--device',default='cpu')
    p.add_argument('--max-batches',type=int,help='Engineering smoke only; omission evaluates the complete cache')
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise ValueError('Refusing to replace prefix evaluation')
    original_sha=sha(a.checkpoint);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=OuterDenoiser(state['config']).to(a.device);model.load_state_dict(state['model']);model.eval()
    ds=AngleDataset(a.data,state['means']);torch.manual_seed(8173)
    sums={r:np.zeros(12,np.float64) for r in (1,2)};counts=np.zeros(12,np.int64)
    groups={r:{k:dict(sum=0.,count=0) for k in ['backbone','existing_sidechain','missing_sidechain']} for r in (1,2)}
    if str(a.device).startswith('cuda'):torch.cuda.synchronize()
    start=time.perf_counter();batch_count=0
    with torch.no_grad():
        for batch in DataLoader(noised_dataset(ds),batch_size=a.batch):
            batch={k:v.to(a.device) for k,v in batch.items()};carry=None
            valid=batch['attn_mask'].bool().unsqueeze(-1).expand_as(batch['known_noise'])
            counts+=valid.sum((0,1)).cpu().numpy();first=None
            for r in (1,2):
                carry,pred=model.transition(batch['corrupted'],batch['t'],batch['attn_mask'],batch['atom_features'],carry)
                if r==1:first=pred
                elif model.carry_mode=='reset':assert torch.equal(first,pred)
                loss=terms(pred,batch['known_noise'])
                sums[r]+=(loss.double()*valid).sum((0,1)).cpu().numpy()
                for name,mask in masks(batch).items():
                    groups[r][name]['sum']+=float((loss.double()*mask).sum())
                    groups[r][name]['count']+=int(mask.sum())
            batch_count+=1
            if a.max_batches is not None and batch_count>=a.max_batches:break
    if str(a.device).startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    prefixes={}
    for r in (1,2):
        for g in groups[r].values():g['mean_per_supported_scalar']=g['sum']/g['count'] if g['count'] else None
        prefixes[r]=dict(aggregate=float(np.mean(sums[r]/counts)),channel_means=(sums[r]/counts).tolist(),groups=groups[r])
    assert sha(a.checkpoint)==original_sha
    result=dict(checkpoint_sha256=original_sha,checkpoint_step=state['step'],data_sha256=sha(a.data),
                trained_outer_rounds=model.outer_rounds,carry_mode=model.carry_mode,evaluation_prefixes=[1,2],
                noise_seed=8173,batch=a.batch,evaluated_batches=batch_count,partial=a.max_batches is not None,
                prefixes=prefixes,shared_prefix_evaluation_seconds=elapsed,
                actual_batched_transitions=2*batch_count,optimizer_updates=0,
                target_usage='Loss only; transitions accept corrupted inputs, time and graph only',
                checkpoint_unchanged=True,selection='Diagnostic only; no checkpoint selection')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
