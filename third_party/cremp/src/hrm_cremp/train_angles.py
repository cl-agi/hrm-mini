"""Bounded small-data denoiser training. No reference conformer reaches generation."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from hrm_cremp.angle_data import AngleDataset, sha
from hrm_cremp.cyclic_augmentation import CyclicTrainingDataset
from hrm_cremp.resume_state import ReplayableLoader, capture, restore
from hrm_cremp.angle_diffusion import noised_dataset, noise_loss, generate
from hrm_cremp.angle_model import configuration, make_model, budget


def synchronized_time(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train','--dataset-path');p.add_argument('--val');p.add_argument('--output',required=True)
    p.add_argument('--resume',help='Own complete state checkpoint; resume into a new output directory')
    p.add_argument('--cyclic-augment',action='store_true')
    p.add_argument('--means-cache',help='Frozen training subset used for shared centering and projection statistics')
    p.add_argument('--family',choices=['hrm','rt','plain','ringer'],required=True)
    p.add_argument('--width',type=int,default=384);p.add_argument('--intermediate',type=int,default=512)
    p.add_argument('--heads',type=int,default=12);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--warmup',type=int,default=100);p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--seed',type=int);p.add_argument('--device',default='cpu')
    p.add_argument('--max-seconds',type=float,default=3600);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--generate-k',type=int,default=2);p.add_argument('--generate-molecules',type=int,default=4)
    p.add_argument('--no-final-generation',action='store_true')
    a=p.parse_args();
    from hrm.hub import cremp_data
    a.train=str(cremp_data(a.train,split='train'))
    a.val=str(cremp_data(a.val,split='validation'))
    if a.seed is None:
        if a.resume:
            a.seed=int(torch.load(a.resume,map_location='cpu',weights_only=False)['training_args']['seed'])
        else:
            from hrm.training.data import freeze_training_seed
            a.seed=freeze_training_seed(a.output)
    torch.set_num_threads(1); torch.manual_seed(a.seed);np.random.seed(a.seed % 2**32)
    out=Path(a.output)
    if (out/'manifest.json').exists():
        raise ValueError('Refusing to overwrite an existing training run')
    out.mkdir(parents=True,exist_ok=True)
    if a.cyclic_augment and a.workers != 0:raise ValueError('Paired augmentation requires workers=0')
    train=CyclicTrainingDataset(a.train,a.seed+4000) if a.cyclic_augment else AngleDataset(a.train)
    mean_data=AngleDataset(a.means_cache) if a.means_cache else train
    if not set(mean_data.ids).issubset(set(train.ids)):raise ValueError('Means source must be a training subset')
    means=mean_data.training_means();train.means=means
    geometry_source_sha=sha(a.means_cache or a.train)
    geometry_path=Path(a.means_cache or a.train).with_suffix('.geometry.json')
    geometry=None
    if geometry_path.exists():
        from hrm.hub import cremp_geometry
        geometry=cremp_geometry(geometry_path)
        if geometry['source_cache_sha256']!=geometry_source_sha:raise ValueError('Training geometry/cache hash mismatch')
        (out/'train_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    val=AngleDataset(a.val,means) if a.val else None
    if val is not None and set(train.ids)&set(val.ids):
        raise ValueError('Train/validation molecule overlap')
    if a.dtype!='float32':
        raise ValueError('Initial comparison requires float32 for every arm; mixed precision is not validated')
    c=configuration(a.family,a.width,a.intermediate,a.heads,train.arrays['graph_features'].shape[-1],a.dtype)
    model=make_model(c).to(a.device)
    # Architecture initialization must not shift the paired training corruption stream.
    torch.manual_seed(a.seed+3000)
    corruption_rng=torch.Generator().manual_seed(a.seed+2000)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.)
    loader=DataLoader(noised_dataset(train),batch_size=a.batch,shuffle=True,num_workers=a.workers,
                      generator=torch.Generator().manual_seed(a.seed+1000),drop_last=False)
    protocol=dict(diffusion='official cosine20 wrapped standard Gaussian',noise_scale=1.,
                  loss='channel means then mean of wrapped smooth L1; beta=pi/10',
                  use_feat_mask=False,mask_noise=False,gradient_clip=1.,weight_decay=0.,
                  corruption_seed=a.seed+2000,corruption_rng_independent_of_model=True,
                  cyclic_relabeling=a.cyclic_augment,augmentation_seed=a.seed+4000 if a.cyclic_augment else None,
                  augmentation='Independent Python RNG; uniform whole-residue cyclic offset; joint angles/features/masks; direction retained',
                  optimizer='AdamW',lr_schedule='linear warmup then linear decay by optimizer step',
                  mean_source='explicit frozen training subset' if a.means_cache else 'train selected conformers only',selection='validation denoising loss',
                  statistical_units='molecules; conformers within each molecule are correlated')
    manifest=dict(args=vars(a),model_config=c,budget=budget(model),protocol=protocol,
                  train_sha256=sha(a.train),val_sha256=sha(a.val) if a.val else None,
                  train_molecules=len(train.ids),train_conformers=len(train),
                  means_source_cache_sha256=geometry_source_sha,geometry_source_cache_sha256=geometry_source_sha,
                  train_means=means.tolist(),source_files_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    stream=ReplayableLoader(loader);records=[];best=float('inf');start=synchronized_time(a.device)
    model.train(); step=0; samples_seen=0
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if 'resume_state' not in saved:raise ValueError('Checkpoint lacks full resume state')
        assert saved['config']==c and saved['train_cache_sha256']==manifest['train_sha256']
        assert saved['geometry_source_cache_sha256']==geometry_source_sha
        assert saved['source_files_sha256']==manifest['source_files_sha256']
        excluded={'output','resume','max_seconds','device'}
        assert {k:v for k,v in saved['training_args'].items() if k not in excluded}=={k:v for k,v in vars(a).items() if k not in excluded}
        model.load_state_dict(saved['model']);step=saved['step'];samples_seen=saved['samples_seen']
        records=saved['records'];best=saved['best_validation_loss']
        restore(saved['resume_state'],opt,corruption_rng,stream,train)
    def payload():
        return dict(model=model.state_dict(),config=c,means=means,step=step,
                    train_cache_sha256=manifest['train_sha256'],geometry_source_cache_sha256=geometry_source_sha,geometry=geometry,
                    resume_state=capture(opt,corruption_rng,stream,train),samples_seen=samples_seen,
                    records=records,best_validation_loss=best,training_args=vars(a),
                    source_files_sha256=manifest['source_files_sha256'])
    for step in range(step+1,a.steps+1):
        if synchronized_time(a.device)-start>=a.max_seconds:
            step-=1;break
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(corruption_rng.get_state())
            batch=stream.next()
            corruption_rng.set_state(torch.get_rng_state())
        batch={k:v.to(a.device) for k,v in batch.items()}
        corruption_hash=None
        if step==1 or step%a.eval_every==0:
            corruption_hash=hashlib.sha256(batch['known_noise'].detach().cpu().numpy().tobytes()+
                                          batch['t'].detach().cpu().numpy().tobytes()).hexdigest()
        lrscale=min(step/max(a.warmup,1),max(0.,(a.steps-step)/max(a.steps-a.warmup,1)))
        for group in opt.param_groups:group['lr']=a.lr*lrscale
        opt.zero_grad(set_to_none=True)
        pred=model(batch['corrupted'],batch['t'],attention_mask=batch['attn_mask'],atom_features=batch['atom_features'])
        loss=noise_loss(pred,batch)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        samples_seen+=len(pred)
        row=dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),lr=opt.param_groups[0]['lr'],
                 elapsed_seconds=synchronized_time(a.device)-start,
                 conformer_samples=samples_seen,nominal_conformer_epochs=samples_seen/len(train))
        if corruption_hash is not None:
            row['corruption_sha256']=corruption_hash
            if a.cyclic_augment:row['augmentation']=train.augmentation_statistics()
        records.append(row)
        if step==1 or step%a.eval_every==0 or step==a.steps:
            if val is not None:
                # Fixed noise for comparable validation, without changing training RNG state.
                with torch.random.fork_rng(devices=[torch.device(a.device).index or 0] if str(a.device).startswith('cuda') else []):
                    torch.manual_seed(8173);model.eval();losses=[]
                    with torch.no_grad():
                        for vb in DataLoader(noised_dataset(val),batch_size=a.batch):
                            vb={k:v.to(a.device) for k,v in vb.items()}
                            vp=model(vb['corrupted'],vb['t'],attention_mask=vb['attn_mask'],atom_features=vb['atom_features'])
                            losses.append((float(noise_loss(vp,vb)),int(vb['attn_mask'].sum())))
                    row['val_loss']=sum(v*n for v,n in losses)/sum(n for _,n in losses)
                    model.train()
                if row['val_loss']<=best:
                    best=row['val_loss'];torch.save(dict(model=model.state_dict(),config=c,means=means,step=step,
                        train_cache_sha256=manifest['train_sha256'],geometry_source_cache_sha256=geometry_source_sha,geometry=geometry),out/'best.pt')
            torch.save(payload(),out/f'step_{step:06d}.pt')
            print(json.dumps(row),flush=True)
            (out/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    train_seconds=synchronized_time(a.device)-start
    torch.save(payload(),out/'last.pt')
    (out/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=dict(completed_updates=step,training_and_validation_seconds=train_seconds,
                 final_loss=records[-1]['loss'] if records else None,budget=budget(model),
                 augmentation=train.augmentation_statistics() if a.cyclic_augment else None)
    if not a.no_final_generation:
        model.eval();ds=val or train;indices=list(range(min(a.generate_molecules,len(ds.ids))))
        gen_start=synchronized_time(a.device)
        samples=generate(model,ds,indices,a.generate_k,327681,a.device)
        padded=np.full((len(samples),18,12),np.nan,np.float32)
        for i,sample in enumerate(samples):padded[i,:len(sample)]=sample
        np.savez(out/'generated_angles.npz',angles=padded,
                 ids=np.repeat(ds.ids[indices],a.generate_k),graph_smiles=np.repeat(ds.arrays['graph_smiles'][indices],a.generate_k),
                 lengths=np.repeat(ds.lengths[indices],a.generate_k),means=means,
                 train_cache_sha256=np.asarray(sha(a.train)),geometry_source_cache_sha256=np.asarray(geometry_source_sha))
        if geometry is not None:(out/'generated_angles.geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
        summary['generation_seconds']=synchronized_time(a.device)-gen_start
        summary['generated_candidates']=len(samples);summary['generated_denoiser_candidate_calls']=len(samples)*20
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
