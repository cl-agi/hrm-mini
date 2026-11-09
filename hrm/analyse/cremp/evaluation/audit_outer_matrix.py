"""Read-only completed-wave audit and a nonduplicated receipt index."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
p.add_argument('--wave',type=int,choices=[1,2],required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();base=a.base;plan=read(base/'model_tools/outer_r2_execution_plan.json')
source=base/'releases/outer_r2_v1';assert sha(source/'source_manifest.json')==plan['source_manifest_sha256']
for name,value in read(source/'source_manifest.json')['files_sha256'].items():assert sha(source/name)==value
original=base/'releases/pilot_v1'
for name,value in read(original/'release_manifest.json')['files_sha256'].items():assert sha(original/name)==value
val=np.load(base/'processed/val_256_angles.npz',allow_pickle=False)
train_sha=sha(base/'processed/train_1000_angles.npz');val_sha=sha(base/'processed/val_256_angles.npz')
means=read(base/'runs/pilot_v1_hrm_seed1/manifest.json')['train_means'];report=dict(runs=[],receipts=[],noise_pairing={})
for task in plan['tasks']:
    if task['wave']>a.wave:continue
    name=task['run_id'];run=base/'runs'/name;inference=base/'inference'/name
    assert not (run/'pipeline_error.json').exists()
    complete=read(inference/'pipeline_summary.json');assert complete['completed']
    metrics=read(run/'metrics.json');manifest=read(run/'manifest.json');summary=read(run/'summary.json')
    assert len(metrics)==summary['completed_updates']==20000
    assert summary['completed_episodes']==10000 and summary['fresh_conformer_samples']==639601
    assert manifest['train_sha256']==train_sha and manifest['val_sha256']==val_sha
    assert manifest['train_means']==means
    assert manifest['args']['seed']==task['seed'] and manifest['args']['family']==task['family']
    assert manifest['args']['carry_mode']==task['carry_mode'] and manifest['args']['outer_rounds']==2
    assert manifest['budget']['parameters']==3963008
    assert manifest['budget']['gradient_blocks_per_training_update']==28
    assert manifest['budget']['forward_blocks_per_generated_candidate']==1120
    hashes={r['step']:r['corruption_sha256'] for r in metrics if 'corruption_sha256'in r}
    assert sorted(hashes)==[1]+list(range(1000,20001,1000))
    seed=str(task['seed'])
    if seed in report['noise_pairing']:assert report['noise_pairing'][seed]['hashes']==hashes
    else:report['noise_pairing'][seed]=dict(hashes=hashes,arms=[])
    report['noise_pairing'][seed]['arms'].append(name)
    prefix=read(inference/'prefix_losses.json');gen=read(inference/'step20000/manifest.json')
    assert prefix['trained_outer_rounds']==gen['trained_outer_rounds']==gen['evaluation_outer_rounds']==2
    assert abs(prefix['prefixes']['2']['aggregate']-metrics[-1]['val_loss'])<1e-6
    assert not gen['reference_angles_used'] and gen['actual_batched_denoiser_calls']==80
    assert gen['args']['seed']==327681 and gen['args']['batch_molecules']==16
    checkpoint=run/'step_020000.pt'
    assert sha(checkpoint)==gen['checkpoint_sha256']==prefix['checkpoint_sha256']
    with np.load(inference/'step20000/generated_angles.npz',allow_pickle=False) as samples:
        assert np.array_equal(samples['ids'],np.repeat(val['ids'][:32],64))
        assert np.array_equal(samples['graph_smiles'],np.repeat(val['graph_smiles'][:32],64))
        assert np.array_equal(samples['lengths'],np.repeat(val['lengths'][:32],64))
        assert np.array_equal(samples['candidate_index'],np.tile(np.arange(64),32))
        assert all(np.isfinite(x[:int(n)]).all() for x,n in zip(samples['angles'],samples['lengths']))
    for r in complete['receipts']:
        path=Path(r['path']);assert sha(path)==r['sha256']
        receipt=read(path);assert receipt['exit_code']==0
        report['receipts'].append(dict(path=str(path),sha256=sha(path),gpu_seconds=receipt['gpu_seconds_conservative']))
    report['runs'].append(dict(run_id=name,family=task['family'],carry_mode=task['carry_mode'],seed=task['seed'],
         prefix1_loss=prefix['prefixes']['1']['aggregate'],prefix2_loss=prefix['prefixes']['2']['aggregate'],
         prefix_groups={r:{k:g['mean_per_supported_scalar'] for k,g in prefix['prefixes'][r]['groups'].items()} for r in ['1','2']},
         checkpoint_sha256=gen['checkpoint_sha256'],generation_directory=str(inference/'step20000')))
for x in report['noise_pairing'].values():assert len(x['arms'])==4
assert len({r['path'] for r in report['receipts']})==len(report['receipts'])
report.update(completed_waves=a.wave,source_manifest_sha256=plan['source_manifest_sha256'],
              frozen_outer_files_verified=11,frozen_A_files_verified=69,test_accessed=False,
              total_gpu_seconds_conservative=sum(r['gpu_seconds'] for r in report['receipts']))
report['total_gpu_hours_conservative']=report['total_gpu_seconds_conservative']/3600
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('noise_pairing','receipts')},indent=2))
