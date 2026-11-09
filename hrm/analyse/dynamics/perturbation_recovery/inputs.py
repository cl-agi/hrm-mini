"""Explicit, fail-fast inputs for the archived E28 recovery protocol."""
import argparse
import json
import os
from pathlib import Path
from hrm.paths import repository_root


def arguments(argv=None, *, landscape=False):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/recurrence')))
    p.add_argument('--trajectory',type=Path)
    p.add_argument('--run-config',type=Path)
    p.add_argument('--case',type=int,required=True,help='Case ID in trajectory summary.selected; no implicit historical case')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--config',type=Path,help='Model config; requires an explicit checkpoint')
    p.add_argument('--reference',type=Path,help='Optional prior results.npz for exact prefix comparison')
    p.add_argument('--output-dir',type=Path)
    if landscape:p.add_argument('--projection',type=Path)
    a=p.parse_args(argv)
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    a.trajectory=a.trajectory or a.records/'trajectory'
    a.run_config=a.run_config or a.records/'config.json'
    a.output_dir=a.output_dir or a.records/('landscape_recovery' if landscape else 'recovery')
    if landscape:a.projection=a.projection or a.records/'global_projection.npz'
    required=[a.run_config,a.trajectory/'summary.json',a.trajectory/'initial.npz']
    required += [a.trajectory/f'H{h:03d}_{suffix}' for h in range(1,6) for suffix in ('L_bits.npy','context.npz')]
    required += [x for x in (a.checkpoint,a.config,a.reference,getattr(a,'projection',None)) if x is not None]
    for path in required:
        if not path.is_file():raise FileNotFoundError(f'Required E28 input missing: {path}. Supply existing records; no analysis inputs are downloaded or fabricated.')
    cfg=json.loads(a.run_config.read_text());summary=json.loads((a.trajectory/'summary.json').read_text())
    if not isinstance(cfg.get('seed'),int) or not 0<=cfg['seed']<2**32:raise ValueError('Frozen integer seed in [0, 2**32) required in run config')
    if not isinstance(cfg.get('steps'),int) or cfg['steps']<1:raise ValueError('Positive integer steps required')
    for key in ('measure_steps','coordinate_steps'):
        values=cfg.get(key,[])
        if not values or any(not isinstance(x,int) for x in values) or values!=sorted(set(values)) or values[0]!=0 or values[-1]>cfg['steps']:
            raise ValueError(f'{key} must be sorted unique integer steps including zero, within the horizon')
    anchors=cfg.get('anchors',[])
    if not anchors or len(set(anchors))!=len(anchors) or any(not isinstance(x,int) or not 2<=x<=4094 for x in anchors):raise ValueError('Unique anchors in [2,4094] required for central tangent')
    radii=cfg.get('radii',[])
    import math
    if not radii or len(set(radii))!=len(radii) or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in radii):raise ValueError('Unique finite positive radii required')
    cases=[row['case'] for row in summary['selected']]
    if len(set(cases))!=len(cases) or a.case not in cases:raise ValueError('Case must occur exactly once in trajectory summary.selected')
    if not summary.get('model',{}).get('checkpoint_sha256'):raise ValueError('Trajectory checkpoint SHA256 required')
    if a.output_dir.exists():raise FileExistsError(f'Use a new output directory: {a.output_dir}')
    return a,cfg,summary,cases.index(a.case)


def load_model(a, summary):
    import torch
    from hrm import hub
    from hrm.paths import checkpoint_config
    from hrm.analyse.dynamics.jacobian.models import HRM,load_config,analysis_architecture,unwrap_state_dict
    from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
    cp=hub.checkpoint('sudoku',a.checkpoint)
    if sha(cp)!=summary['model']['checkpoint_sha256']:raise ValueError('Checkpoint does not match the supplied trajectory')
    cfg=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(cfg)
    name=arch.pop('name','')
    if name not in ('hrm@HRM','hrm.models.hrm@HRM') or arch.get('is_mlp_mixer',False) or arch.get('rope_steps') is not None or arch.get('hh_link',True) is not True or arch.get('sym_io',False):raise ValueError('Standard stationary full-link attention HRM required')
    if (arch.get('seq_len'),arch.get('vocab_size'),arch.get('hidden_size'),arch.get('forward_dtype'))!=(82,10,512,'bfloat16'):raise ValueError('Archived E28 protocol requires 82-token, 512-dimensional BF16 Sudoku HRM')
    model=HRM(arch).to('cuda').eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    return model,dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),config=cfg)
