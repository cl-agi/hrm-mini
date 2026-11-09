"""Freeze public SAT inputs and analysis settings before preflight and execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from hrm import hub
from hrm.paths import repository_root
from hrm.training.data import sat_directory, freeze_training_seed
from hrm.analyse.sat.evaluate import load_checkpoint
from . import study


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--data-root',type=Path);p.add_argument('--limit',type=int)
    p.add_argument('--cycles',type=int,default=64)
    a=p.parse_args(argv)
    if a.cycles<3 or (a.limit is not None and a.limit<2):p.error('At least two cases and three inference cycles required')
    if study.PROTOCOL.exists():raise FileExistsError(f'Prepared protocol exists: {study.PROTOCOL}')
    cp=hub.checkpoint('sat',a.checkpoint)
    config=json.loads(a.config.read_text()) if a.config else None
    model,config,_=load_checkpoint(cp,config);del model
    source=sat_directory(a.data_root)
    root=study.PROTOCOL.parent;root.mkdir(parents=True,exist_ok=True)
    seed=freeze_training_seed(root)
    protocol=json.loads((repository_root()/'exp/error_readout/configs/sat_intervention.json').read_text())
    protocol['model_config']=config
    protocol['trajectory']['feature_dim']=config['arch']['hidden_size']
    protocol['checkpoint'].update(path=str(cp),sha256=study.sha256(cp))
    data=root/'inputs';data.mkdir(exist_ok=True);digests={}
    for split in ('train','valid','test'):
        with np.load(source/'data'/f'{split}_n1024.npz') as z:arrays={k:v[:a.limit] if a.limit else v for k,v in z.items()}
        protocol['data'][split]=len(arrays['clauses'])
        path=data/f'{split}_n1024.npz';study.write_npz(path,**arrays);digests[path.name]=study.sha256(path)
    protocol['data'].update(root=str(data.resolve()),sha256=digests,intervention_count=protocol['data']['test'])
    protocol['trajectory'].update(outer_cycles=a.cycles,input_seed_base=seed)
    protocol['statistics']['bootstrap_seed']=seed
    protocol['intervention'].update(continue_through_outer=a.cycles,four_direction_seed_base=seed,random_control_seeds=[seed+i for i in range(4)])
    study.write_json(study.PROTOCOL,protocol)
    print(study.PROTOCOL)

if __name__=='__main__':main()
