from hrm.paths import work_path
from hrm.paths import repo_path
from pathlib import Path
import sys, json, argparse, hashlib
import numpy as np
import torch
from hrm.paths import repository_root
R = repository_root()/'exp/error_readout/records/sudoku_constraints/population_analysis'
pass


def count(logits):
    b = logits.argmax(-1)[..., 1:].reshape(-1, 9, 9)
    o = torch.nn.functional.one_hot(b.long(), 10)[..., 1:]
    return ((o.sum(1) - 1).clamp_min(0).sum((1, 2)) + (o.sum(2) - 1).clamp_min(0).sum((1, 2)) + (o.reshape(-1, 3, 3, 3, 3, 9).sum((2, 4)) - 1).clamp_min(0).sum((1, 2, 3))).float()

@torch.inference_mode()
def main():
    from hrm import hub
    from hrm.training.data import puzzle_directory,freeze_training_seed
    from hrm.paths import checkpoint_config
    from hrm.analyse.dynamics.jacobian.models import HRM,load_config,analysis_architecture,unwrap_state_dict
    from hrm.analyse.dynamics.jacobian.data import read_test_records,select_board_ids,build_loader
    p=argparse.ArgumentParser(description='Collect native population constraint dynamics')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output',type=Path,default=R);p.add_argument('--boards',type=int,default=1056)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if min(args.boards,args.batch_size)<1:p.error('Positive sample and batch counts required')
    root=args.output;root.mkdir(parents=True,exist_ok=True);seed=freeze_training_seed(root)
    cp=hub.checkpoint('sudoku',args.checkpoint)
    cfg=load_config(args.config or checkpoint_config(cp)) if args.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(cfg);arch.pop('name',None)
    records=read_test_records(args.dataset or puzzle_directory())
    ids=select_board_ids(len(records),boards=args.boards,offset=0,selection='random',selection_seed=seed,explicit=None)
    torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=False
    model=HRM(arch).to(args.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    ident=dict(seed=seed,checkpoint=str(cp),config=cfg,row_ids=ids)
    with (root/'config.json').open('x') as stream:json.dump(ident,stream,indent=2)
    out = root / 'population'
    out.mkdir(parents=True, exist_ok=True)
    for batch_index,(inputs,_,row_ids) in enumerate(build_loader(records,ids,args.batch_size)):
        start=batch_index*args.batch_size
        x=inputs.to(args.device)
        emb = model.embed(x)
        h = model.initial_carry['z_H'].expand_as(emb)
        l = model.initial_carry['z_L'].expand_as(emb)
        f = []
        t = []
        delta = []
        norm = []
        for _ in range(16):
            for j in range(6):
                l = model.L_level(l + h + emb)
                n = model.L_level(l + h + emb)
                f.append(torch.cat((l, h), -1).mean(1).float().cpu().numpy())
                t.append(count(model.lm_head(model.H_level(h + l))).cpu().numpy())
                delta.append((n.float() - l.float()).flatten(1).norm(dim=1).cpu().numpy())
                norm.append(l.float().flatten(1).norm(dim=1).cpu().numpy())
            h = model.H_level(h + l)
        h = model.initial_carry['z_H'].expand_as(emb)
        l = model.initial_carry['z_L'].expand_as(emb)
        e = np.full((len(x), 4, 50), np.nan)
        for branch in range(4):
            current = l
            for j in range(branch * 10, 50):
                current = model.L_level(current + h + emb)
                e[:, branch, j] = count(model.lm_head(model.H_level(h + current))).cpu().numpy()
                if j == branch * 10 + 9:
                    native = current
            l = native
            h = model.H_level(h + l)
        np.savez_compressed(out / f'batch_{start:04}.npz', features=np.stack(f, 1), target=np.stack(t, 1), delta=np.stack(delta, 1), norm=np.stack(norm, 1), freeze_count=e)
        print('DONE', start + len(x), flush=True)
    (out / 'identity.json').write_text(json.dumps(ident, indent=2))
if __name__ == '__main__':
    main()
