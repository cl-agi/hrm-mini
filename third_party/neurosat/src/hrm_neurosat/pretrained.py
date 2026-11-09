"""Verified RandCSPBench pretrained NeuroSAT weights used by E43."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import torch
from hrm import hub
from hrm_neurosat.gnn import GNN

URL='https://drive.usercontent.google.com/download?id=1qnq0-vNoAbJpiAw6wgxcdllLabUMaNub&export=download&confirm=t'
SHA256='6d90914da4d44c64c790afd3bfac7a19c6b1d9fd3e6860b334150714f656af99'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def checkpoint(path=None):
    if path is not None:return hub.local_file(path)
    target=hub.cache_path('sat',f'neurosat/{SHA256}/model.ckpt')
    if not target.exists():
        def download(tmp):
            with urllib.request.urlopen(URL,timeout=60) as source,tmp.open('wb') as destination:
                shutil.copyfileobj(source,destination)
            if digest(tmp)!=SHA256:raise ValueError('Public NeuroSAT checkpoint hash mismatch')
        hub._write(target,download)
    if digest(target)!=SHA256:raise ValueError('Cached NeuroSAT checkpoint changed')
    return target


def load_model(device='cpu',path=None,iterations=2048):
    if iterations<1:raise ValueError('iterations must be positive')
    source=checkpoint(path)
    state=torch.load(source,map_location='cpu',weights_only=False)
    opts=state['hyper_parameters']['args']
    required=dict(model='neurosat',graph='lcg',init_emb='learned',task='assignment')
    if any(getattr(opts,key,None)!=value for key,value in required.items()) or getattr(opts,'decoding','standard')!='standard':
        raise ValueError('Checkpoint is not the supported assignment NeuroSAT architecture')
    opts.n_iterations=iterations
    model=GNN(opts)
    model.load_state_dict({key.removeprefix('model.'):value for key,value in state['state_dict'].items()},strict=True)
    model.eval().requires_grad_(False).to(device)
    info=dict(checkpoint=str(source),checkpoint_sha256=digest(source),epoch=int(state['epoch']),
              global_step=int(state['global_step']),iterations=iterations,parameters=sum(p.numel() for p in model.parameters()))
    return model,info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--iterations',type=int,default=2048)
    args=p.parse_args()
    _,info=load_model(args.device,args.checkpoint,args.iterations)
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
