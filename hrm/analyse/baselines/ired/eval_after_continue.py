from hrm.paths import analysis_directory
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--gpu',required=True)
a=p.parse_args()
root=analysis_directory(__file__)
while True:
    status=a.directory/'status.json'
    if status.exists():
        try:
            if json.loads(status.read_text())['step']>=60000:
                break
        except json.JSONDecodeError:
            pass
    time.sleep(10)
# The status is written just before the final atomic checkpoint save.
time.sleep(10)
import torch
assert torch.load(a.directory/'latest.pt',map_location='cpu',weights_only=False)['step']==60000
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpu
cmd=[str(root/'.venv/bin/python'),'-u',str(Path(__file__).resolve().parent / 'diagnose.py'),'--checkpoint',str(a.directory/'latest.pt'),
     '--output',str(a.directory/'eval_ema20')]
raise SystemExit(subprocess.call(cmd,env=env))
