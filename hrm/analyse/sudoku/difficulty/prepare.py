"""Prepare a new Easy-training run with public data and the shared trainer."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from hrm.paths import repository_root


def main(argv=None, tiny=False):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('hrm','rt'),default='hrm')
    p.add_argument('--output',type=Path)
    p.add_argument('--dataset-path',type=Path)
    p.add_argument('--data-seed',type=int)
    a=p.parse_args(argv)
    stem='sudoku-hrm-easy' if a.model=='hrm' else 'sudoku-rt-easy'
    recipe=stem+('-layers2-variant-01' if a.model=='hrm' else '-layers4-recurrent7-variant-01') if tiny else stem+('-resampled-layers2-wd0-1' if a.model=='hrm' else '-resampled-layers4-recurrent7-wd0-1')
    out=a.output or repository_root()/f'exp/generalization/records/easy{1000 if tiny else 80000}_{a.model}'
    prepared=out/'resolved_config.json'
    settings=dict(recipe=recipe,dataset_path=str(a.dataset_path.resolve()) if a.dataset_path else None,data_seed=a.data_seed)
    manifest=out/'preparation.json'
    if prepared.exists():
        if not manifest.exists() or json.loads(manifest.read_text())!=settings:raise ValueError('Prepared settings differ; use a new output directory')
        print(prepared);return
    cmd=[sys.executable,'-m','hrm','train','--recipe',recipe,'--fresh-seed','--output',str(out),'--prepare-only']
    if a.dataset_path:cmd+=['--dataset-path',str(a.dataset_path)]
    if a.data_seed is not None:cmd+=['--data-seed',str(a.data_seed)]
    subprocess.run(cmd,check=True)
    manifest.write_text(json.dumps(settings,indent=2))


if __name__=='__main__':main()
