"""Prepare a frozen Flat evaluation in a separate output directory."""
import argparse,json
from pathlib import Path
from hrm.paths import repository_root

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['hrm','rt'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=repository_root()
    c=json.loads((root/f'data/sat/flat200/{a.model}.json').read_text())
    for key in ['checkpoint','data_root','data_file','data_manifest','training_config']:
        c[key]=str(root/c[key])
    c['original_cnf_files']=[str(root/v) for v in c['original_cnf_files']]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'config.json').write_text(json.dumps(c,indent=2))
    print(a.output.resolve())

if __name__=='__main__':main()
