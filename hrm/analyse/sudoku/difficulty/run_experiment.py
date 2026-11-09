"""Train a prepared Easy configuration with the shared distributed launcher."""
import argparse
from pathlib import Path
import subprocess
from hrm.paths import repository_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='Directory created by the preparation command')
    a=p.parse_args()
    config=a.output.resolve()/'resolved_config.json'
    if not config.is_file():raise FileNotFoundError(f'Prepare this run first: {config}')
    subprocess.run(['bash',str(repository_root()/'scripts/distributed.sh'),'-m','hrm','train',
                    '--config',str(config),'--output',str(a.output.resolve())],check=True,cwd=repository_root())

if __name__=='__main__':main()
