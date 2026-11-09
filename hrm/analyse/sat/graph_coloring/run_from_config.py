"""Run one QuerySAT shard from the supplied numerical protocol."""
import argparse,json,sys,subprocess
from pathlib import Path
from hrm.paths import repo_path,work_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=repo_path('data/sat/graph_coloring/querysat.json'));p.add_argument('--model',choices=['single_431000','batch4_172500'],required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--print-command',action='store_true');args=p.parse_args()
    c=json.loads(args.config.read_text());out=work_path('sat/graph_coloring/querysat')/args.model/f'shard{args.shard:02d}'
    command=[sys.executable,'-m','hrm.analyse.sat.graph_coloring.querysat','--config',str(args.config),'--checkpoint',str(repo_path(c['models'][args.model]['checkpoint'])),'--model-label',args.model,'--starts',str(c['starts']),'--rounds',str(c['rounds']),'--seed-base',str(c['seed_base']),'--shards',str(c['shards']),'--shard',str(args.shard),'--out',str(out)]
    if args.resume:command.append('--resume')
    if args.print_command:print(json.dumps(command));return
    subprocess.run(command,check=True)
if __name__=='__main__':main()
