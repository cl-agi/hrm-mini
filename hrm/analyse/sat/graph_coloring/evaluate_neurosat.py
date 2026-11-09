"""Evaluate E43 NeuroSAT from public weights and HF graph inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from hrm.paths import repository_root
from hrm.training.data import freeze_training_seed
from hrm_neurosat.pretrained import load_model
from hrm.analyse.sat.graph_coloring.data import load_data
from hrm.analyse.sat.graph_coloring.neurosat import make_graph,trajectory,verify_trajectory,check_source_equivalence,METRICS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--dataset-path',type=Path,help='Local published-schema Parquet; missing paths never fall back')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--colors',type=int,nargs='+',choices=(3,4))
    p.add_argument('--benchmark',choices=('generated','flat200'),default='generated')
    p.add_argument('--iterations',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--limit',type=int,help='Diagnostic subset per color count; omitted means all rows')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    args.output=args.output or repository_root()/'exp/sat/records/graph_coloring/neurosat'/args.benchmark
    args.colors=args.colors or ([3] if args.benchmark=="flat200" else [3,4])
    if args.benchmark=="flat200" and args.colors!=[3]:p.error("Flat200 uses3 colors")
    if args.iterations<1 or args.batch_size<1 or (args.limit is not None and args.limit<1) or len(set(args.colors))!=len(args.colors):p.error('Positive budgets and unique color counts required')
    args.output.mkdir(parents=True,exist_ok=True)
    seed=freeze_training_seed(args.output)
    data={k:load_data(k,args.dataset_path,args.benchmark) for k in args.colors}
    source=check_source_equivalence()
    for name in ("data.py","evaluate_neurosat.py","neurosat.py","flat_metrics.py"):
        source[name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    model,checkpoint=load_model('cpu',args.checkpoint,args.iterations)
    config=dict(benchmark=args.benchmark,seed=seed,checkpoint=checkpoint,iterations=args.iterations,batch_size=args.batch_size,
                device=args.device,limit=args.limit,source_sha256=source,
                data={str(k):dict(n=len(d['ids']),ordered_sha256=d['ordered_sha256']) for k,d in data.items()})
    manifest=args.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Run configuration changed; use a new output directory')
    manifest.write_text(json.dumps(config,indent=2)+'\n')
    if args.prepare_only:
        print(json.dumps(config,indent=2));return
    with (args.output/'started.json').open('x') as f:json.dump({'seed':seed},f)
    torch.manual_seed(seed);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device);model.to(device)
    metrics=METRICS if args.benchmark=="generated" else ("full","graph")
    results={};started=time.monotonic()
    with torch.inference_mode():
        for k,d in data.items():
            n=min(len(d['ids']),args.limit or len(d['ids']))
            final=np.zeros(len(metrics),dtype=np.int64);any_iteration=final.copy()
            for begin in range(0,n,args.batch_size):
                c=d['clauses'][begin:min(begin+args.batch_size,n)];e=d['edges'][begin:min(begin+args.batch_size,n)]
                pred,_=trajectory(model,make_graph(c,device,variables=d["variables"],flat=args.benchmark=="flat200"),args.iterations)
                predictions=pred.cpu().numpy()
                if args.benchmark=='generated':verified=verify_trajectory(predictions,c,e,k)['success']
                else:
                    from hrm.analyse.sat.graph_coloring.flat_metrics import check_boolean_graph_predictions
                    verified=check_boolean_graph_predictions(predictions,c,e)
                final+=verified[-1].sum(axis=0);any_iteration+=verified.any(axis=0).sum(axis=0)
            results[str(k)]=dict(n=n,available=len(d['ids']),case_ids=d['ids'][:n],
                final={m:dict(correct=int(final[i]),accuracy=float(final[i]/n)) for i,m in enumerate(metrics)},
                any_iteration={m:dict(correct=int(any_iteration[i]),accuracy=float(any_iteration[i]/n)) for i,m in enumerate(metrics)})
    output=dict(configuration=config,results=results,wall_seconds=time.monotonic()-started,
                scope='full published protocol' if args.limit is None and args.iterations==2048 and args.dataset_path is None and args.checkpoint is None and (set(args.colors)=={3,4} or args.benchmark=="flat200") else 'custom budget, source or subset')
    (args.output/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
