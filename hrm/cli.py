"""Command-line entry for shared training, evaluation and analysis modules."""
import argparse, csv, importlib, json, os, runpy, sys, time
from pathlib import Path
from hrm.paths import repository_root, repo_path
from hrm.config import read_config
from hrm import hub
from hrm.registry import load_class, model_class

def main():
    p=argparse.ArgumentParser(prog='hrm')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    r=sub.add_parser('recipes');r.add_argument('--experiment');r.add_argument('--json',action='store_true')
    sat=sub.add_parser('train-sat');sat.add_argument('--config',type=Path,required=True);sat.add_argument('--data-root','--dataset-path',type=Path);sat.add_argument('--output',type=Path,required=True);sat.add_argument('--prepare-only',action='store_true')
    t=sub.add_parser('train');t.add_argument('--recipe');t.add_argument('--fresh-seed',action='store_true',help='Run one new timestamp-seeded repetition instead of archived seeds');t.add_argument('--config',type=Path);t.add_argument('--output',type=Path,required=True);t.add_argument('--experiment');t.add_argument('--dataset-path',type=Path);t.add_argument('--eval-dataset-path',type=Path);t.add_argument('--data-seed',type=int);t.add_argument('--prepare-only',action='store_true');t.add_argument('--deterministic',action='store_true',help='Require deterministic PyTorch algorithms for numerical verification')
    e=sub.add_parser('evaluate')
    from hrm.evaluation.published import add_arguments
    add_arguments(e)
    s=sub.add_parser('analyse');s.add_argument('module');s.add_argument('arguments',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.command=='list':
        from hrm.experiments import experiments
        for spec in experiments():
            print(spec['id'], spec['name'])
        return
    if args.command=='recipes':
        from hrm.experiments import recipes
        entries=recipes(args.experiment)
        if args.json:print(json.dumps(entries,indent=2))
        else:
            for key,entry in entries.items():
                config=entry['config']
                print(key,config['arch']['name'],'seeds='+str(config.get('seeds')),
                      'policy='+config.get('training_policy','standard'),
                      'dataset='+entry['dataset_defaults']['variant'],entry['status'],sep='\t')
        return
    if args.command=='analyse':
        name=args.module if args.module.startswith('hrm.analyse.') else 'hrm.analyse.'+args.module
        from hrm.experiments import analysis_scope
        options=argparse.ArgumentParser(add_help=False)
        options.add_argument('--experiment');options.add_argument('--records',type=Path)
        scope, remaining=options.parse_known_args(args.arguments)
        sys.argv=[name]+remaining
        with analysis_scope(name,scope.experiment,scope.records):
            runpy.run_module(name,run_name='__main__')
        return
    if args.command=='train-sat':
        config=read_config(args.config)
        if 'seed' not in config:
            from hrm.training.data import freeze_training_seed
            config['seed']=freeze_training_seed(args.output)
        args.output.mkdir(parents=True,exist_ok=True)
        saved=args.output/'resolved_config.json'
        if saved.exists() and json.loads(saved.read_text())!=config:
            raise ValueError('SAT run configuration differs; use a new output directory')
        if int(os.environ.get('RANK','0'))==0 and not saved.exists():
            hub._write(saved,lambda path:path.write_text(json.dumps(config,indent=2)+'\n'))
        from hrm.training.data import sat_directory
        data_root=sat_directory(args.data_root,n_vars=config.get('n_vars',1024))
        if args.prepare_only:
            print(json.dumps(dict(config=str(saved),seed=config['seed'],data_root=str(data_root))))
            return
        from hrm.training.sat import train
        train(config,args.output,data_root);return
    if args.command=='train':
        defaults=None
        if args.experiment:
            from hrm.experiments import experiment
            spec=experiment(args.experiment)
            if args.recipe and args.recipe not in spec['recipes']:p.error('Recipe does not belong to this experiment')
        if args.deterministic:
            import torch
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False
        if bool(args.recipe)==bool(args.config):p.error('choose exactly one of --recipe and --config')
        if args.recipe:
            recipes=json.loads(repo_path('exp/recipes.json').read_text());recipe=recipes['recipes'][args.recipe]
            from hrm.experiments import recipe_config
            config=recipe_config(recipe,args.fresh_seed);defaults=recipe['dataset_defaults']
        else:
            config=read_config(args.config)
            if args.fresh_seed:config.pop('seeds',None)
        from hrm.training.data import resolve_config,freeze_sampling_seed
        if defaults and defaults['variant']=='easy_resampled' and args.dataset_path is None:
            defaults=dict(defaults,sampling_seed=freeze_sampling_seed(args.output,args.data_seed))
        if config.get('training_policy') == 'data_scaling' and args.data_seed is not None:
            from hrm.training.data import _freeze_seed, SUDOKU_UPSTREAM, SUDOKU_UPSTREAM_REVISION
            seed=_freeze_seed(args.output,'dataset_sampling.json',args.data_seed,
                dict(source=SUDOKU_UPSTREAM,revision=SUDOKU_UPSTREAM_REVISION,
                     policy='Shared shuffled train/test prefixes; new data draw'))
            config['data']['subsample_seed']=seed
        config=resolve_config(config,defaults,args.dataset_path,args.eval_dataset_path)
        if 'seeds' not in config:
            from hrm.training.data import freeze_training_seed
            config['seeds']=[freeze_training_seed(args.output)]
        if len(set(config['seeds']))!=len(config['seeds']):raise ValueError('independent runs require distinct seeds')
        args.output.mkdir(parents=True,exist_ok=True)
        config['output_dir']=str(args.output.resolve())
        saved=args.output/'resolved_config.json'
        if int(os.environ.get('RANK','0'))==0 and (not saved.exists() or json.loads(saved.read_text())!=config):
            hub._write(saved,lambda path:path.write_text(json.dumps(config,indent=2)+'\n'))
        if args.fresh_seed and int(os.environ.get('RANK','0'))==0:
            provenance=dict(recipe=args.recipe,config=str(args.config) if args.config else None,seeds=config['seeds'],seed_policy='New independent run; archived seeds not used',historical_seed_reproduction=False)
            hub._write(args.output/'run_provenance.json',lambda path:path.write_text(json.dumps(provenance,indent=2)+'\n'))
        if args.prepare_only:
            print(args.output/'resolved_config.json');return
        import torch
        if int(os.environ.get('RANK','0')) == 0:
            settings=dict(torch=torch.__version__,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),nccl_algorithm=os.environ.get('NCCL_ALGO'),nccl_protocol=os.environ.get('NCCL_PROTO'))
            hub._write(args.output/'execution_settings.json',lambda path:path.write_text(json.dumps(settings,indent=2)+'\n'))
        # Dataset/checkpoint locations in recipes are repository-relative.
        if 'data' in config and 'dataset_name' in config['data']:
            value=config['data']['dataset_name']
            if value.startswith(('data/','checkpoint/')):config['data']['dataset_name']=str(repo_path(value))
        if config.get('data', {}).get('prepared_data_path', '').startswith('data/'):
            config['data']['prepared_data_path']=str(repo_path(config['data']['prepared_data_path']))
        for name in ['init_checkpoint','eval_dataset_name','resume_from_checkpoint','pilot_checkpoint']:
            value=config.get(name)
            if isinstance(value,str) and value.startswith(('data/','checkpoint/')):config[name]=str(repo_path(value))
        os.environ.setdefault('WANDB_MODE','disabled')
        policy=config.pop('training_policy','standard')
        if policy not in {'standard','stochastic','fixed_steps','outer_sweep','mixer','data_scaling'}:raise ValueError('Unknown training policy: '+policy)
        module=importlib.import_module('hrm.training.'+('trainer' if policy=='standard' else policy))
        module.train(config);return
    from hrm.evaluation.published import run
    run(args)

if __name__=='__main__':main()
