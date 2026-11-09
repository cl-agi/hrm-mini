"""Audit fresh same-puzzle SA, DFS and HRM records without historical archives."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from hrm.paths import repository_root
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .run_sa import check_puzzle
from .run_dfs import solve as dfs_solve


def require(condition,message):
    if not condition:raise ValueError(message)


def equal(a,b,label):
    require(np.array_equal(a,b),'Mismatch: '+label)


def load(root):
    plan=json.loads((root/'run_config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    require(manifest.get('status')=='complete','Incomplete records: '+str(root))
    require(manifest['plan_sha256']==sha(root/'run_config.json'),'Changed run plan')
    for name,digest in manifest['artifacts'].items():require(sha(root/name)==digest,'Changed artifact: '+name)
    return plan,manifest


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def solved(board,puzzle):
    grid=board.reshape(9,9);goal=set(range(1,10))
    return bool(np.all(board[puzzle!=0]==puzzle[puzzle!=0]) and all(set(u)==goal for u in [*grid,*grid.T,*[grid[r:r+3,c:c+3].flatten() for r in (0,3,6) for c in (0,3,6)]]))


def indices(times,start=0):
    values=times[start:];return np.array([start+np.argmin(abs(values-t)) for t in np.linspace(values[0],values[-1],99)])


def board_distance(boards,ix):return (boards[ix,None,:]!=boards[None,ix,:]).sum(-1)/81


def check_distance(actual,expected):
    require(actual.shape==(99,99) and np.isfinite(actual).all() and np.allclose(actual,expected,atol=1e-7,rtol=1e-6),'Distance reconstruction differs')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/search'))
    p.add_argument('--records',type=Path,default=root);p.add_argument('--sa-records',type=Path);p.add_argument('--dfs-records',type=Path);p.add_argument('--hrm-records',type=Path);p.add_argument('--output-dir',type=Path)
    a=p.parse_args(argv);out=a.output_dir or a.records/'comparison'
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    sources={'sa':a.sa_records or a.records/'sa','dfs':a.dfs_records or a.records/'dfs','hrm':a.hrm_records or a.records/'hrm'}
    loaded={k:load(path) for k,path in sources.items()};reference=loaded['sa'][0]
    for plan,_ in loaded.values():
        require(all(plan[k]==reference[k] for k in ('puzzle','target','board_id','dataset_sha256')),'Methods use different puzzles or data sources')
    puzzle=np.array([int(c) if c!='.' else 0 for c in reference['puzzle']]);target=np.array(list(map(int,reference['target'])))
    check_puzzle(puzzle);require(solved(target,puzzle),'Reference answer is not a valid completion')
    results={}
    plan,m=loaded['sa'];require(len(plan['seeds'])==3 and len(set(plan['seeds']))==3,'Three distinct SA seeds required')
    require([r['seed'] for r in m['runs']]==plan['seeds'],'Missing, duplicate or reordered SA runs')
    require(set(m['artifacts'])=={f'seed_{seed}.npz' for seed in plan['seeds']},'Unexpected SA artifacts')
    for row in m['runs']:
        d=arrays(sources['sa']/f"seed_{row['seed']}.npz");b=d['boards'];t=d['trace']
        equal(d['puzzle'],puzzle,'SA puzzle');equal(d['target'],target,'SA target');equal(d['final'],b[-1],'SA final board')
        require(b.ndim==2 and b.shape[1]==81 and t.shape==(len(b),5) and np.isfinite(t).all(),'Malformed SA trace')
        require(np.all((b>=1)&(b<=9)) and np.all(b[:,puzzle!=0]==puzzle[puzzle!=0]),'SA changed clues or invalid digit')
        grids=b.reshape(-1,9,9)
        for r in (0,3,6):
            for c in (0,3,6):equal(np.sort(grids[:,r:r+3,c:c+3].reshape(-1,9)),np.tile(np.arange(1,10),(len(b),1)),'SA box invariant')
        costs=np.array([sum(9-len(set(u)) for u in [*g,*g.T]) for g in grids]);equal(costs,t[:,1],'SA recorded energy')
        require(t[0,0]==0 and np.all(np.diff(t[:,0])>=0) and int(t[-1,0])==row['steps']<=plan['budget'],'SA proposal counts')
        equal(t[:,4],(np.diff(t[:,3],prepend=0)>0).astype(float),'SA restart events')
        require(int(t[-1,3])==row['reheats'] and row['uphill_accepted']<=row['accepted']<=row['steps'],'SA counters')
        ok=solved(b[-1],puzzle);require(row['solved']==ok and row['window_ends_at_solution']==ok and row['final_energy']==costs[-1],'SA terminal status')
        require(ok or row['steps']==plan['budget'],'Unsolved SA stopped before budget')
        require(row['matches_target']==bool(np.array_equal(b[-1],target)),'SA target comparison')
        starts=np.flatnonzero(t[:,4]==1);start=int(starts[-1]) if len(starts) else 0;ix=indices(t[:,0],start)
        equal(d['sampled_indices'],ix,'SA sampling');equal(d['sampled_proposals'],t[ix,0],'SA sampled proposals');check_distance(d['distance'],board_distance(b,ix))
        require(row['window_start']==int(t[start,0]) and row['window_end']==int(t[-1,0]) and row['unique_sampled_frames']==len(set(ix)),'SA window metadata')
    results['sa']=m['runs']
    plan,m=loaded['dfs'];require(set(m['artifacts'])=={'trajectory.npz'},'Unexpected DFS artifacts')
    d=arrays(sources['dfs']/'trajectory.npz');equal(d['puzzle'],puzzle,'DFS puzzle')
    b,ops,status=dfs_solve(puzzle,plan['max_states']);equal(d['boards'],b,'DFS deterministic replay');equal(d['operations'],ops,'DFS actions')
    ix=indices(np.arange(len(b)));equal(d['sampled_indices'],ix,'DFS sampling');check_distance(d['distance'],board_distance(b,ix))
    require(m['solver_status']==status and m['states']==len(b) and m['operations']==len(b)-1 and m['unique_sampled_frames']==len(set(ix)),'DFS metadata')
    require(m['window_ends_at_solution']==(status=='solved') and m['matches_target']==bool(np.array_equal(b[-1],target)),'DFS terminal status')
    require(status!='solved' or solved(b[-1],puzzle),'DFS invalid solution');results['dfs']={k:m[k] for k in ('solver_status','states','window_ends_at_solution')}
    plan,m=loaded['hrm'];require(set(m['artifacts'])=={'trajectory.npz'},'Unexpected HRM artifacts')
    d=arrays(sources['hrm']/'trajectory.npz');equal(d['puzzle'],puzzle,'HRM puzzle');equal(d['target'],target,'HRM target')
    ops=np.array(['initial']+(['L']*6+['H'])*(2*plan['cycles']));equal(d['operations'],ops,'HRM schedule')
    n=len(ops);h,l=d['zH'],d['zL'];require(h.shape==l.shape and h.ndim==3 and h.shape[:2]==(n,82) and np.isfinite(h).all() and np.isfinite(l).all(),'HRM state dimensions/values')
    pred=d['predictions'];require(pred.shape==(n,81) and np.all((pred>=1)&(pred<=9)),'HRM readout dimensions/digits')
    for i in range(1,n):
        equal(h[i],h[i-1],'H held during L') if ops[i]=='L' else equal(l[i],l[i-1],'L held during H')
        if ops[i]=='L':equal(pred[i],pred[i-1],'Direct H readout during L')
    successes=np.flatnonzero((pred==target).all(-1));first=int(successes[0]) if len(successes) else None;end=first if first is not None else n-1
    require(m['first_success']==first and m['window_ends_at_solution']==(first is not None) and m['window_end']==end and int(d['window_end'])==end and m['total_states']==n,'HRM window metadata')
    z=(h[:end+1].astype(np.float32)+l[:end+1].astype(np.float32)).reshape(end+1,-1).astype(np.float64)
    ix=indices(np.arange(end+1));equal(d['sampled_indices'],ix,'HRM sampling')
    # Direct differences, independent of producer's Gram-matrix distance formula.
    maximum=max(float(np.linalg.norm(z-v,axis=1).max()) for v in z)
    expected=np.stack([np.linalg.norm(z[ix]-z[i],axis=1) for i in ix]);expected=expected/maximum if maximum>0 else expected
    check_distance(d['distance'],expected);require(np.isclose(m['distance_scale'],maximum,atol=1e-7,rtol=1e-6) and m['zero_distance_trajectory']==(maximum==0),'HRM distance scale')
    require(m['unique_sampled_frames']==len(set(ix)) and m['native_forward_parity'] is True,'Missing native parity receipt or frame mismatch')
    results['hrm']={k:m[k] for k in ('first_success','window_end','window_ends_at_solution','total_states')}
    report=dict(status='complete',board_id=reference['board_id'],results=results,
        source_manifests={k:sha(path/'manifest.json') for k,path in sources.items()},code_sha256=sha(Path(__file__)),
        limits='DFS is replayed; SA checks cover recorded states, not every unrecorded proposal or random draw. HRM recurrence parity is the producer receipt; this audit does not reload its checkpoint or recompute logits. Distances and selected-window success are recomputed from saved arrays. Censored windows are retained, not counted as reproduced successful examples.')
    out.mkdir(parents=True,exist_ok=True);(out/'comparison.json').write_text(json.dumps(report,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
