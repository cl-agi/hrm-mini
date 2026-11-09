"""Paired B02 cohort; retain full actual case-6566 trajectories, all seeds."""
import argparse,time
from hrm.analyse.dynamics.mixer.common import *

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    out=ROOT/f'beads_seed{args.seed}';out.mkdir(exist_ok=False)
    m,meta=load(args.seed)
    cohort=OLD/'experiment/L4096_case_search_20260928/cohort.npz'
    d=np.load(cohort);ids=d['ids'];x=torch.tensor(d['inputs'],device='cuda');y=torch.tensor(d['targets'],device='cuda')
    tx,ty,_=test_data();assert np.array_equal(tx[ids],d['inputs']) and np.array_equal(ty[ids],d['targets'])
    ix=int(np.flatnonzero(ids==6566)[0]); emb=m.embed(x);h=m.zH_init.expand_as(emb).clone();l=m.zL_init.expand_as(emb).clone()
    # Verify the explicit recurrence against archive forward before changing schedule.
    native,logits=m(dict(z_H=h.clone(),z_L=l.clone()),x)
    hh,ll=transition(m,emb,h,l)
    assert torch.equal(hh,native['z_H']) and torch.equal(ll,native['z_L'])
    np.savez(out/'cohort.npz',**dict(d))
    records=[];started=time.time()
    meta.update(execution_host='unspecified',schedule='4096L+1H from initialization, 8 H cycles',
        cohort_source=str(cohort),cohort_sha256=sha(cohort),board_ids=ids.tolist(),
        selection='All 32 original B02 cohort inputs; full trajectory is preselected original case 6566, independent of Mixer outcome',
        precision='native BF16 eager inference',native_forward_parity='bitwise H and L',status='running')
    save_json(out/'manifest.json',meta)
    for hc in range(1,9):
        trace=np.lib.format.open_memmap(out/f'H{hc:03d}_case6566_bits.npy',mode='w+',dtype=np.uint16,shape=(4097,82,512))
        trace[0]=l[ix].contiguous().view(torch.uint16).cpu().numpy()
        hb=h.clone();tail=[]
        for step in range(1,4097):
            l=m.L_level(l+h+emb)
            trace[step]=l[ix].contiguous().view(torch.uint16).cpu().numpy()
            if step>3968:tail.append(l.clone())
        trace.flush();del trace
        h=m.H_level(h+l);pred=m.lm_head(h).argmax(-1)
        err=(pred[:,1:]!=y[:,1:]).sum(1).cpu().numpy()
        tails=torch.stack(tail)
        np.save(out/f'H{hc:03d}_cohort_tail_bits.npy',tails.contiguous().view(torch.uint16).cpu().numpy())
        np.savez(out/f'H{hc:03d}_boundary.npz',H_before_bits=hb.contiguous().view(torch.uint16).cpu().numpy(),
            H_after_bits=h.contiguous().view(torch.uint16).cpu().numpy(),L_after_bits=l.contiguous().view(torch.uint16).cpu().numpy(),predictions=pred.cpu().numpy())
        for bi,case in enumerate(ids):
            c=tails[:,bi].reshape(128,-1).double();c-=c.mean(0);ev=torch.linalg.eigvalsh(c@c.T).clamp_min(0)
            records.append(dict(case=int(case),H_cycle=hc,errors=int(err[bi]),
                tail_pc123=float(ev[-3:].sum()/ev.sum().clamp_min(1e-30)),tail_scatter_rms=float(c.square().mean().sqrt())))
        save_json(out/'screen.json',records)
        meta.update(cycles_done=hc,elapsed_seconds=time.time()-started)
        save_json(out/'manifest.json',meta);print(json.dumps(dict(H=hc,correct=int((err==0).sum()),case6566_errors=int(err[ix]),seconds=meta['elapsed_seconds'])),flush=True)
    summary=[]
    for case in ids:
        rows=[r for r in records if r['case']==case];e=[r['errors'] for r in rows];first=next((i+1 for i,v in enumerate(e) if v==0),None)
        summary.append(dict(case=int(case),errors=e,first_correct_H=first,stable_after_first=bool(first and all(v==0 for v in e[first-1:]))))
    save_json(out/'summary.json',summary);meta['status']='complete';save_json(out/'manifest.json',meta)

if __name__=='__main__':main()
