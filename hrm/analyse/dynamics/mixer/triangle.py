"""Same raw gain estimator as historical triangle, vectorized over probes."""
import argparse, sys, time
from hrm.analyse.dynamics.mixer.common import *

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--boards', type=int, default=1250)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    m, meta = load(a.seed, True)
    x, y, path = test_data()
    pass
    from hrm.analyse.dynamics.output_sensitivity.h_l_output_triangle_vjp import boundary_linearizations, rademacher_output
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    accuracy = []
    started = time.time()
    for start in range(a.offset, a.offset + a.boards, a.batch):
        ids = list(range(start, min(start + a.batch, a.offset + a.boards)))
        xb = torch.tensor(x[ids], device='cuda')
        yb = torch.tensor(y[ids, 1:], device='cuda')
        hs, ls, logits, vjps = boundary_linearizations(m, xb)
        gains = torch.full((len(ids), 2, 16, 16), float('nan'), device='cuda')
        exact = torch.stack([v.argmax(-1).eq(yb).all(-1) for v in logits], 1)
        for j in range(16):
            ps = []
            for q in range(16):
                bs = []
                for board in ids:
                    gen = torch.Generator(device='cuda').manual_seed(20260828 + board * 1000003 + j * 10007 + q)
                    bs.append(rademacher_output(logits[j][:1].shape, xb.device, gen))
                ps.append(torch.cat(bs))
            probe = torch.stack(ps)
            ch = torch.zeros((16, *hs[j].shape), device='cuda')
            ch[:, :, 1:] = probe @ m.lm_head.weight.float()
            cl = torch.zeros_like(ch)

            def record(i):
                for k, v in enumerate((ch, cl)):
                    gains[:, k, i, j] = v[:, :, 1:].float().square().sum((-1, -2)).mean(0).div(810).sqrt()
            record(j)
            for i in range(j - 1, -1, -1):
                ch, cl = torch.vmap(vjps[i])((ch, cl))
                record(i)
        valid = gains[:, :, torch.triu_indices(16, 16, device='cuda')[0], torch.triu_indices(16, 16, device='cuda')[1]]
        assert torch.isfinite(valid).all() and (valid >= 0).all()
        assert (gains[:, 1].diagonal(dim1=-2, dim2=-1) == 0).all()
        results.append(gains.cpu().numpy())
        accuracy.append(exact.cpu().numpy())
        del hs, ls, logits, vjps, gains, ch, cl
        save_json(out / 'progress.json', dict(boards_done=sum((len(v) for v in results)), total=a.boards, elapsed_seconds=time.time() - started))
        print(json.dumps(dict(boards_done=sum((len(v) for v in results)), seconds=time.time() - started)), flush=True)
    np.savez_compressed(out / 'per_board.npz', ids=np.arange(a.offset, a.offset + a.boards), rms_gain=np.concatenate(results), exact=np.concatenate(accuracy))
    meta.update(status='complete', n_boards=a.boards, board_offset=a.offset, probes=16, outer_steps=16, h_cycles_per_outer=2, l_steps_per_h=6, precision='FP32 dequantized dynamics; TF32 disabled', dataset_file=str(path), dataset_sha256=sha(path), elapsed_seconds=time.time() - started, estimator='sqrt(mean_probe(sum(non-BOS gradient^2)/(81*10))); class-centered Rademacher probes, seeds identical to historical audit', implementation='torch.vmap over 16 independent VJPs; BOS source excluded, L diagonal structurally zero', original_estimator_sha256=sha(AUDIT / 'h_l_output_triangle_vjp.py'), execution_host='unspecified')
    save_json(out / 'manifest.json', meta)
if __name__ == '__main__':
    main()
