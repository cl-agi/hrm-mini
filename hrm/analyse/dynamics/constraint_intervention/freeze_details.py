from hrm.paths import repo_path
from hrm.analyse.dynamics.constraint_intervention.population import *

@torch.inference_mode()
def run():
    torch.set_num_threads(4)
    torch.manual_seed(json.loads((repo_path('data/dynamics/constraint_intervention/config.json')).read_text())['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    model, ident = load_metadata(argparse.Namespace(checkpoint=str(repo_path('checkpoint/sudoku/cl_agi_hrm_mini/checkpoints/1000_tuned_hrm_new/epoch_19.pt')), config=str(repo_path('data/dynamics/constraint_intervention/model_config.json')), architecture='hrm', device='cuda'))
    model.eval().requires_grad_(False)
    d = np.load(repo_path('data/dynamics/constraint_intervention/cohort.npz'))
    all_e = []
    all_a = []
    all_r = []
    for start in range(0, len(d['input_ids']), 96):
        x = torch.tensor(d['input_ids'][start:start + 96], device='cuda')
        y = torch.tensor(d['labels'][start:start + 96], device='cuda')
        emb = model.embed(x)
        h = model.initial_carry['z_H'].expand_as(emb)
        l = model.initial_carry['z_L'].expand_as(emb)
        e = np.full((len(x), 4, 50), np.nan)
        a = e.copy()
        r = e.copy()
        for branch in range(4):
            current = l
            for j in range(branch * 10, 50):
                old = current
                current = model.L_level(current + h + emb)
                logits = model.lm_head(model.H_level(h + current))
                e[:, branch, j] = count(logits).cpu().numpy()
                pred = logits.argmax(-1)
                a[:, branch, j] = ((pred == y) | (y == -100)).all(-1).float().cpu().numpy()
                r[:, branch, j] = ((current.float() - old.float()).flatten(1).norm(dim=1) / old.float().flatten(1).norm(dim=1)).cpu().numpy()
                if j == branch * 10 + 9:
                    native = current
            l = native
            h = model.H_level(h + l)
        all_e.append(e)
        all_a.append(a)
        all_r.append(r)
        print(start + len(x), flush=True)
    np.savez_compressed(R / 'freeze_details.npz', count=np.concatenate(all_e), accuracy=np.concatenate(all_a), relative=np.concatenate(all_r))
if __name__ == "__main__":
    run()
