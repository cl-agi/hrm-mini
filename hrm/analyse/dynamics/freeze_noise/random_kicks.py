import torch, json
import numpy as np
@torch.inference_mode()
def evaluate_random_kicks(model, eval_loader, H_warmup: int = 1, H_cycles: int = 32, noise_scales: list[float] = [0.0, 0.1, 0.2, 0.3], *, output=None):
    em = {std: 0 for std in noise_scales}
    full_em = 0
    total = 0
    records = []
    for x, y in eval_loader:
        x, y = x.cuda(), y.cuda()
        total += x.shape[0]
        # --- Solve with random kicks
        batch_records = {}
        for std in noise_scales:
            e = model.embed(x)
            carry = model.initial_carry
            z_H, z_L = carry["z_H"], carry["z_L"]
            correct = False

            for _i in range(H_warmup):
                for _k in range(model.L_cycles):
                    z_L = model.L_level(z_L + z_H + e)
                z_H = model.H_level(z_H + z_L)

            for _i in range(H_cycles - H_warmup):
                perturb_H = z_H + std * torch.randn_like(e)
                for _k in range(model.L_cycles):
                    z_L = model.L_level(z_L + perturb_H + e)
            
                    preds = model.lm_head(model.H_level(z_H + z_L)).argmax(-1)
                    cur_correct = (preds == y).all(dim=-1)
                    if correct is None:
                        correct = cur_correct
                    correct |= cur_correct

            em[std] += torch.sum(correct).item()
            batch_records[f'noise_{std}_any'] = correct.cpu().numpy()
            batch_records[f'noise_{std}_final'] = cur_correct.cpu().numpy()
        
        # --- Control: Full HRM Solve
        e = model.embed(x)
        carry = model.initial_carry
        z_H, z_L = carry["z_H"], carry["z_L"]
        for _i in range(H_cycles):
            for _k in range(model.L_cycles):
                z_L = model.L_level(z_L + z_H + e)
            z_H = model.H_level(z_H + z_L)
        
        preds = model.lm_head(model.H_level(z_H + z_L)).argmax(-1)
        full_em += ((preds == y).all(dim=-1)).sum().item()

        print(json.dumps(dict(completed=total, counts=em, full_correct=full_em)), flush=True)
        batch_records['full_final_diagnostic'] = (preds == y).all(-1).cpu().numpy()
        records.append(batch_records)
        if total >= 1000:
            break

    em = {k: x / total * 100 for k, x in em.items()}

    result = dict(total=total, cumulative_em_percent=em, full_final_diagnostic_percent=full_em / total * 100)
    if output is not None:
        np.savez_compressed(output, **{k: np.concatenate([r[k] for r in records]) for k in records[0]})
    return result
