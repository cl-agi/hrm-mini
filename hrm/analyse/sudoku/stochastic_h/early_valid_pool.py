"""Inference-only pool: remove successful rows, refill, then compact the tail."""
import torch


@torch.inference_mode()
def evaluate_pool(model, inputs, validate, batch_size=96, max_outer=4096, progress=None):
    n = len(inputs)
    first_hit = torch.zeros(n, dtype=torch.long, device=inputs.device)
    used = torch.zeros_like(first_hit)
    initial = model.initial_carry
    def fresh(count):
        return {k:v.expand(count, inputs.shape[1], -1).clone() for k,v in initial.items()}
    next_id = min(batch_size, n)
    ids = torch.arange(next_id, device=inputs.device)
    age = torch.zeros_like(ids)
    carry = fresh(next_id)
    iteration = 0
    while len(ids):
        carry, output = model(carry, inputs[ids])
        logits = output['logits'] if isinstance(output, dict) else output
        pred = logits.argmax(-1)
        age += 1
        valid = validate(pred, inputs[ids])
        done = valid | (age >= max_outer)
        first_hit[ids[valid]] = age[valid]
        used[ids[done]] = age[done]
        keep = ~done
        ids, age = ids[keep], age[keep]
        carry = {k:v[keep] for k,v in carry.items()}
        take = min(batch_size-len(ids), n-next_id)
        if take:
            ids = torch.cat((ids, torch.arange(next_id,next_id+take,device=inputs.device)))
            age = torch.cat((age, torch.zeros(take,dtype=torch.long,device=inputs.device)))
            new = fresh(take)
            carry = {k:torch.cat((v,new[k])) for k,v in carry.items()}
            next_id += take
        iteration += 1
        if progress and (iteration == 1 or iteration % 500 == 0 or not len(ids)):
            progress(iteration, next_id-len(ids), next_id, len(ids), first_hit, used)
    return first_hit, used
