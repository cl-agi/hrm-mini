"""Table-8 style width scaling; Adam-like adaptation for native AdamATan2.

Screening variant: hidden WD = base_weight_decay, independent of width.
Only this coefficient differs from the frozen linear-WD baseline.
Reference for LR grouping: microsoft/mup/optim.py MuAdamW. Fixed head_dim=64 means the
attention logit factor does not change. QKV's leading size-3 dimension is
finite; both trailing dimensions grow, so QKV belongs to the hidden group.
"""

def parameter_groups(model, base_weight_decay, hidden_lr_factor=1.0):
    m = model.mup_width_mult
    hidden, boundary = [], []
    hidden_names, boundary_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in {"embed.weight", "lm_head.weight"}:
            boundary.append(p); boundary_names.append(name)
        elif name.startswith(("H_level.", "L_level.", "core.")) and name.endswith(".weight") and p.ndim in (2, 3):
            hidden.append(p); hidden_names.append(name)
        else:
            raise ValueError(f"Unclassified muP parameter: {name} {tuple(p.shape)}")
    if not hidden or len(boundary) != 2:
        raise ValueError("Incomplete muP parameter groups")
    return [dict(params=hidden, group_name="hidden", param_names=hidden_names,
                 lr_multiplier=hidden_lr_factor/m, weight_decay=base_weight_decay),
            dict(params=boundary, group_name="embedding_readout", param_names=boundary_names,
                 lr_multiplier=1.0, weight_decay=1.0)]


def group_metadata(optim, base_lr):
    return [dict(name=g["group_name"], parameter_names=g["param_names"],
                 parameters=sum(p.numel() for p in g["params"]),
                 lr_multiplier=g["lr_multiplier"], peak_lr=base_lr*g["lr_multiplier"],
                 weight_decay=g["weight_decay"],
                 peak_lr_times_wd=base_lr*g["lr_multiplier"]*g["weight_decay"])
            for g in optim.param_groups]
