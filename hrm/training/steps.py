"""Shared optimizer step; experiment differences are explicit loss policies."""
from typing import Optional
import inspect
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from hrm.models.layers import Carry
from hrm.training.losses import model_input, readouts, readout_loss, generate

def forward(model, carry, x, attn_mask=None, step=None):
    base = model.module if hasattr(model, "module") else model
    parameters = inspect.signature(base.forward).parameters
    kwargs = {}
    if "attn_mask" in parameters: kwargs["attn_mask"] = attn_mask
    elif attn_mask is not None: raise ValueError("This model does not accept an attention mask")
    if "step" in parameters: kwargs["step"] = step
    return model(carry, x, **kwargs)

def train_step(model, carry, opt, x, y, is_autoregressive=False, final_readout_weight=None,
               y_mask=None, attn_mask=None, step=None, loss_reduction="cross_entropy_mean"):
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "supervised_auxiliary_loss") and "active" not in carry:
        carry = dict(carry, active=torch.ones(x.shape[0], dtype=torch.bool, device=x.device))
    carry, output = forward(model, carry, model_input(x, y, is_autoregressive), attn_mask, step)
    logits = output["logits"] if isinstance(output, dict) else output
    stacked = readouts(logits)
    auxiliary = output.get("auxiliary_loss") if isinstance(output, dict) else None
    if isinstance(output, dict) and hasattr(base, "supervised_auxiliary_loss"):
        extra = base.supervised_auxiliary_loss(output, y)
        auxiliary = extra if auxiliary is None else auxiliary + extra
    active = output.get("active") if isinstance(output, dict) else None
    if active is not None:
        count = active.sum()
        dist.all_reduce(count)
        row_loss = F.cross_entropy(logits.flatten(0,1).float(), y.flatten().long(), reduction="none").view_as(y).mean(1)
        task_loss = (row_loss * active).sum() * dist.get_world_size() / count
    elif y_mask is not None or loss_reduction == "element_mean":
        loss_values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1).long(), reduction="none")
        task_loss = loss_values.mean() if y_mask is None else (loss_values * y_mask.reshape(-1)).sum() / y_mask.sum()
    elif final_readout_weight is None or stacked.shape[1] == 1:
        task_loss = readout_loss(stacked, y)
    else:
        w = final_readout_weight
        task_loss = w * readout_loss(stacked[:, -1:], y) + (1.0-w) * readout_loss(stacked[:, :-1], y)
    loss = task_loss if auxiliary is None else task_loss + auxiliary
    loss.backward()
    opt.step()
    opt.zero_grad()
    with torch.no_grad():
        preds = stacked[:, -1].argmax(-1)
        target = y
        if y_mask is not None:
            preds, target = preds * y_mask, y * y_mask
        metrics = dict(loss=loss.detach(), per_position_accuracy=torch.mean(preds == target, dtype=torch.float32), exact_match=torch.mean(torch.all(preds == target, dim=-1), dtype=torch.float32))
        if isinstance(output, dict):
            metrics["task_loss"] = task_loss.detach()
            for name, value in output.items():
                if name not in {"logits", "auxiliary_loss"} and isinstance(value, Tensor) and value.numel() == 1:
                    metrics[name] = value.detach()
            if active is not None:
                metrics["_correct"] = (preds[:,1:] == y[:,1:]).all(-1)
                metrics["exact_match"] = (metrics["_correct"] * active).sum() * dist.get_world_size() / count
                metrics["per_position_accuracy"] = ((preds == y).float().mean(1) * active).sum() * dist.get_world_size() / count
    return carry, metrics

@torch.inference_mode()
def run_inference(model, carry, x, attn_mask=None, step=None):
    carry, output = forward(model, carry, x, attn_mask, step)
    logits = output["logits"] if isinstance(output, dict) else output
    return carry, readouts(logits)[:, -1].argmax(-1)
