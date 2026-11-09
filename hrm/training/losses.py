from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from hrm.models.layers import Carry

def model_input(x: Tensor, y: Tensor, is_autoregressive: bool) -> Tensor:
    """Decoder-only models are teacher-forced on `[question ++ answer]`; encoders only see the question."""
    return torch.cat([x, y], dim=-1) if is_autoregressive else x

def readouts(y_hat: Tensor) -> Tensor:
    """Normalise model outputs to `[batch, num_readouts, seq_len, vocab]`.

    Deeply supervised models return one readout per supervision point; every other model returns a
    single `[batch, seq_len, vocab]` set of logits. The last readout is always the final prediction.
    """
    return y_hat if y_hat.ndim == 4 else y_hat.unsqueeze(1)

def readout_loss(logits: Tensor, y: Tensor) -> Tensor:
    """Mean cross-entropy (in f32) of `[batch, num_readouts, seq_len, vocab]` against `y`."""
    targets = y.unsqueeze(1).expand(-1, logits.shape[1], -1)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).to(torch.float32), targets.reshape(-1).long(), reduction="mean")

@torch.inference_mode()
def generate(model: nn.Module, x: Tensor) -> Tensor:
    """Greedy decoding of the answer block for autoregressive models, one token at a time.

    The answer slots start as the BOS token (id 0) and are overwritten left to right; causal masking
    keeps the not-yet-decoded slots from leaking into earlier positions. Shapes stay static so the
    compiled graph is reused across all decoding steps.
    """
    block_len = x.shape[-1]
    seq = torch.cat([x, torch.zeros_like(x)], dim=-1)
    for pos in range(1, block_len):
        _carry, y_hat = model({}, seq)
        seq[:, block_len + pos] = torch.argmax(y_hat[:, pos], dim=-1)
    return seq[:, block_len:]
