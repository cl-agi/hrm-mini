from __future__ import annotations

import math

from pathlib import Path

from typing import Any, Mapping, Sequence

import torch

import torch.nn.functional as F

import yaml

from torch import Tensor, nn

from hrm.models.hrm import HRM as BaseHRM
from hrm.models.rt import RecurrentTransformer as BaseRT
from hrm.models.layers import CastedLinear, CastedScaledEmbedding, RotaryEmbedding, MLP, SwiGLU, Attention, TransformerBlock, Transformer, rotate_half, trunc_normal_init_

class HRM(BaseHRM):
    family = "hrm"

class RecurrentTransformer(BaseRT):
    family = "rt"

def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict) or not isinstance(loaded.get("arch"), dict):
        raise ValueError(f"Config does not contain an arch mapping: {path}")
    return loaded

def analysis_architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the Mixer trainer's transformer-then-architecture precedence."""
    architecture = dict(config.get("transformer", {}))
    architecture.update({k: v for k, v in config["arch"].items() if k != "cycles_per_data"})
    architecture.setdefault("vocab_size", 10)
    architecture.setdefault("seq_len", 82)
    architecture.setdefault("is_causal", False)
    return architecture


def unwrap_state_dict(loaded: Any) -> dict[str, Tensor]:
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        loaded = loaded["model_state_dict"]
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("Checkpoint is not a non-empty state dict")
    result: dict[str, Tensor] = {}
    for raw_key, value in loaded.items():
        if not isinstance(raw_key, str) or not isinstance(value, Tensor):
            continue
        key = raw_key
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.removeprefix("module.").removeprefix("_orig_mod.")
        result[key] = value
    if not result:
        raise ValueError("Checkpoint contains no tensor-valued model parameters")
    return result

def build_model(
    checkpoint: Path,
    config_path: Path,
    device: torch.device,
    *,
    float32_analysis: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    config = load_config(config_path)
    architecture = analysis_architecture(config)
    architecture.setdefault("vocab_size", 10)
    architecture.setdefault("seq_len", 82)
    architecture.setdefault("is_causal", False)
    name = str(architecture.pop("name", ""))
    if name in ("hrm@HRM", "hrm.models.hrm@HRM"):
        model: nn.Module = HRM(architecture)
    elif name in ("rt@RecurrentTransformer", "hrm.models.rt@RecurrentTransformer"):
        model = RecurrentTransformer(architecture)
    else:
        raise ValueError(
            f"Unsupported standard train.py architecture {name!r}; "
            "supported: hrm@HRM, rt@RecurrentTransformer"
        )
    state = unwrap_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    if float32_analysis:
        model.float()
        model.embed.cast_to = torch.float32  # type: ignore[attr-defined]
    return model, config
