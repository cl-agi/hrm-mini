#!/usr/bin/env python3
"""Test how well a current H-cycle Jacobian predicts the natural future trajectory.

Two tests are emitted:

1. Natural secant: J_s (z_{s+1} - z_s) predicts the next natural update
   z_{s+2} - z_{s+1}.  The four H/L blocks are retained separately.
2. Frozen affine rollout: the local affine model at s is iterated for several
   H-cycles and compared with the true nonlinear trajectory, persistence, and
   constant-velocity baselines.

Primary values are raw, non-BOS latent/logit vectors.  No cell centering,
projection, or spectrum normalization is applied.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .h_l_joint_transition_jacobian import joint_transition
from .h_l_state_rank_audit import atomic_csv, atomic_json

SCHEMA_VERSION = "h-l-natural-taylor-prediction-v1"
H_BLOCKS = 32


def raw_nonbos(value: Tensor) -> Tensor:
    return value[:, 1:].to(torch.float32).reshape(value.shape[0], -1)


def zero_bos(value: Tensor) -> Tensor:
    result = value.to(torch.float32).clone()
    result[:, 0] = 0
    return result


def vector_metrics(predicted: Tensor, actual: Tensor) -> dict[str, Tensor]:
    pred = raw_nonbos(predicted)
    true = raw_nonbos(actual)
    error = pred - true
    pred_l2 = pred.norm(dim=-1)
    true_l2 = true.norm(dim=-1)
    error_l2 = error.norm(dim=-1)
    denominator = (pred_l2 * true_l2).clamp_min(1e-30)
    cosine = (pred * true).sum(-1) / denominator
    dimensions = pred.shape[-1]
    return {
        "pred_l2": pred_l2,
        "true_l2": true_l2,
        "error_l2": error_l2,
        "pred_rms": pred_l2 / math.sqrt(dimensions),
        "true_rms": true_l2 / math.sqrt(dimensions),
        "error_rms": error_l2 / math.sqrt(dimensions),
        "cosine": cosine,
        "cosine_distance": 1 - cosine,
        "norm_ratio": pred_l2 / true_l2.clamp_min(1e-30),
        "relative_l2_error": error_l2 / true_l2.clamp_min(1e-30),
        "r2_vs_zero": 1 - error_l2.square() / true_l2.square().clamp_min(1e-30),
    }


def append_metrics(
    rows: list[dict[str, Any]],
    identifiers: Mapping[str, Any],
    predicted: Tensor,
    actual: Tensor,
    board_ids: Sequence[int],
) -> None:
    metrics = vector_metrics(predicted, actual)
    for board_index, board_id in enumerate(board_ids):
        rows.append({
            "board_id": int(board_id),
            **identifiers,
            **{name: float(value[board_index].item()) for name, value in metrics.items()},
        })


def collect_trajectory(
    model: nn.Module, latent: Tensor
) -> tuple[list[Tensor], list[Tensor]]:
    z_h = torch.broadcast_to(model.initial_carry["z_H"], latent.shape).clone()  # type: ignore[attr-defined]
    z_l = torch.broadcast_to(model.initial_carry["z_L"], latent.shape).clone()  # type: ignore[attr-defined]
    hs, ls = [z_h.detach()], [z_l.detach()]
    with torch.no_grad():
        for _ in range(H_BLOCKS):
            z_h, z_l = joint_transition(model, latent, z_h, z_l)
            hs.append(z_h.detach())
            ls.append(z_l.detach())
    return hs, ls


def run_batch(
    model: nn.Module,
    x: Tensor,
    board_ids: Sequence[int],
    rollout_sources: Sequence[int],
    max_horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    latent = model.embed(x).detach()  # type: ignore[attr-defined]
    hs, ls = collect_trajectory(model, latent)
    local_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    zero = torch.zeros_like(hs[0], dtype=torch.float32)
    max_base_error = 0.0

    # The local secant test uses every source with two subsequent states.
    for source in range(H_BLOCKS - 1):
        def mapping(h_value: Tensor, l_value: Tensor) -> tuple[Tensor, Tensor]:
            return joint_transition(model, latent, h_value, l_value)

        delta_h = zero_bos(hs[source + 1] - hs[source])
        delta_l = zero_bos(ls[source + 1] - ls[source])
        true_h = zero_bos(hs[source + 2] - hs[source + 1])
        true_l = zero_bos(ls[source + 2] - ls[source + 1])
        (base_h, base_l), (hh, lh) = torch.func.jvp(
            mapping, (hs[source], ls[source]), (delta_h, zero)
        )
        (base_h_l, base_l_l), (hl, ll) = torch.func.jvp(
            mapping, (hs[source], ls[source]), (zero, delta_l)
        )
        max_base_error = max(
            max_base_error,
            float((base_h.float() - base_h_l.float()).abs().max().item()),
            float((base_l.float() - base_l_l.float()).abs().max().item()),
            float((base_h.float() - hs[source + 1].float()).abs().max().item()),
            float((base_l.float() - ls[source + 1].float()).abs().max().item()),
        )
        predictions = {
            "H": {"H_source": hh, "L_source": hl, "total": hh + hl, "constant_velocity": delta_h},
            "L": {"H_source": lh, "L_source": ll, "total": lh + ll, "constant_velocity": delta_l},
        }
        actuals = {"H": true_h, "L": true_l}
        for destination, conditions in predictions.items():
            for condition, predicted in conditions.items():
                append_metrics(
                    local_rows,
                    {
                        "source_h_block": source,
                        "predicted_update_h_block": source + 2,
                        "destination_state": destination,
                        "condition": condition,
                    },
                    predicted,
                    actuals[destination],
                    board_ids,
                )

    # Multi-step prediction freezes one local affine model at each requested source.
    for source in rollout_sources:
        if source + 2 > H_BLOCKS:
            continue

        def mapping(h_value: Tensor, l_value: Tensor) -> tuple[Tensor, Tensor]:
            return joint_transition(model, latent, h_value, l_value)

        (base_h, base_l), jvp_fn = torch.func.linearize(mapping, hs[source], ls[source])
        anchor_h, anchor_l = hs[source], ls[source]
        initial_delta_h = zero_bos(hs[source + 1] - anchor_h)
        initial_delta_l = zero_bos(ls[source + 1] - anchor_l)
        predicted_h, predicted_l = base_h, base_l
        base_logits = model.lm_head(base_h)[:, 1:].to(torch.float32)  # type: ignore[attr-defined]

        for horizon in range(2, min(max_horizon, H_BLOCKS - source) + 1):
            offset_h = zero_bos(predicted_h - anchor_h)
            offset_l = zero_bos(predicted_l - anchor_l)
            image_h, image_l = jvp_fn(offset_h, offset_l)
            predicted_h = base_h + image_h
            predicted_l = base_l + image_l
            true_h, true_l = hs[source + horizon], ls[source + horizon]
            persistence_h, persistence_l = base_h, base_l
            velocity_h = base_h + (horizon - 1) * initial_delta_h
            velocity_l = base_l + (horizon - 1) * initial_delta_l

            for state, predicted, true, persistence, velocity, base in (
                ("H", predicted_h, true_h, persistence_h, velocity_h, base_h),
                ("L", predicted_l, true_l, persistence_l, velocity_l, base_l),
            ):
                actual_displacement = zero_bos(true - base)
                for condition, estimate in (
                    ("frozen_jacobian", predicted),
                    ("persistence", persistence),
                    ("constant_velocity", velocity),
                ):
                    predicted_displacement = zero_bos(estimate - base)
                    append_metrics(
                        rollout_rows,
                        {
                            "source_h_block": source,
                            "horizon": horizon,
                            "destination_state": state,
                            "condition": condition,
                        },
                        predicted_displacement,
                        actual_displacement,
                        board_ids,
                    )

            predicted_logits = model.lm_head(predicted_h)[:, 1:].to(torch.float32)  # type: ignore[attr-defined]
            true_logits = model.lm_head(true_h)[:, 1:].to(torch.float32)  # type: ignore[attr-defined]
            velocity_logits = model.lm_head(velocity_h)[:, 1:].to(torch.float32)  # type: ignore[attr-defined]
            for condition, estimate in (
                ("frozen_jacobian", predicted_logits),
                ("persistence", base_logits),
                ("constant_velocity", velocity_logits),
            ):
                predicted_cells = estimate - base_logits
                actual_cells = true_logits - base_logits
                predicted_displacement = torch.cat(
                    (torch.zeros_like(predicted_cells[:, :1]), predicted_cells), dim=1
                )
                actual_displacement = torch.cat(
                    (torch.zeros_like(actual_cells[:, :1]), actual_cells), dim=1
                )
                metrics = vector_metrics(predicted_displacement, actual_displacement)
                argmax_match = estimate.argmax(-1).eq(true_logits.argmax(-1)).float().mean(-1)
                for board_index, board_id in enumerate(board_ids):
                    rollout_rows.append({
                        "board_id": int(board_id),
                        "source_h_block": source,
                        "horizon": horizon,
                        "destination_state": "output_logits",
                        "condition": condition,
                        **{name: float(value[board_index].item()) for name, value in metrics.items()},
                        "argmax_cell_match_fraction": float(argmax_match[board_index].item()),
                    })

    return local_rows, rollout_rows, max_base_error



def main(argv=None):
    from .local_analysis import main as run
    return run('taylor', argv)

if __name__ == "__main__":
    raise SystemExit(main())
