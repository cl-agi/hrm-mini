#!/usr/bin/env python3
"""Estimate the four blocks of one HRM H-block's joint H/L Jacobian.

One transition follows the loaded model's recurrence.  For native HRM:

    L_i = L_level^6(L_{i-1}, H_{i-1}, x)
    H_i = H_level(H_{i-1} + L_i)

Models exposing ``h_step`` may override the H update; e.g. the L-driven-H
ablation uses ``H_i = h_step(L_i) = H_level(L_i)``.

For every requested test board and H-block, matrix-free JVPs estimate the
isotropic RMS gains of d(H_i, L_i) / d(H_{i-1}, L_{i-1}).  BOS is held fixed
in the input probes and excluded from output RMS, matching the other audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

SCHEMA_VERSION = "h-l-joint-transition-jacobian-v1"
BLOCKS = (
    ("H", "H", "j_h_from_h_gain_sq"),
    ("L", "H", "j_l_from_h_gain_sq"),
    ("H", "L", "j_h_from_l_gain_sq"),
    ("L", "L", "j_l_from_l_gain_sq"),
)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def nonbos_mean_square(value: Tensor) -> Tensor:
    return value[:, 1:].to(torch.float32).square().mean(dim=(1, 2))


def rademacher_probe(
    shape: torch.Size, device: torch.device, generator: torch.Generator
) -> Tensor:
    value = torch.randint(
        0, 2, shape, device=device, generator=generator, dtype=torch.int8
    ).to(torch.float32)
    value.mul_(2).sub_(1)
    value[:, 0] = 0
    return value


def joint_transition(
    model: nn.Module, latent: Tensor, z_h: Tensor, z_l: Tensor
) -> tuple[Tensor, Tensor]:
    next_l = z_l
    for _ in range(6):
        next_l = model.L_level(next_l + z_h + latent)  # type: ignore[attr-defined]
    next_h = advance_h(model, z_h, next_l)
    return next_h, next_l


def advance_h(model: nn.Module, z_h: Tensor, z_l: Tensor) -> Tensor:
    """Apply the loaded architecture's actual H update."""
    h_step = getattr(model, "h_step", None)
    if h_step is not None:
        return h_step(z_l)
    return model.H_level(z_h + z_l)  # type: ignore[attr-defined]


def summarize(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    wide_by_block: dict[int, dict[str, Any]] = {}
    blocks = sorted({int(row["h_block"]) for row in rows})
    for block in blocks:
        selected = [row for row in rows if int(row["h_block"]) == block]
        wide: dict[str, Any] = {
            "h_block": block,
            "outer_cycle": int(selected[0]["outer_cycle"]),
            "h_cycle_in_outer": int(selected[0]["h_cycle_in_outer"]),
            "n_board_probes": len(selected),
        }
        for output_state, input_state, field in BLOCKS:
            square = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
            gain = np.sqrt(np.maximum(square, 0.0))
            mean_square = float(square.mean())
            square_se = float(square.std(ddof=1) / math.sqrt(square.size)) if square.size > 1 else 0.0
            rms_gain = math.sqrt(max(mean_square, 0.0))
            rms_gain_se = 0.5 * square_se / max(rms_gain, 1e-30)
            prefix = f"j_{output_state.lower()}_from_{input_state.lower()}"
            wide[f"{prefix}_rms_gain"] = rms_gain
            wide[f"{prefix}_rms_gain_se"] = rms_gain_se
            long_rows.append(
                {
                    "h_block": block,
                    "outer_cycle": int(selected[0]["outer_cycle"]),
                    "h_cycle_in_outer": int(selected[0]["h_cycle_in_outer"]),
                    "output_state": output_state,
                    "input_state": input_state,
                    "n_board_probes": int(square.size),
                    "mean_gain_sq": mean_square,
                    "mean_gain_sq_se": square_se,
                    "rms_gain": rms_gain,
                    "rms_gain_se_delta": rms_gain_se,
                    "directional_gain_mean": float(gain.mean()),
                    "directional_gain_sd": float(gain.std(ddof=1)) if gain.size > 1 else 0.0,
                    "directional_gain_q10": float(np.quantile(gain, 0.10)),
                    "directional_gain_q50": float(np.quantile(gain, 0.50)),
                    "directional_gain_q90": float(np.quantile(gain, 0.90)),
                }
            )
        wide_by_block[block] = wide
    return long_rows, [wide_by_block[block] for block in blocks]


def run_batch(
    *,
    model: nn.Module,
    x: Tensor,
    board_ids: Sequence[int],
    probes: int,
    generator: torch.Generator,
) -> tuple[list[dict[str, Any]], float]:
    latent = model.embed(x)  # type: ignore[attr-defined]
    z_h = torch.broadcast_to(model.initial_carry["z_H"], latent.shape).clone()  # type: ignore[attr-defined]
    z_l = torch.broadcast_to(model.initial_carry["z_L"], latent.shape).clone()  # type: ignore[attr-defined]
    rows: list[dict[str, Any]] = []
    zero = torch.zeros_like(z_h, dtype=torch.float32)
    max_forward_parity = 0.0

    for h_block in range(1, 33):
        def mapping(h_value: Tensor, l_value: Tensor) -> tuple[Tensor, Tensor]:
            return joint_transition(model, latent, h_value, l_value)

        clean_h: Tensor | None = None
        clean_l: Tensor | None = None
        for probe in range(probes):
            v_h = rademacher_probe(z_h.shape, x.device, generator)
            v_l = rademacher_probe(z_l.shape, x.device, generator)
            (base_h, base_l), (dh_h, dl_h) = torch.func.jvp(
                mapping, (z_h, z_l), (v_h, zero)
            )
            (base_h_l, base_l_l), (dh_l, dl_l) = torch.func.jvp(
                mapping, (z_h, z_l), (zero, v_l)
            )
            max_forward_parity = max(
                max_forward_parity,
                float((base_h.to(torch.float32) - base_h_l.to(torch.float32)).abs().max().item()),
                float((base_l.to(torch.float32) - base_l_l.to(torch.float32)).abs().max().item()),
            )
            clean_h, clean_l = base_h, base_l
            values = {
                "j_h_from_h_gain_sq": nonbos_mean_square(dh_h),
                "j_l_from_h_gain_sq": nonbos_mean_square(dl_h),
                "j_h_from_l_gain_sq": nonbos_mean_square(dh_l),
                "j_l_from_l_gain_sq": nonbos_mean_square(dl_l),
            }
            for board_index, board_id in enumerate(board_ids):
                rows.append(
                    {
                        "board_id": int(board_id),
                        "h_block": h_block,
                        "outer_cycle": (h_block + 1) // 2,
                        "h_cycle_in_outer": 1 + (h_block - 1) % 2,
                        "probe": probe,
                        **{
                            name: float(value[board_index].item())
                            for name, value in values.items()
                        },
                    }
                )
        assert clean_h is not None and clean_l is not None
        z_h = clean_h.detach()
        z_l = clean_l.detach()
    return rows, max_forward_parity



def main(argv=None):
    from .local_analysis import main as run
    return run('local', argv)

if __name__ == "__main__":
    raise SystemExit(main())
