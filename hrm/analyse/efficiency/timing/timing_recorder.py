"""Synchronized, per-rank training phase records; no training work at import."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingClock:
    """Record native training phases with one CUDA/DDP boundary at each end.

    CUDA completion and the slowest rank are included in phase wall time. JSON
    writes are outside phase intervals; full_seed_seconds includes recorder
    overhead. Lazy compilation is part of the first training/evaluation epoch.
    No model forward, optimizer step, or random-number draw is added.
    """

    def __init__(self, *, config: dict, seed: int, rank: int, world_size: int, task: str):
        output_dir = os.environ.get("HRM_TIMING_DIR")
        if not output_dir:
            raise RuntimeError("Set HRM_TIMING_DIR to a new directory for this timing run")
        self.path = Path(output_dir).resolve() / f"seed_{seed}" / f"rank_{rank:03d}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Timing record already exists: {self.path}")

        import torch
        self.torch = torch
        self._boundary()
        self._origin = time.perf_counter()
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.record = {
            "schema_version": 1,
            "status": "running",
            "task": task,
            "seed": seed,
            "rank": rank,
            "world_size": world_size,
            "started_utc": _utc_now(),
            "config": config,
            "hardware": {
                "hostname": platform.node(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "local_rank": os.environ.get("LOCAL_RANK"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_compute_capability": [properties.major, properties.minor],
            },
            "timing_method": "perf_counter with CUDA synchronization and distributed barriers at phase boundaries",
            "compilation": "Lazy compilation is included in the first training/evaluation epoch; no optimizer warmup steps are added.",
            "phases": [],
        }
        self._active = None
        self._write()

    def _boundary(self):
        self.torch.cuda.synchronize()
        dist = self.torch.distributed
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self.torch.cuda.synchronize()

    def _write(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def start_phase(self, phase: str, *, epoch=None, step=None):
        if self._active is not None or self.record["status"] != "running":
            raise RuntimeError("Cannot start phase while another phase is active or the run is complete")
        self._active = {
            "phase": phase, "epoch": epoch, "step_start": step,
            "started_utc": _utc_now(),
        }
        # Persist active-phase identity before starting its timer, keeping I/O
        # outside the measured phase interval.
        self.record["active_phase"] = dict(self._active)
        self._write()
        self._boundary()
        self._active["started_utc"] = _utc_now()
        self._phase_start = time.perf_counter()

    def stop_phase(self, *, step=None):
        if self._active is None:
            raise RuntimeError("No active phase to stop")
        self._boundary()
        elapsed = time.perf_counter() - self._phase_start
        phase = self._active | {
            "finished_utc": _utc_now(), "elapsed_seconds": elapsed, "step_end": step,
        }
        if step is not None and phase["step_start"] is not None:
            phase["optimizer_steps"] = step - phase["step_start"]
        self.record["phases"].append(phase)
        self.record.pop("active_phase", None)
        self._active = None
        self._write()

    def finish(self, *, actual_steps: int, expected_steps: int):
        if self._active is not None or self.record["status"] != "running":
            raise RuntimeError("Finish requires no active phase and a running record")
        self._boundary()
        elapsed = time.perf_counter() - self._origin
        totals = {}
        for phase in self.record["phases"]:
            name = phase["phase"]
            totals[name] = totals.get(name, 0.0) + phase["elapsed_seconds"]
        self.record.update({
            "status": "completed" if actual_steps == expected_steps else "step_count_mismatch",
            "finished_utc": _utc_now(),
            "full_seed_seconds": elapsed,
            "phase_totals_seconds": totals,
            "actual_steps": actual_steps,
            "expected_steps": expected_steps,
        })
        self._write()
        if actual_steps != expected_steps:
            raise RuntimeError(f"Training step count mismatch: {actual_steps} != {expected_steps}")
