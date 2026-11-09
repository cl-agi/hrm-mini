"""Sequential stages with independent puzzle workers; no launch on import."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import hrm.analyse.sat.constraint_intervention.study as study


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_idle_devices(ids):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    busy = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    found = {}
    for line in rows.splitlines():
        index, uid, name, used, total, utilization = [x.strip() for x in line.split(",")]
        index = int(index)
        if index in ids:
            if uid in busy or int(used) > 128 or int(utilization) != 0:
                raise RuntimeError(f"Approved GPU {index} is currently occupied: {line}")
            found[index] = dict(uuid=uid, name=name, used_mib=int(used), total_mib=int(total))
    if set(found) != set(ids):
        raise RuntimeError("Some approved GPUs were not found")
    return found


def status(stage, state, **details):
    study.write_json(study.RUN / "status.json", {
        "stage": stage, "status": state, "utc": utc(), "coordinator_pid": os.getpid(), **details,
    })


def stop_children(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + 15
    for child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
    for child in children:
        child.wait()


def run_stage(stage, commands):
    children, streams = [], []
    try:
        for label, command, environment in commands:
            log = (study.RUN / "logs" / f"{label}.log").open("a", buffering=1)
            log.write(f"\n{utc()} START {json.dumps(command)}\n")
            streams.append(log)
            child = subprocess.Popen(command, cwd=study.ROOT, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
        while True:
            codes = [child.poll() for child in children]
            status(stage, "RUNNING", workers=[dict(label=item[0], pid=child.pid, exit_code=code)
                                              for item, child, code in zip(commands, children, codes)])
            errors = [(item[0], code) for item, code in zip(commands, codes) if code is not None and code != 0]
            if errors:
                raise RuntimeError(f"Stage {stage} failed: {errors}; see per-worker logs")
            if all(code is not None for code in codes):
                return
            time.sleep(10)
    finally:
        stop_children(children)
        for stream in streams:
            stream.close()


def worker_commands(stage, script, gpu_ids):
    commands = []
    for index, physical in enumerate(gpu_ids):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(physical), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        command = [sys.executable, "-B", "-m", "hrm.analyse.sat.constraint_intervention."+Path(script).stem, "--device", "cuda:0", "--worker-index", str(index),
                   "--workers", str(len(gpu_ids))]
        commands.append((f"{stage}_{index}_gpu{physical}", command, env))
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be unique")
    p = study.require_preflight()
    study.RUN.mkdir(parents=True, exist_ok=True)
    lock = (study.RUN / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (study.RUN / "logs").mkdir(exist_ok=True)
    stage = "integrity"
    try:
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        study.check_backup()
        study.freeze_run()
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
        stage = "native_collection"
        hardware = verify_idle_devices(args.gpus)
        study.write_json(study.RUN / "hardware_at_launch.json", {"utc": utc(), "gpus": hardware})
        run_stage(stage, worker_commands("collect", "collect.py", args.gpus))
        stage = "probe_fit"
        run_stage(stage, [("fit", [sys.executable, "-B", "-m", "hrm.analyse.sat.constraint_intervention.fit_and_report", "fit"], env)])
        stage = "interventions"
        verify_idle_devices(args.gpus)
        run_stage(stage, worker_commands("intervene", "intervene.py", args.gpus))
        stage = "statistics"
        run_stage(stage, [("report", [sys.executable, "-B", "-m", "hrm.analyse.sat.constraint_intervention.fit_and_report", "report"], env)])
        status("complete", "COMPLETE", run_manifest_sha256=study.sha256(study.RUN / "run_manifest.json"))
    except BaseException as exc:
        status(stage, "STOPPED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
