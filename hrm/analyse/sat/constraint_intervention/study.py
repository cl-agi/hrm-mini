"""Local protocol, integrity, and atomic artifact helpers."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import numpy as np

from hrm.paths import repo_path, work_path, repository_root
ROOT = repository_root()
CODE = Path(__file__).resolve().parent
DIRECTORY = Path(os.environ.get('HRM_ANALYSIS_RECORDS', str(ROOT/'exp/error_readout/records/sat_constraints')))
CHECKS = DIRECTORY/'checks'
RUN = DIRECTORY/'run'
PROTOCOL = DIRECTORY/'protocol.json'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    return x

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)

def write_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    tmp.replace(path)

def protocol():
    return json.loads(PROTOCOL.read_text())

def require_preflight():
    p = protocol()
    if p["status"] != "ready" or p["execution"]["forward_dtype"] is None:
        raise RuntimeError("Protocol is not ready: resolve pending precision/engineering checks")
    for name in ("preflight_parity.json", "preflight_precision.json"):
        f = CHECKS / "fp32" / name
        if not f.is_file(): raise RuntimeError(f"Missing preflight: {name}")
        report = json.loads(f.read_text())
        if report.get("passed") is not True: raise RuntimeError(f"Preflight has not passed: {name}")
        if report.get("forward_dtype") != p["execution"]["forward_dtype"]:
            raise RuntimeError(f"Preflight precision does not match protocol: {name}")
        if report.get("checkpoint_sha256") != p["checkpoint"]["sha256"]:
            raise RuntimeError(f"Preflight checkpoint does not match: {name}")
        for module in ("runtime", "preflight", "perturbations"):
            expected = report.get(f"{module}_sha256")
            if expected is not None and expected != sha256(CODE / f"{module}.py"):
                raise RuntimeError(f"Preflight code changed: {module}")
    return p

def freeze_run():
    p = require_preflight()
    RUN.mkdir(parents=True, exist_ok=True)
    files = [PROTOCOL] + sorted(CODE.glob('*.py'))
    hashes = {str(f.resolve()): sha256(f) for f in files}
    path = RUN / "run_manifest.json"
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "hashes": hashes,
               "protocol": p, "root": str(ROOT)}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["hashes"] != hashes:
            raise RuntimeError("Code/protocol changed after run freeze; use a separate experiment version")
    else:
        # Single coordinator creates this before parallel workers are started.
        write_json(path, payload)
    return p

def assert_frozen():
    p = require_preflight()
    path = RUN / "run_manifest.json"
    if not path.is_file(): raise RuntimeError("Run the coordinator freeze step first")
    for rel, expected in json.loads(path.read_text())["hashes"].items():
        if sha256(ROOT / rel) != expected: raise RuntimeError(f"Frozen input changed: {rel}")
    return p

def check_backup():
    from .runtime import verify_checkpoint
    checkpoint=verify_checkpoint()
    p=protocol()
    files=[Path(p['data']['root'])/f'{split}_n1024.npz' for split in ('train','valid','test')]
    for file in files:
        if sha256(file)!=p['data']['sha256'][file.name]:raise RuntimeError(f'Dataset changed: {file}')
    return dict(status='passed',checkpoint=checkpoint['sha256'],files=len(files))
