#!/usr/bin/env bash
# Single-node torchrun with separate compiler caches for each worker.
set -euo pipefail
if [[ "${1:-}" != "--worker" ]]; then
  : "${HRM_WORKDIR:?Set HRM_WORKDIR to a writable work directory}"
  export HRM_CACHE_LAUNCH="$(hostname)-$(date -u +%Y%m%dT%H%M%S)-$$"
  exec python -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
    --no-python bash "$(cd "$(dirname "$0")" && pwd)/distributed.sh" --worker "$@"
fi
shift
: "${LOCAL_RANK:?This entry must be launched by torchrun}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HRM_WORKDIR/cache/inductor}/$HRM_CACHE_LAUNCH/rank-$LOCAL_RANK"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HRM_WORKDIR/cache/triton}/$HRM_CACHE_LAUNCH/rank-$LOCAL_RANK"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec python "$@"
