#!/usr/bin/env bash
# run_docker.sh — launch the BTP container on the LNMIIT DGX.
#
# WHY THIS EXISTS (and why we do NOT `docker build`):
#   /var/lib/docker sits on `/`, which is 100% full with ~1.8 GB free. Building the
#   image in Dockerfile (~6 GB) would fail, and even if it squeaked through it would
#   push a SHARED machine over the edge. So instead:
#     - reuse nvcr.io/nvidia/pytorch:24.05-py3, already pulled locally (no download)
#     - keep every byte we create on /home, which has ~1.1 TB free
#     - install python deps into a venv on /home, not into the image
#
#   Net effect: zero new bytes written to `/`.
#
# Usage:
#   ./run_docker.sh                 # interactive shell on the first free GPU
#   ./run_docker.sh 3               # interactive shell, pin GPU 3
#   ./run_docker.sh 3,7 "python rooftop/train.py --epochs 50"   # run a command
#
set -euo pipefail

BTP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="nvcr.io/nvidia/pytorch:24.05-py3"
# Deps live in a PYTHONUSERBASE dir on /home rather than a venv: the nvcr image
# ships no python3-venv (ensurepip), and we cannot apt-install it while running as
# the host uid. `pip install --user` layers cleanly on top of the image's system
# site-packages, which is exactly the "inherit torch, add the rest" we want.
# This node has 80 cores and 13 users. Torch defaults OMP_NUM_THREADS to the full
# core count, so EVERY process spawns 80 CPU threads for intra-op work; two of our
# training runs alone drove load average to 292 (3.7x oversubscribed) and slowed
# each epoch ~8x. Cap it. Override per-run with OMP_THREADS=n ./run_docker.sh ...
: "${OMP_THREADS:=4}"

PYDEPS="${BTP_ROOT}/.pydeps"         # on /home, survives container restarts
TMPDIR_HOST="${BTP_ROOT}/.tmp"       # container /tmp -> /home, because host /tmp is full

GPUS="${1:-}"
CMD="${2:-bash}"

# ---- pick a GPU if the caller didn't ----------------------------------------
if [[ -z "${GPUS}" ]]; then
  GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')"
  echo "[run_docker] no GPU given; picked least-used device: ${GPUS}"
fi

mkdir -p "${TMPDIR_HOST}"

# ---- data parked on host /tmp -----------------------------------------------
# The 40 GB /home quota is the binding constraint; host /tmp is on / (18 GB free)
# and outside the quota. Large REGENERABLE datasets live there and are symlinked
# from data/. Those symlinks work on the host but NOT in the container, because
# container /tmp is bind-mounted to .tmp on /home -- so mount each parked dataset
# directly over its expected path instead. Only regenerable data goes here: host
# /tmp is shared and has no retention guarantee.
DATA_MOUNTS=()
if [[ -d /tmp/btp_data ]]; then
  for _d in /tmp/btp_data/*/; do
    _n="$(basename "${_d%/}")"
    DATA_MOUNTS+=(-v "${_d%/}":"/workspace/data/${_n}")
  done
  [[ ${#DATA_MOUNTS[@]} -gt 0 ]] && \
    echo "[run_docker] parked data: ${#DATA_MOUNTS[@]} dataset(s) from /tmp/btp_data"
fi

# ---- ownership -------------------------------------------------------------
# Running as root inside the container makes every file it writes to the bind
# mount root:root on the host, and this user has no sudo to undo that. So run as
# the host uid:gid. Trade-off: you cannot apt-get inside the container this way.
# If you need apt, drop --user, then fix ownership afterwards with:
#   docker run --rm -v "${BTP_ROOT}":/w ubuntu:22.04 chown -R 1205:1206 /w
UIDGID="$(id -u):$(id -g)"

echo "[run_docker] image  : ${IMAGE}"
echo "[run_docker] gpus   : ${GPUS}"
echo "[run_docker] pydeps : ${PYDEPS}"
echo "[run_docker] as     : ${UIDGID}"

# -it only when we actually have a terminal, so the script also works from cron,
# `screen -dm`, and non-interactive agents.
# (declared with a dummy element under `set -u`, which treats an empty array
# expansion as unbound on bash < 4.4)
TTY_FLAGS=(--rm); [[ -t 0 && -t 1 ]] && TTY_FLAGS=(--rm -it)

exec docker run "${TTY_FLAGS[@]}" \
  --gpus "\"device=${GPUS}\"" \
  --user "${UIDGID}" \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=16g \
  -v "${BTP_ROOT}":/workspace \
  "${DATA_MOUNTS[@]+"${DATA_MOUNTS[@]}"}" \
  -v "${TMPDIR_HOST}":/tmp \
  -e HOME=/workspace \
  -e TMPDIR=/tmp \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -e OMP_NUM_THREADS="${OMP_THREADS:-4}" \
  -e MKL_NUM_THREADS="${OMP_THREADS:-4}" \
  -e OPENBLAS_NUM_THREADS="${OMP_THREADS:-4}" \
  -e PYTHONUSERBASE=/workspace/.pydeps \
  -e PATH="/workspace/.pydeps/bin:/usr/local/bin:/usr/local/nvidia/bin:/usr/bin:/bin" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc '
    STAMP=/workspace/.pydeps/.installed
    if [[ ! -f "${STAMP}" ]]; then
      echo "[run_docker] first run: installing deps into /workspace/.pydeps (on /home)"
      pip install --user --no-cache-dir -r /workspace/requirements-docker.txt \
        && touch "${STAMP}"
    fi
    exec '"${CMD}"'
  '
