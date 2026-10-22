#!/usr/bin/env bash
# Run one uncut home-tabletop OPEN -> PICK -> PLACE -> CLOSE pi0.5-first demo.
# Two local GPUs are expected unless POLICY_HOST points at an already-running
# OpenPI policy server. A failed collector or provenance audit never publishes
# a video.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8000}"
ISAAC_GPU="${ISAAC_GPU:-0}"
POLICY_GPU="${POLICY_GPU:-1}"
ISAAC_PYTHON="${ISAAC_LAB_VENV:-/home/ubuntu/env_isaaclab}/bin/python"
RUN_NAME="home_full_pi05_$(date +%Y%m%d_%H%M%S)"
DATASET="${DATASET:-/home/ubuntu/data/vla-tidybench/eval/${RUN_NAME}.hdf5}"
VIDEO="${VIDEO:-/home/ubuntu/data/vla-tidybench/videos/${RUN_NAME}.mp4}"
CORRECTION_LIMIT="${DLS_MAX_CORRECTION_NORM:-0.20}"
CORRECTION_MODE="${DLS_CORRECTION_MODE:-skill_router}"
GRIPPER_OPEN_CONFIRM_STEPS="${POLICY_GRIPPER_OPEN_CONFIRM_STEPS:-4}"
PROMPT_ROUTE_ARGS=()
if [[ "${ROUTE_PLACE_RELEASE_PROMPT:-0}" == "1" ]]; then
  PROMPT_ROUTE_ARGS+=(--route-place-release-prompt)
fi
SERVER_PID=""

if [[ "${POLICY_HOST}" == "127.0.0.1" || "${POLICY_HOST}" == "localhost" ]]; then
  : "${PI05_CHECKPOINT:?Set PI05_CHECKPOINT to the four-skill pi0.5 LoRA checkpoint directory}"
  if [[ ! -d "${PI05_CHECKPOINT}" ]]; then
    echo "Missing checkpoint directory: ${PI05_CHECKPOINT}" >&2
    exit 2
  fi
  GPU_COUNT="$(nvidia-smi -L | wc -l)"
  if (( GPU_COUNT < 2 )); then
    echo "Need two GPUs for co-located Isaac RTX and pi0.5, or set POLICY_HOST to an external server." >&2
    exit 2
  fi
  mkdir -p "$(dirname "${DATASET}")" "$(dirname "${VIDEO}")"
  OPENPI_CUDA_VISIBLE_DEVICES="${POLICY_GPU}" ./scripts/run_openpi.sh scripts/serve_drawer_policy.py \
    --checkpoint "${PI05_CHECKPOINT}" --host 127.0.0.1 --port "${POLICY_PORT}" --four-skill \
    > "${VIDEO%.mp4}.policy-server.log" 2>&1 &
  SERVER_PID="$!"
  trap 'if [[ -n "${SERVER_PID}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi' EXIT
fi

echo "Waiting for pi0.5 policy server at ${POLICY_HOST}:${POLICY_PORT}"
for _ in $(seq 1 120); do
  if "${ISAAC_PYTHON}" -c \
    'import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),2); s.close()' \
    "${POLICY_HOST}" "${POLICY_PORT}" 2>/dev/null; then
    break
  fi
  if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "pi0.5 server exited; inspect ${VIDEO%.mp4}.policy-server.log" >&2
    exit 2
  fi
  sleep 5
done

"${ISAAC_PYTHON}" -c \
  'import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),2); s.close()' \
  "${POLICY_HOST}" "${POLICY_PORT}"

./scripts/run_isaac.sh scripts/collect_scripted_drawer.py \
  --skill full --home-tabletop --policy-first \
  --policy-host "${POLICY_HOST}" --policy-port "${POLICY_PORT}" \
  --dls-max-correction-norm "${CORRECTION_LIMIT}" --dls-correction-mode "${CORRECTION_MODE}" \
  --policy-replan-steps 4 --policy-gripper-close-confirm-steps 6 \
  --policy-gripper-open-confirm-steps "${GRIPPER_OPEN_CONFIRM_STEPS}" \
  "${PROMPT_ROUTE_ARGS[@]}" \
  --num_demos 1 --max_attempts 3 --max_steps 850 --seed 610 \
  --dataset_file "${DATASET}" --device "cuda:${ISAAC_GPU}" --enable_cameras --headless

"${ISAAC_PYTHON}" scripts/validate_home_full_pi05.py "${DATASET}"
"${ISAAC_PYTHON}" scripts/render_pi05_showcase.py "${DATASET}" "${VIDEO}" --fps 20
echo "Validated model-first full-task video: ${VIDEO}"
