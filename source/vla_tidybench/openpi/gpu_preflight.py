"""GPU exclusivity guard for memory-sensitive three-stage OpenPI training."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUUsage:
    index: int
    uuid: str
    memory_used_mib: int
    compute_pids: tuple[int, ...]


def selected_gpu_indices(value: str | None = None) -> tuple[int, ...]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "") if value is None else value
    if not value or value.strip() == "-1":
        return ()
    try:
        indices = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("GPU preflight requires numeric CUDA_VISIBLE_DEVICES indices") from error
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain unique non-negative indices")
    return indices


def inspect_gpu_usage() -> dict[int, GPUUsage]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows: dict[int, tuple[str, int]] = {}
    for line in gpu_result.stdout.splitlines():
        index, uuid, used = (part.strip() for part in line.split(",", maxsplit=2))
        rows[int(index)] = (uuid, int(used))

    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    pids_by_uuid: dict[str, list[int]] = {}
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid = (part.strip() for part in line.split(",", maxsplit=1))
        numeric_pid = int(pid)
        if numeric_pid != os.getpid():
            pids_by_uuid.setdefault(uuid, []).append(numeric_pid)
    return {
        index: GPUUsage(index, uuid, used, tuple(sorted(pids_by_uuid.get(uuid, []))))
        for index, (uuid, used) in rows.items()
    }


def wait_for_exclusive_gpus(
    indices: tuple[int, ...],
    *,
    max_used_mib: int = 512,
    timeout_s: float = 21_600.0,
    poll_s: float = 30.0,
) -> dict[int, GPUUsage]:
    """Wait until selected GPUs have no foreign compute process and low baseline memory."""

    if not indices:
        return {}
    if max_used_mib < 0 or timeout_s < 0 or poll_s <= 0:
        raise ValueError("GPU preflight limits must be non-negative and poll_s must be positive")
    started = time.monotonic()
    while True:
        usage = inspect_gpu_usage()
        missing = [index for index in indices if index not in usage]
        if missing:
            raise ValueError(f"selected GPUs are not reported by nvidia-smi: {missing}")
        busy = {
            index: item
            for index in indices
            if (item := usage[index]).memory_used_mib > max_used_mib or item.compute_pids
        }
        if not busy:
            print(
                "gpu_preflight ready "
                + " ".join(f"gpu={index} used_mib={usage[index].memory_used_mib}" for index in indices),
                flush=True,
            )
            return {index: usage[index] for index in indices}
        elapsed = time.monotonic() - started
        description = " ".join(
            f"gpu={index} used_mib={item.memory_used_mib} pids={list(item.compute_pids)}"
            for index, item in busy.items()
        )
        if elapsed >= timeout_s:
            raise TimeoutError(f"GPU preflight timed out after {elapsed:.0f}s: {description}")
        print(f"gpu_preflight waiting elapsed_s={elapsed:.0f} {description}", flush=True)
        time.sleep(min(poll_s, max(0.0, timeout_s - elapsed)))
