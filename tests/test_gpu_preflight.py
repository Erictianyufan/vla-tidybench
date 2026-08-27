from __future__ import annotations

import subprocess

import pytest
from vla_tidybench.openpi import gpu_preflight


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_selected_gpu_indices_are_strict_and_ordered() -> None:
    assert gpu_preflight.selected_gpu_indices("2,0,1") == (2, 0, 1)
    assert gpu_preflight.selected_gpu_indices("") == ()
    with pytest.raises(ValueError, match="unique"):
        gpu_preflight.selected_gpu_indices("0,0")
    with pytest.raises(ValueError, match="numeric"):
        gpu_preflight.selected_gpu_indices("GPU-uuid")


def test_inspection_maps_compute_processes_to_physical_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            completed("0, GPU-a, 12\n1, GPU-b, 34\n"),
            completed("GPU-b, 1234\nGPU-b, 5678\n"),
        ]
    )
    monkeypatch.setattr(gpu_preflight.subprocess, "run", lambda *_args, **_kwargs: next(responses))

    usage = gpu_preflight.inspect_gpu_usage()

    assert usage[0].memory_used_mib == 12
    assert usage[0].compute_pids == ()
    assert usage[1].compute_pids == (1234, 5678)


def test_wait_retries_busy_gpu_then_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = iter(
        [
            {0: gpu_preflight.GPUUsage(0, "GPU-a", 900, (1234,))},
            {0: gpu_preflight.GPUUsage(0, "GPU-a", 10, ())},
        ]
    )
    clock = iter((0.0, 1.0))
    monkeypatch.setattr(gpu_preflight, "inspect_gpu_usage", lambda: next(snapshots))
    monkeypatch.setattr(gpu_preflight.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(gpu_preflight.time, "sleep", lambda _seconds: None)

    ready = gpu_preflight.wait_for_exclusive_gpus((0,), timeout_s=10, poll_s=1)

    assert ready[0].compute_pids == ()


def test_wait_times_out_with_occupying_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = {0: gpu_preflight.GPUUsage(0, "GPU-a", 900, (1234,))}
    clock = iter((0.0, 11.0))
    monkeypatch.setattr(gpu_preflight, "inspect_gpu_usage", lambda: usage)
    monkeypatch.setattr(gpu_preflight.time, "monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="1234"):
        gpu_preflight.wait_for_exclusive_gpus((0,), timeout_s=10)
