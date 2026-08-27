from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recover_pi05_metrics_from_console", ROOT / "scripts" / "recover_pi05_metrics_from_console.py"
)
assert SPEC is not None and SPEC.loader is not None
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def test_parser_handles_tqdm_timestamp_concatenated_to_param_norm() -> None:
    text = (
        "Step 0: grad_norm=8.0426, loss=1.1714, param_norm=1938.4993\n"
        "Step 1: grad_norm=6.8419, loss=1.6156, param_norm=1938.499523:17:15.468 [I] Progress\n"
    )

    metrics = recovery.parse_console_metrics(text)

    assert metrics == [
        {"step": 0, "loss": 1.1714, "grad_norm": 8.0426, "param_norm": 1938.4993},
        {"step": 1, "loss": 1.6156, "grad_norm": 6.8419, "param_norm": 1938.4995},
    ]


def test_cli_writes_appendable_jsonl_snapshot(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "Step 3: grad_norm=2.0000, loss=1.0000, param_norm=4.0000\n",
        encoding="utf-8",
    )
    output = tmp_path / "train_metrics.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_pi05_metrics_from_console.py",
            "--log",
            str(log),
            "--output",
            str(output),
            "--experiment",
            "stage1-lora",
            "--config",
            "pi05_lora",
            "--mode",
            "lora",
            "--dataset-repo",
            "owner/data",
            "--num-train-steps",
            "5000",
        ],
    )

    assert recovery.main() == 0
    [record] = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert record["step"] == 3
    assert record["recovered_from_console"] is True
    assert len(record["source_log_sha256"]) == 64
