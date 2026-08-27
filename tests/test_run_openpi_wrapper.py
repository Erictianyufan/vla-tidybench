from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_openpi_exports_the_resolved_openpi_root() -> None:
    wrapper = (ROOT / "scripts" / "run_openpi.sh").read_text(encoding="utf-8")
    assert 'OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/openpi}"' in wrapper
    assert "export OPENPI_ROOT VLA_TIDYBENCH_DATA" in wrapper


def test_norm_stats_are_prepared_for_each_training_mode() -> None:
    wrapper = (ROOT / "scripts" / "compute_drawer_norm_stats.py").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert 'parser.add_argument("--mode", choices=("lora", "expert", "full"), default="lora")' in wrapper
    assert "finetune_mode=args.mode" in wrapper
    target = makefile.split("pi05-prepare-norm-stats:", 1)[1].split("pi05-formal-prepare:", 1)[0]
    assert target.count("--mode lora") == 1
    assert target.count("--mode full") == 2
