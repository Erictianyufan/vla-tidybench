from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_openpi_exports_the_resolved_openpi_root() -> None:
    wrapper = (ROOT / "scripts" / "run_openpi.sh").read_text(encoding="utf-8")
    assert 'OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/openpi}"' in wrapper
    assert "export OPENPI_ROOT VLA_TIDYBENCH_DATA" in wrapper
