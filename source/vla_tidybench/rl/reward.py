"""Auditable reward terms for an OPEN-drawer residual specialist."""

from __future__ import annotations


def open_drawer_reward(
    *, previous_q: float, current_q: float, contact: bool, collision: bool, residual_norm: float, success: bool
) -> dict[str, float]:
    terms = {
        "progress": 5.0 * (current_q - previous_q),
        "contact": 0.02 if contact else 0.0,
        "collision": -0.25 if collision else 0.0,
        "residual": -0.01 * residual_norm * residual_norm,
        "success": 2.0 if success else 0.0,
        "time": -0.002,
    }
    terms["total"] = sum(terms.values())
    return terms
