"""Deterministic long-horizon task orchestration for the drawer benchmark."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class DrawerSkill(enum.StrEnum):
    OPEN = "open"
    PICK = "pick"
    PLACE = "place"
    CLOSE = "close"
    DONE = "done"
    FAILED = "failed"


_NEXT = {
    DrawerSkill.OPEN: DrawerSkill.PICK,
    DrawerSkill.PICK: DrawerSkill.PLACE,
    DrawerSkill.PLACE: DrawerSkill.CLOSE,
    DrawerSkill.CLOSE: DrawerSkill.DONE,
}


@dataclass
class DrawerTaskGraph:
    """Advance only on audited skill success; fail closed on any skill error."""

    current: DrawerSkill = DrawerSkill.OPEN

    @property
    def prompt(self) -> str:
        prompts = {
            DrawerSkill.OPEN: "open the top drawer",
            DrawerSkill.PICK: "pick up the tomato soup can",
            DrawerSkill.PLACE: "put the tomato soup can into the top drawer",
            DrawerSkill.CLOSE: "close the top drawer",
        }
        if self.current not in prompts:
            raise RuntimeError(f"terminal graph state has no prompt: {self.current.value}")
        return prompts[self.current]

    def update(self, *, succeeded: bool, exhausted: bool = False) -> DrawerSkill:
        if self.current in (DrawerSkill.DONE, DrawerSkill.FAILED):
            return self.current
        if succeeded:
            self.current = _NEXT[self.current]
        elif exhausted:
            self.current = DrawerSkill.FAILED
        return self.current

    def reset(self) -> None:
        self.current = DrawerSkill.OPEN
