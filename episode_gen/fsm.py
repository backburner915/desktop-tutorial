"""Shared FSM phase definitions.

One state machine for every scenario in the taxonomy (N/R/F/P). A new
scenario ID must never require a new phase — only a new config and,
if genuinely novel, a new detector/recovery function.
"""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    RESET = "RESET"
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    CONTACT_CHECK = "CONTACT_CHECK"
    GRASP = "GRASP"
    LIFT = "LIFT"
    HOLD = "HOLD"
    MOVE_TO_TARGET = "MOVE_TO_TARGET"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


# Phases after which the episode is over and the runner loop must stop.
TERMINAL_PHASES = frozenset({Phase.SUCCESS, Phase.FAILURE})

# Nominal forward order when no anomaly is detected. RETREAT is reached
# only via a detector/recovery decision, never from this table.
NOMINAL_NEXT = {
    Phase.RESET: Phase.PREGRASP,
    Phase.PREGRASP: Phase.APPROACH,
    Phase.APPROACH: Phase.CONTACT_CHECK,
    Phase.CONTACT_CHECK: Phase.GRASP,
    Phase.GRASP: Phase.LIFT,
    Phase.LIFT: Phase.HOLD,
    Phase.HOLD: Phase.MOVE_TO_TARGET,
    Phase.MOVE_TO_TARGET: Phase.PLACE,
    Phase.PLACE: Phase.RELEASE,
    Phase.RELEASE: Phase.SUCCESS,
}
