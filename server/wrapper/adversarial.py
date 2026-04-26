"""
Load OWASP ASI adversarial scenarios from YAML files.

Curriculum learning (per hackathon guide Section 6): start training on
lower-difficulty scenarios so the policy sees successful trajectories,
then ramp to harder classes as reward rises.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..models import ASIClass, DecisionType


# Default curriculum difficulty by ASI class (1=easy, 4=hard).
CURRICULUM_DIFFICULTY_BY_CLASS: Dict[str, int] = {
    "ASI01": 1,  # Goal hijacking — direct pressure, easiest
    "ASI02": 2,  # Tool misuse — requires tool-semantics understanding
    "ASI03": 3,  # Privilege abuse — subtle scope confusion
    "ASI06": 4,  # Memory poisoning — structured prompt injection
    "ASI07": 4,  # Inter-agent — spoofed peer messages
    "ASI04": 3,
    "ASI05": 3,
    "ASI08": 3,
    "ASI09": 3,
    "ASI10": 3,
}


def _scenarios_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scenarios"


def _assign_difficulty(scenario: Dict[str, Any]) -> int:
    if "difficulty" in scenario and isinstance(scenario["difficulty"], int):
        return scenario["difficulty"]
    return CURRICULUM_DIFFICULTY_BY_CLASS.get(scenario.get("asi_class", ""), 3)


def load_adversarial_scenarios(
    domain: Optional[str] = None,
    asi_classes: Optional[List[str]] = None,
    max_difficulty: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load adversarial scenarios.

    Args:
        domain: filter by target domain (e.g. "robo_advisor").
        asi_classes: optional list of ASI class codes to include.
        max_difficulty: curriculum cap. If set, only return scenarios with
            difficulty <= this value. Use max_difficulty=1 at the start of
            training, then ramp to 2, 3, 4 as reward rises.
    """
    scenarios: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(_scenarios_dir() / "*.yaml"))):
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or []
        for s in loaded:
            if domain is not None and s.get("domain") != domain:
                continue
            if asi_classes is not None and s.get("asi_class") not in asi_classes:
                continue
            diff = _assign_difficulty(s)
            s["difficulty"] = diff
            if max_difficulty is not None and diff > max_difficulty:
                continue
            scenarios.append(s)
    return scenarios


def parse_decision(raw: str) -> DecisionType:
    return DecisionType(raw)


def parse_asi_class(raw: str) -> ASIClass:
    return ASIClass(raw)
