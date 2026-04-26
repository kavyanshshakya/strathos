"""Strathos training utilities — TRL rollout, SFT prep, etc."""

from .rollout import (
    build_prompt,
    make_rollout_func,
    parse_action_from_completion,
    strathos_reward_fns,
)

__all__ = [
    "build_prompt",
    "make_rollout_func",
    "parse_action_from_completion",
    "strathos_reward_fns",
]
