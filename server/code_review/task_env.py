"""
Code Review stub — demonstrates the Strathos extension pattern.

Not fully implemented in V1. Shows how a code review agent (with tools
for reading diffs, running tests, checking CVEs, approving PRs) would
plug into the same StrathosWrapper adversarial layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import DecisionType, EpisodeMeta, StrathosAction
from ..task_env import PolicyRule, TaskEnv, ToolSpec


class CodeReviewTaskEnv(TaskEnv):
    """STUB. Extension pattern demonstration only."""

    @property
    def domain_name(self) -> str:
        return "code_review"

    @property
    def tools(self) -> List[ToolSpec]:
        # TODO: read_diff, run_tests, check_dependencies, lint, approve_pr, request_changes
        return []

    @property
    def policy_rules(self) -> List[PolicyRule]:
        # TODO: security review, standards compliance, dependency safety
        return []

    @property
    def policy_summary(self) -> str:
        return (
            "Code review agent stub. When implemented, policy summary would "
            "describe security review criteria, CI requirements, and "
            "organization coding standards."
        )

    def sample_legitimate_scenario(
        self, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError("CodeReviewTaskEnv is a stub.")

    def evaluate_legitimate_response(
        self,
        action: StrathosAction,
        context: Dict[str, Any],
        meta: EpisodeMeta,
    ) -> bool:
        raise NotImplementedError("See class docstring.")
