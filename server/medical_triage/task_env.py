"""
Medical Triage stub — demonstrates the Strathos extension pattern.

This file is INTENTIONALLY a stub. It exists to show how another domain
plugs into Strathos via the TaskEnv interface. Not fully implemented in V1.

To implement: fill in medical tools, HIPAA/clinical policy rules, and
legitimate triage scenarios. The same StrathosWrapper + OWASP ASI
scenario library works — just swap the TaskEnv at construction time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import DecisionType, EpisodeMeta, StrathosAction
from ..task_env import PolicyRule, TaskEnv, ToolSpec


class MedicalTriageTaskEnv(TaskEnv):
    """STUB. Extension pattern demonstration only."""

    @property
    def domain_name(self) -> str:
        return "medical_triage"

    @property
    def tools(self) -> List[ToolSpec]:
        # TODO: implement medical tools (search_ehr, check_allergies,
        # order_labs, prescribe, refer, escalate_to_attending, ...)
        return []

    @property
    def policy_rules(self) -> List[PolicyRule]:
        # TODO: HIPAA, drug interaction, standard-of-care checks
        return []

    @property
    def policy_summary(self) -> str:
        return (
            "Medical triage agent stub. When implemented, policy summary "
            "would describe HIPAA boundaries, clinical guidelines, "
            "escalation criteria, and duty-of-care obligations."
        )

    def sample_legitimate_scenario(
        self, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "MedicalTriageTaskEnv is a stub. Implement tools, policies, "
            "and scenarios to use this domain with Strathos."
        )

    def evaluate_legitimate_response(
        self,
        action: StrathosAction,
        context: Dict[str, Any],
        meta: EpisodeMeta,
    ) -> bool:
        raise NotImplementedError("See class docstring.")
