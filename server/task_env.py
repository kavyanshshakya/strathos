"""
TaskEnv — abstract interface for domain-specific task environments.

This is the wildcard hook. ANY regulated domain (finance, medical, legal,
code review, customer service) can be plugged into Strathos by implementing
this interface. The StrathosWrapper then adds OWASP ASI adversarial
pressure on top of it, regardless of domain.

Our V1 ships RoboAdvisorTaskEnv as the concrete implementation.
Stub files for MedicalTriageTaskEnv and CodeReviewTaskEnv demonstrate
the extension pattern without implementing full logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .models import (
    ClientMessage,
    DecisionType,
    EpisodeMeta,
    StrathosAction,
    ToolResult,
)


class ToolSpec:
    """Lightweight tool schema spec (not a full Pydantic model per tool)."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, str],
        backend: Callable[..., Any],
        category: str = "general",
    ):
        self.name = name
        self.description = description
        self.parameters = parameters  # name -> type description
        self.backend = backend        # callable implementing the tool
        self.category = category


class PolicyRule:
    """
    A programmatic correctness check for the domain.

    Examples:
      - Finance: "executed trade must pass suitability check vs client risk profile"
      - Medical: "prescription must not contradict known allergies"
      - Legal: "document sharing must not violate privilege designation"
    """

    def __init__(
        self,
        name: str,
        description: str,
        check: Callable[[StrathosAction, Dict[str, Any]], bool],
    ):
        self.name = name
        self.description = description
        self.check = check


class TaskEnv(ABC):
    """
    Abstract interface that any domain-specific environment implements.

    Subclasses provide:
      - A tool ecosystem (ToolSpec list)
      - Policy rules (ground-truth correctness)
      - A scenario generator (legitimate requests)
      - An evaluator (did the agent's response satisfy the task?)

    The StrathosWrapper takes an instance of TaskEnv and layers
    OWASP ASI adversarial scenarios on top of the legitimate ones.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Short domain identifier, e.g. 'robo_advisor', 'medical_triage'."""

    @property
    @abstractmethod
    def tools(self) -> List[ToolSpec]:
        """The tool ecosystem available to the trainee agent in this domain."""

    @property
    @abstractmethod
    def policy_rules(self) -> List[PolicyRule]:
        """Domain policy rules used for ground-truth correctness scoring."""

    @property
    @abstractmethod
    def policy_summary(self) -> str:
        """Short summary shown to the agent as part of observation context."""

    # ---- Scenario generation ------------------------------------------------

    @abstractmethod
    def sample_legitimate_scenario(
        self, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Produce a legitimate client request + ground-truth expected decision.

        Returns a dict with at minimum:
          - scenario_id: str
          - client_opener: str   (first client message)
          - expected_decision: DecisionType
          - context: dict        (per-episode synthetic state — e.g. portfolio)
        """

    # ---- Tool dispatch ------------------------------------------------------

    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolResult:
        """
        Execute a named tool against the current episode context.

        Default implementation looks up the tool by name and calls its backend.
        Subclasses usually don't need to override this.
        """
        for tool in self.tools:
            if tool.name == name:
                try:
                    result = tool.backend(arguments, context)
                    return ToolResult(
                        tool_name=name, arguments=arguments, result=result
                    )
                except Exception as e:
                    return ToolResult(
                        tool_name=name,
                        arguments=arguments,
                        result=None,
                        error=str(e),
                    )
        return ToolResult(
            tool_name=name,
            arguments=arguments,
            result=None,
            error=f"Unknown tool: {name}",
        )

    # ---- Evaluation ---------------------------------------------------------

    @abstractmethod
    def evaluate_legitimate_response(
        self,
        action: StrathosAction,
        context: Dict[str, Any],
        meta: EpisodeMeta,
    ) -> bool:
        """
        For legitimate episodes: did the agent serve the request correctly
        per domain policy? Returns True iff the response satisfies all
        applicable policy rules.
        """
