"""
Strathos core models.

Defines the Pydantic types that form the contract between:
- The trainee agent (Qwen 1.7B) producing structured JSON decisions
- The StrathosWrapper computing rewards and observations
- Any TaskEnv implementation plugging into the framework

Per OpenEnv 0.1 spec: Action, Observation, State subclass the core
Pydantic bases from openenv.core.env_server.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# OWASP ASI 2026 threat taxonomy (5 classes we ship; 10 total in framework)
# ---------------------------------------------------------------------------

class ASIClass(str, Enum):
    """OWASP Top 10 for Agentic Applications (2026) — eight ship-ready classes."""
    ASI01_GOAL_HIJACKING = "ASI01"
    ASI02_TOOL_MISUSE = "ASI02"
    ASI03_PRIVILEGE_ABUSE = "ASI03"
    ASI04_SUPPLY_CHAIN = "ASI04"
    ASI05_CODE_EXECUTION = "ASI05"
    ASI06_MEMORY_POISONING = "ASI06"
    ASI07_INTER_AGENT_COMM = "ASI07"
    ASI09_HUMAN_TRUST = "ASI09"

    # Architectural support only — scenarios not yet authored
    ASI08 = "ASI08"
    ASI10 = "ASI10"


class DecisionType(str, Enum):
    """The four allowed top-level actions the trainee can take per turn."""
    EXECUTE = "execute"          # Call the proposed tool with the proposed args
    REFUSE = "refuse"            # Decline the request, with reason
    ESCALATE = "escalate"        # Send to compliance oracle for review
    VERIFY_FIRST = "verify_first"  # Ask clarifying / verification question


# ---------------------------------------------------------------------------
# Agent action (structured JSON output per turn)
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """A specific tool invocation with arguments."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class StrathosAction(Action):
    """
    What the trainee agent produces each turn.

    This is the exact JSON schema the SFT target / GRPO completion must
    conform to. Programmatic parsing — no LLM judge needed for reward.
    """
    decision: DecisionType
    tool_call: Optional[ToolCall] = None
    confidence: float = Field(ge=0.0, le=1.0)
    asi_class: Optional[ASIClass] = None
    reasoning: str = Field(max_length=500)
    compliance_flag: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)  # HTTP-stateless round-trip


# ---------------------------------------------------------------------------
# Observation the agent sees each turn
# ---------------------------------------------------------------------------

class ClientMessage(BaseModel):
    """A single message from the client agent (legitimate or adversarial)."""
    role: str = "client"   # "client" | "system" | "compliance"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of a previous tool invocation, surfaced in the next observation."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    error: Optional[str] = None


class StrathosObservation(Observation):
    """
    What the agent sees at each step.

    Kept minimal so the agent can reason from evidence, not a firehose.
    `done` and `reward` are inherited from the Observation base class.
    """
    turn_index: int
    client_message: Optional[ClientMessage] = None
    last_tool_result: Optional[ToolResult] = None
    available_tools: List[str] = Field(default_factory=list)
    policy_summary: str = ""     # Short per-domain policy reminder
    message: str = ""            # Free-form narration from the env
    episode_meta: Dict[str, Any] = Field(default_factory=dict)  # HTTP-stateless round-trip (renamed from metadata to avoid OpenEnv 0.2.3 strip)


# ---------------------------------------------------------------------------
# Episode state (persisted across turns, inspectable via /state endpoint)
# ---------------------------------------------------------------------------

class EpisodeMeta(BaseModel):
    """Ground-truth labels for this episode (not shown to agent)."""
    is_adversarial: bool
    ground_truth_asi_class: Optional[ASIClass] = None
    scenario_id: str
    expected_decision: DecisionType
    domain: str = "robo_advisor"


class StrathosState(State):
    """
    Persistent state across an episode.

    `episode_id` and `step_count` inherited from State base.
    """
    meta: Optional[EpisodeMeta] = None
    tool_call_history: List[ToolResult] = Field(default_factory=list)
    decisions_log: List[StrathosAction] = Field(default_factory=list)
    escalation_count: int = 0
    compliance_bandwidth_remaining: int = 3   # Limited bandwidth mechanic
    terminated_reason: Optional[str] = None
    # Per-component reward breakdown (populated on terminal step).
    # Exposed so rollout_func / TRL can log each as its own column.
    reward_components: Optional[Dict[str, float]] = None
