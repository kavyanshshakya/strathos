"""
Patched StrathosEnvironment for HTTP-stateless deployment.

Key change: OpenEnv 0.2.3's HTTP /reset and /step endpoints create a NEW
env instance per request (env.close() is called after each call). State
does NOT persist across HTTP calls.

To survive this, we:
1. Embed scenario_id + is_adversarial + ground_truth_asi_class in the
   StrathosObservation.metadata returned by reset().
2. The client (inference.py / TRL rollout) MUST round-trip those values
   back via StrathosAction.metadata on every /step call.
3. step() reads scenario meta from action.metadata and looks up the
   scenario by id to reconstruct context.

This matches DocForge's stateless pattern exactly.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server import Environment

from ..models import (
    ASIClass,
    ClientMessage,
    DecisionType,
    EpisodeMeta,
    StrathosAction,
    StrathosObservation,
    StrathosState,
    ToolResult,
)
from ..task_env import TaskEnv
from .adversarial import load_adversarial_scenarios, parse_asi_class, parse_decision
from .reward import compute_reward


DEFAULT_ADVERSARIAL_RATIO = 0.30
DEFAULT_MAX_TURNS = 6
DEFAULT_MAX_DIFFICULTY = None


class StrathosEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        task_env: TaskEnv,
        adversarial_ratio: float = DEFAULT_ADVERSARIAL_RATIO,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_difficulty: Optional[int] = DEFAULT_MAX_DIFFICULTY,
        seed: Optional[int] = None,
    ):
        self.task_env = task_env
        self.adversarial_ratio = adversarial_ratio
        self.max_turns = max_turns
        self.max_difficulty = max_difficulty
        self._rng = random.Random(seed)

        self._adv_scenarios = load_adversarial_scenarios(
            domain=task_env.domain_name,
            max_difficulty=max_difficulty,
        )
        # Index by scenario_id for fast lookup in step()
        self._adv_by_id: Dict[str, Dict[str, Any]] = {
            s["id"]: s for s in self._adv_scenarios
        }

        self._state = StrathosState()
        self._context: Dict[str, Any] = {}
        self._optimal_tools: list = []  # populated per-episode in reset()

    def set_curriculum_difficulty(self, max_difficulty: Optional[int]) -> None:
        self.max_difficulty = max_difficulty
        self._adv_scenarios = load_adversarial_scenarios(
            domain=self.task_env.domain_name,
            max_difficulty=max_difficulty,
        )
        self._adv_by_id = {s["id"]: s for s in self._adv_scenarios}

    def _build_meta_from_id(self, scenario_id: str, is_adversarial: bool) -> EpisodeMeta:
        """Reconstruct EpisodeMeta from a scenario_id (used in step())."""
        if is_adversarial:
            scn = self._adv_by_id.get(scenario_id)
            if scn is None:
                # Defensive: unknown id — return a stub meta for adv episodes
                return EpisodeMeta(
                    is_adversarial=True,
                    ground_truth_asi_class=None,
                    scenario_id=scenario_id,
                    expected_decision=DecisionType.REFUSE,
                    domain=self.task_env.domain_name,
                )
            return EpisodeMeta(
                is_adversarial=True,
                ground_truth_asi_class=parse_asi_class(scn["asi_class"]),
                scenario_id=scenario_id,
                expected_decision=parse_decision(scn["expected_decision"]),
                domain=self.task_env.domain_name,
            )
        else:
            return EpisodeMeta(
                is_adversarial=False,
                ground_truth_asi_class=None,
                scenario_id=scenario_id,
                expected_decision=DecisionType.EXECUTE,
                domain=self.task_env.domain_name,
            )

    def reset(self, seed=None, episode_id=None, **kwargs) -> StrathosObservation:
        if seed is not None:
            self._rng = random.Random(seed)

        is_adversarial = (
            self._rng.random() < self.adversarial_ratio
            and len(self._adv_scenarios) > 0
        )

        if is_adversarial:
            scn = self._rng.choice(self._adv_scenarios)
            meta = EpisodeMeta(
                is_adversarial=True,
                ground_truth_asi_class=parse_asi_class(scn["asi_class"]),
                scenario_id=scn["id"],
                expected_decision=parse_decision(scn["expected_decision"]),
                domain=self.task_env.domain_name,
            )
            client_opener = scn["client_opener"]
            self._context = {
                "account_id": "acct_" + uuid.uuid4().hex[:8],
                **scn.get("context", {}),
            }
            # NEW: capture optimal_tools for multi-step investigation reward
            self._optimal_tools = scn.get("optimal_tools", [])
        else:
            scn = self.task_env.sample_legitimate_scenario(
                seed=self._rng.randrange(2**31)
            )
            meta = EpisodeMeta(
                is_adversarial=False,
                ground_truth_asi_class=None,
                scenario_id=scn["scenario_id"],
                expected_decision=scn["expected_decision"],
                domain=self.task_env.domain_name,
            )
            client_opener = scn["client_opener"]
            self._context = scn["context"]
            # No optimal tools for legitimate scenarios — fast correct execution rewarded
            self._optimal_tools = []

        self._state = StrathosState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            meta=meta,
            tool_call_history=[],
            decisions_log=[],
            escalation_count=0,
            compliance_bandwidth_remaining=3,
            terminated_reason=None,
        )

        # Embed meta in the observation so the client can round-trip it.
        # OpenEnv 0.2.3 HTTP path creates fresh env per request, so we can't
        # rely on self._state surviving between /reset and /step calls.
        return StrathosObservation(
            done=False,
            reward=None,
            turn_index=0,
            client_message=ClientMessage(role="client", content=client_opener),
            last_tool_result=None,
            available_tools=[t.name for t in self.task_env.tools],
            policy_summary=self.task_env.policy_summary,
            message="Episode started. Respond with a structured JSON decision.",
            episode_meta={
                "scenario_id": meta.scenario_id,
                "is_adversarial": meta.is_adversarial,
                "asi_class_hint": meta.ground_truth_asi_class.value if meta.ground_truth_asi_class else None,
                "episode_id": self._state.episode_id,
                "domain": meta.domain,
            },
        )

    def step(
        self, action: StrathosAction, timeout_s=None, **kwargs
    ) -> StrathosObservation:
        # HTTP-stateless reconstruction: read scenario_id from action.metadata
        # if our own _state.meta got nulled by a fresh env instance.
        meta = self._state.meta
        if meta is None:
            scenario_id = action.metadata.get("scenario_id") if action.metadata else None
            is_adv = action.metadata.get("is_adversarial", False) if action.metadata else False
            if scenario_id:
                meta = self._build_meta_from_id(scenario_id, is_adv)
                # Reconstruct minimal context for tool calls
                if is_adv and scenario_id in self._adv_by_id:
                    self._context = {
                        "account_id": "acct_" + uuid.uuid4().hex[:8],
                        **self._adv_by_id[scenario_id].get("context", {}),
                    }
                    # NEW: rehydrate optimal_tools too
                    self._optimal_tools = self._adv_by_id[scenario_id].get("optimal_tools", [])
                else:
                    # Legitimate or unknown — use a default context
                    self._context = {"account_id": "acct_" + uuid.uuid4().hex[:8]}
                    self._optimal_tools = []
                # Attach the recovered meta to state so reward computation works
                self._state.meta = meta
            else:
                # No scenario hint — assume legitimate single-turn fallback
                meta = EpisodeMeta(
                    is_adversarial=False,
                    ground_truth_asi_class=None,
                    scenario_id="unknown",
                    expected_decision=DecisionType.EXECUTE,
                    domain=self.task_env.domain_name,
                )
                self._state.meta = meta
                self._context = {"account_id": "acct_" + uuid.uuid4().hex[:8]}
                self._optimal_tools = []

        self._state.step_count += 1
        self._state.decisions_log.append(action)

        # --- Side effects based on decision type --------------------------
        last_tool_result: Optional[ToolResult] = None

        if action.decision == DecisionType.EXECUTE and action.tool_call is not None:
            last_tool_result = self.task_env.call_tool(
                action.tool_call.name,
                action.tool_call.arguments,
                self._context,
            )
            self._state.tool_call_history.append(last_tool_result)

        elif action.decision == DecisionType.ESCALATE:
            self._state.escalation_count += 1
            self._state.compliance_bandwidth_remaining = max(
                0, self._state.compliance_bandwidth_remaining - 1
            )

        # --- Terminal condition detection ---------------------------------
        done = False
        terminated_reason = None

        if action.decision in {DecisionType.REFUSE, DecisionType.ESCALATE, DecisionType.VERIFY_FIRST}:
            done = True
            terminated_reason = f"agent_{action.decision.value}"
        elif self._state.step_count >= self.max_turns:
            done = True
            terminated_reason = "max_turns_reached"
        elif (
            action.decision == DecisionType.EXECUTE
            and action.tool_call
            and action.tool_call.name == "execute_trade"
        ):
            done = True
            terminated_reason = "trade_executed"

        # --- Compute reward iff terminal ----------------------------------
        reward: Optional[float] = None
        if done:
            legit_ok = True
            if not meta.is_adversarial:
                legit_ok = self.task_env.evaluate_legitimate_response(
                    action, self._context, meta
                )
            reward, components = compute_reward(
                action, self._context, meta,
                legitimate_eval=legit_ok,
                tool_call_history=self._state.tool_call_history,
                optimal_tools=self._optimal_tools,
            )
            self._state.reward_components = components

        self._state.terminated_reason = terminated_reason

        return StrathosObservation(
            done=done,
            reward=reward,
            turn_index=self._state.step_count,
            client_message=None,
            last_tool_result=last_tool_result,
            available_tools=[t.name for t in self.task_env.tools],
            policy_summary=self.task_env.policy_summary,
            message=terminated_reason or "Continue. Next decision?",
            episode_meta={
                "scenario_id": meta.scenario_id,
                "is_adversarial": meta.is_adversarial,
                "asi_class_hint": meta.ground_truth_asi_class.value if meta.ground_truth_asi_class else None,
                "episode_id": self._state.episode_id,
                "domain": meta.domain,
            },
        )

    @property
    def state(self) -> StrathosState:
        return self._state
