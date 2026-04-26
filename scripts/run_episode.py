"""
scripts/run_episode.py

Quickly runs one Strathos episode (legitimate or adversarial) with a
hardcoded policy to sanity-check the full env/reward loop. Useful before
we wire up real LLM rollouts.

Usage:
    python scripts/run_episode.py              # random (70/30 split)
    python scripts/run_episode.py --adv        # force adversarial
    python scripts/run_episode.py --legit      # force legitimate
    python scripts/run_episode.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys

from server.models import ASIClass, DecisionType, StrathosAction, ToolCall
from server.robo_advisor import RoboAdvisorTaskEnv
from server.wrapper import StrathosEnvironment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--adv", action="store_true", help="Force adversarial episode")
    ap.add_argument("--legit", action="store_true", help="Force legitimate episode")
    args = ap.parse_args()

    if args.adv and args.legit:
        print("Pick one: --adv OR --legit")
        sys.exit(1)

    ratio = 1.0 if args.adv else 0.0 if args.legit else 0.3

    env = StrathosEnvironment(
        task_env=RoboAdvisorTaskEnv(),
        adversarial_ratio=ratio,
        max_turns=4,
        seed=args.seed,
    )

    obs = env.reset(seed=args.seed)
    meta = env.state.meta
    print("=" * 70)
    print(f"Episode: {env.state.episode_id}")
    print(f"Adversarial: {meta.is_adversarial}  Scenario: {meta.scenario_id}")
    if meta.is_adversarial:
        print(f"Ground truth ASI class: {meta.ground_truth_asi_class}")
    print(f"Expected decision: {meta.expected_decision}")
    print("-" * 70)
    print("Client opener:")
    print(f"  {obs.client_message.content}")
    print("-" * 70)

    # Dumb hardcoded policy: always refuse adversarial, always query_portfolio on legit
    if meta.is_adversarial:
        action = StrathosAction(
            decision=DecisionType.REFUSE,
            confidence=0.85,
            asi_class=meta.ground_truth_asi_class,   # cheat: use ground truth to verify reward wiring
            reasoning=(
                "The request pattern matches a known OWASP ASI threat class. "
                "Refusing per Reg BI and flagging concern."
            ),
            compliance_flag=True,
        )
    else:
        action = StrathosAction(
            decision=DecisionType.EXECUTE,
            tool_call=ToolCall(name="query_portfolio", arguments={}),
            confidence=0.9,
            asi_class=None,
            reasoning=(
                "Legitimate request. Reading portfolio first to confirm "
                "account state before considering any suitability risk."
            ),
            compliance_flag=False,
        )

    obs = env.step(action)
    print("Agent decision:")
    print(json.dumps(action.model_dump(), indent=2, default=str))
    print("-" * 70)
    print(f"done={obs.done}  reward={obs.reward}  reason={env.state.terminated_reason}")
    print(f"last tool result: {obs.last_tool_result}")
    print("=" * 70)


if __name__ == "__main__":
    main()
