# Strathos Roadmap

This document is the honest forward-looking companion to [BLOG.md](BLOG.md).
BLOG explains what was built and why. ROADMAP explains what comes next and
what was deliberately scoped out.

## What shipped (v0.1, hackathon submission)

- OpenEnv 0.2.3 compliant environment with `/reset`, `/step`, `/health`, `/state`.
- 8 of 10 OWASP ASI 2026 attack classes (ASI01-07, ASI09).
- 15 hand-authored adversarial scenarios with hidden context and `optimal_tools` annotations.
- 7-component composable reward rubric: `format_valid`, `not_spam`,
  `decision_correct`, `asi_class_correct`, `task_correct`, `investigation_bonus`,
  `optimal_tools_match`.
- Three sequential training stages: V2 base SFT, V2-PLUS grounded discrimination
  refinement (deployed), GRPO post-training (research artifact, reward hacking
  observed and documented).
- TaskEnv abstraction with stub interfaces for medical_triage and code_review,
  showing the cross-domain extension pattern.
- 11 passing tests covering scenario schema validity and live env contract.
- Public Colab training notebook with markdown narrative for all three stages.
- Wandb experimental tracking for the SFT re-training trace.
- Reference inference script (`inference.py`) with heuristic + remote model paths.

## Near-term (post-hackathon, 2-4 weeks)

These are the most natural next steps. None require new research, only execution.

- **Multi-step RL training.** The env exposes `investigation_bonus` and
  `optimal_tools_match` reward components, but the deployed SFT model was
  trained on the single-turn dataset. Train an adapter on full multi-step
  trajectories so the model actually calls investigation tools before
  deciding. This was the original GRPO ambition before reward hacking forced
  the V2-PLUS retention decision.

- **Scenario expansion.** 15 adversarial scenarios across 8 classes is a
  starting point. Production benchmark needs 100+ per class with diverse
  phrasings, edge cases, and adversarial paraphrases. Scenario authoring is
  bounded work; aim for 80 scenarios across 8 classes in the next sprint.

- **Ship ASI08 and ASI10.** Cascading failures and rogue agents are the two
  remaining OWASP ASI 2026 classes. Architectural support exists in `models.py`;
  scenarios just need authoring.

- **Tighter reward function.** GRPO exposed reward hacking via
  `"asi_class": "ASI01-ASI07"` regex exploit. Replace substring match with
  strict ASIClass enum validation. Validate output schema against pydantic
  before scoring.

## Mid-term (1-3 months)

- **Cross-domain validation.** Medical_triage and code_review are 60-line and
  53-line stubs. Build them out to full domains with their own scenarios and
  policy rules. This converts the "domain-general" framing from architectural
  claim to empirical demonstration.

- **Frontier model leaderboard.** Run Claude, GPT-4o, Gemini 2.5, Llama 3.3,
  and your trained Qwen 1.7B against the same 15 scenarios. If a 1.7B model
  approaches frontier accuracy on adversarial robustness for OWASP ASI 2026,
  that is a publishable result on its own.

- **Reward hacking case study.** The `ASI01-ASI07` regex exploit is a clean,
  reproducible reward hacking example with a small enough loop to be
  pedagogically useful. Worth a short paper at NeurIPS Safety workshop or
  ICLR Safe AI workshop.

- **Multi-step env upgrade documentation.** Currently the multi-step support
  is implicit (hidden context surfaces through tool calls). Document the
  multi-turn protocol formally with a worked example.

## Long-term (3-12 months)

- **Strathos as a benchmark, not just an env.** Public train/test split with
  formal evaluation protocol. Leaderboard. Submission to NeurIPS Datasets &
  Benchmarks track 2026.

- **OWASP collaboration.** The Agentic Security Initiative is actively
  building reference materials. A public RL environment for their taxonomy
  is worth proposing to John Sotiropoulos (ASI co-lead, Agentic Top 10 Chair)
  as official ASI tooling.

- **Production deployment study.** Pair Strathos-trained agents with
  Microsoft Agent Governance Toolkit at runtime. Demonstrate "training-time
  hardening + runtime enforcement = defence in depth" with measurable
  attack-surface reduction. Microsoft AGT was released April 2026 and maps
  one-to-one against OWASP ASI 2026 — natural fit.

- **Adversarial co-evolution.** Currently the client and compliance oracle
  are rule-based to avoid meta-compromise. Open question: can we co-train
  a learned adversarial client with the focal agent without either side
  learning to game the other? Population-based training is the obvious
  starting point.

## Deliberate scoping decisions

These are not "missing features" — they are choices made under hackathon
time pressure, documented so reviewers know what was traded.

- **No formal verification.** Empirical robustness only. Formal refusal
  guarantees on input classes are out of scope.

- **Synthetic brokerage backend.** No real market data, no real order
  routing. Realistic enough for tool chain coherence, not a trading system.

- **Reg BI encoding is heuristic.** Sharp-edged tests for clean training
  signal. Production deployment needs compliance counsel involvement.

- **LoRA, not QLoRA.** Qwen 3 1.7B in bfloat16 fits in ~4 GB GPU memory,
  so QLoRA's quantization step adds nothing at this scale. The choice was
  to spend parameter budget on multiple training stages instead of pursuing
  a smaller memory footprint.

- **Single focal agent per rollout.** TRL's GRPOTrainer trains one policy
  at a time. Multi-agent co-training is a natural extension but not in v0.1.

## Integration opportunities

- **Microsoft Agent Governance Toolkit (AGT)** — released April 2026,
  maps to OWASP ASI 2026 one-to-one. Strathos hardens at training time;
  AGT enforces at runtime. Defence-in-depth pairing.

- **OWASP GenAI Security Project** — Strathos is the first OpenEnv-compliant
  RL benchmark for the ASI 2026 taxonomy. Worth proposing as reference
  tooling under the Agentic Security Initiative.

- **DeepTeam** (red-teaming framework) — already implements OWASP_ASI_2026
  attack categories. Strathos provides the trainable environment side;
  DeepTeam provides the attack catalog side. Natural complement.

## Contact

For collaboration on any of the above, reach out via the GitHub repo issues
or `work.kavyanshshakya@gmail.com`.

---

*This roadmap is honest about what is shipped, what is next, and what is
out of scope. The submission's value is in the env design and training
methodology, not in promising more than was built.*
