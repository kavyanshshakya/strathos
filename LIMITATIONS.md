# Limitations and Known Scope

Strathos v0.1 is a hackathon-scale demonstration. This document lists honest limitations, scoping choices, and future work.

## Scoping Choices (By Design)

- **Single focal agent per rollout.** TRL's GRPOTrainer trains one policy at a time via `rollout_func`. The client and compliance oracle are frozen/rule-based. Multi-agent co-training (e.g., population-based training, self-play) is a natural extension but not in v0.1.

- **Rule-based compliance oracle.** The escalation target uses deterministic rules, not a learned model. This is deliberate — a learned monitor would inherit the same attack surface as the trainee (meta-compromise risk). Research users who want adversarial co-evolution between worker and monitor should extend this.

- **Eight of ten OWASP ASI classes shipped.** v0.1 includes scenarios for ASI01, 02, 03, 04, 05, 06, 07, 09 (15 hand-authored scenarios). Architectural support exists for ASI08 (cascading failures) and ASI10 (rogue agents); they need scenario authoring.

- **Training-time defence only.** Strathos hardens agents during training. Production deployment should pair Strathos-trained agents with a runtime policy layer like Microsoft Agent Governance Toolkit (AGT). These are complementary: deterministic input-side policies + RL-trained internalised behaviour + runtime observability.

## Considered Threats (Reward Hacking)

Reward hacking — the tendency of RL agents to exploit unintended loopholes in the reward function — is a known failure mode (Amodei et al. 2016, *Concrete Problems in AI Safety*). During Strathos design we considered three specific attack surfaces on our own reward function:

- **Always-refuse policy.** An agent could refuse every request and still earn partial credit via ASI class labels. Mitigated by: (1) 70/30 legitimate/adversarial sampling during training — refusing on a legitimate request earns 0.0 reward, not positive partial credit; (2) per-class accuracy reporting exposes always-refuse policies via flat per-class performance.

- **Escalation spam.** An agent could escalate every uncertain case to bypass the refuse/comply decision. Mitigated by the confidence threshold (confidence < 0.35 on escalate → `not_spam` component = 0 → overall scalar = 0) and the `compliance_bandwidth_remaining` counter (limited oracle bandwidth per episode).

- **ASI class guessing.** An agent could refuse correctly but output arbitrary ASI class labels to collect the 0.5 partial credit. Mitigated by reporting per-ASI-class precision/recall in the leaderboard — random-class guessing produces uniform low accuracy, which is immediately visible.

These are standard alignment engineering defences, not novel contributions. We document them here because explicit consideration of reward hacking should be a baseline expectation for any RL-based safety system.

## Known Limitations

- **Domain coverage.** v0.1 ships one fully-implemented domain (`RoboAdvisorTaskEnv`) plus stub interfaces for medical triage and code review. The "domain-general" framing is architectural (TaskEnv interface separates domain from adversarial layer) rather than empirical. Empirical cross-domain validation is future work.

- **Reg BI encoding is heuristic.** Policy rules use sharp-edged tests (blacklists, keyword checks in reasoning) to generate clean training signal. Real Reg BI compliance has more nuance; users building production systems should work with compliance counsel on rule specification.

- **Synthetic brokerage backend.** In-memory state, no real market data, no real order routing. Realistic enough for tool chain coherence; not a trading system.

- **Scenario library size.** 15 hand-authored scenarios across 8 classes is small (production benchmark would need 100+ per class). Trained agent may overfit to specific phrasings. Continual scenario authoring is an ongoing effort.

- **No formal verification.** We provide empirical robustness via training and held-out evaluation. Formal refusal guarantees (e.g., SMT-based proofs that the agent refuses on specific input classes) are not provided.

## Future Work

- Multi-agent co-training via self-play or population-based methods
- Continual learning from production adversarial patterns (online SFT)
- Empirical cross-domain transfer experiments (full medical triage + code review implementations)
- Adversarial robustness certification via perturbation analysis
- Integration with Microsoft Agent Governance Toolkit as a combined training + runtime stack
- arXiv preprint submission and NeurIPS 2026 Datasets & Benchmarks track submission
