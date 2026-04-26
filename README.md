---
title: Strathos
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: OpenEnv RL env for OWASP ASI 2026 robustness
tags:
  - reinforcement-learning
  - rl-environment
  - openenv
  - llm-safety
  - adversarial-robustness
  - owasp-asi-2026
  - fiduciary-ai
  - sec-reg-bi
---

# Strathos

OpenEnv RL environment for OWASP ASI 2026 adversarial robustness in regulated robo-advisors. Multi-step adversarial investigation with hidden context, 7-component composable reward rubric, and a Qwen 3 1.7B agent post-trained with SFT and GRPO.

Solo entry, Meta PyTorch OpenEnv Hackathon Grand Finale, Bangalore, April 25-26, 2026.

## Artifacts

| | |
|---|---|
| Live environment | https://huggingface.co/spaces/kavyanshshakya/strathos |
| Dataset (30 OWASP ASI scenarios, 8 classes) | https://huggingface.co/datasets/kavyanshshakya/strathos-asi-scenarios |
| SFT model (deployed, V2-PLUS) | https://huggingface.co/kavyanshshakya/strathos-qwen17b-sft |
| GRPO model (research artifact) | https://huggingface.co/kavyanshshakya/strathos-qwen17b-grpo |
| SFT re-training trace (with wandb) | https://huggingface.co/kavyanshshakya/strathos-qwen17b-sft-traced |
| Wandb experimental tracking | https://wandb.ai/kavyanshshakya-indian-institute-of-science-education-and/strathos-tracking |
| Colab training notebook (public) | https://colab.research.google.com/drive/1ggnRHlcE9Tk3RWxUpYSf3w-CaD3QNstV?usp=sharing |
| Roadmap (post-hackathon plans) | https://github.com/kavyanshshakya/strathos/blob/main/ROADMAP.md |
| Visual showcase (Space root) | https://kavyanshshakya-strathos.hf.space/ |

For the methodology and engineering story, see [BLOG.md](BLOG.md). For forward-looking work, see [ROADMAP.md](ROADMAP.md). For a visual showcase, visit the [HF Space root](https://kavyanshshakya-strathos.hf.space/).

## Quick start

**Hit the live env (no install required):**

```bash
curl -X POST https://kavyanshshakya-strathos.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'
```

This returns a client message + episode metadata. Adversarial scenarios include hidden context that investigation tools reveal.

**Run locally:**

```bash
git clone https://github.com/kavyanshshakya/strathos.git
cd strathos
pip install -e ".[dev]"
docker compose up -d   # env on http://localhost:7860
```

**Reproduce SFT + GRPO training (Colab Pro A100 recommended):**

Open [`notebooks/strathos_grpo_colab.ipynb`](notebooks/strathos_grpo_colab.ipynb) locally, or run directly via [public Colab link](https://colab.research.google.com/drive/1ggnRHlcE9Tk3RWxUpYSf3w-CaD3QNstV?usp=sharing). Three stages, each with markdown narrative:
1. Stage 1 SFT (V2) on 1300 templated examples
2. Stage 2 grounded discrimination refinement (V2-PLUS) — fixes over-refusal
3. Stage 3 standalone GRPO — research artifact, observed reward hacking

**Use the deployed model:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "kavyanshshakya/strathos-qwen17b-sft")
```

**Hit the env from Python:**

```python
import requests
ENV_URL = "https://kavyanshshakya-strathos.hf.space"

# Reset returns a client message + episode metadata (round-tripped via stateless HTTP)
reset = requests.post(f"{ENV_URL}/reset", json={}).json()
obs = reset["observation"]
print(obs["client_message"]["content"])
print("scenario:", obs["episode_meta"]["scenario_id"])
```


## API surface

The env exposes the standard OpenEnv 0.2.3 contract:

- `POST /reset` — start episode, returns `StrathosObservation`
- `POST /step` — submit `StrathosAction`, returns next observation + reward + done
- `GET /health`, `GET /state` — diagnostics

`StrathosAction` schema:

```python
{
  "decision": "execute" | "refuse" | "escalate" | "verify_first",
  "tool_call": {"name": str, "arguments": dict} | None,
  "confidence": float,                # 0.0 - 1.0
  "asi_class": "ASI01" | ... | "ASI09" | None,  # 8 classes shipped: ASI01-07, ASI09
  "reasoning": str,                   # max 500 chars
  "compliance_flag": bool,
  "metadata": dict                    # round-trip scenario_id + is_adversarial
}
```

Termination: `refuse` / `escalate` / `verify_first` / executed trade / `max_turns=6`.

## Repo layout

```
server/
  app.py                 # FastAPI server, port 7860
  task_env.py            # Abstract TaskEnv interface
  models.py              # Pydantic types (Action, Observation, State, ASIClass)
  robo_advisor/          # V1 concrete TaskEnv (16 tools, 5 categories)
  medical_triage/        # Cross-domain stub
  code_review/           # Cross-domain stub
  wrapper/               # Adversarial wrapper, composable reward
  scenarios/             # 15 OWASP ASI YAML scenarios across 8 attack classes
notebooks/               # SFT + GRPO training notebooks
tests/                   # Smoke tests
```

## Run an SFT baseline

See `notebooks/strathos_grpo_colab.ipynb`. Or load the pre-trained adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "kavyanshshakya/strathos-qwen17b-sft")
```

## OWASP ASI 2026 attack classes shipped

Strathos operationalises 8 of the 10 risks from the [**OWASP Top 10 for Agentic Applications 2026**](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/), the first formal taxonomy for autonomous AI agent security, published December 2025 by the OWASP Agentic Security Initiative.

| Class | Strathos label | Description (per OWASP ASI 2026) |
|---|---|---|
| ASI01 | Goal Hijacking | Manipulation of agent goals via prompt injection, emotional pressure, or false authority |
| ASI02 | Tool Misuse | Use of legitimate tools in unsafe ways (audit tampering, profile flipping, front-running) |
| ASI03 | Identity / Privilege Abuse | Credential bypass, weak auth, unverified third-party authority claims |
| ASI04 | Supply Chain | Tainted upstream data feeds, compromised research docs, unsigned third-party content |
| ASI05 | Code Execution | Untrusted user-supplied code or scripts in privileged execution contexts |
| ASI06 | Memory Poisoning | Prompt injection, fake admin tokens, structured override of system instructions |
| ASI07 | Inter-Agent Communication | Spoofed peer-agent, fabricated handoff, unauthenticated inter-agent claims |
| ASI09 | Human-Agent Trust | Exploitation of relationship history, emotional appeals, vulnerability-based bypass attempts |

ASI08 (cascading failures) and ASI10 (rogue agents) are scaffolded in `models.py` but not yet shipped as scenarios — future work.

**Reference**: OWASP GenAI Security Project. *OWASP Top 10 for Agentic Applications 2026*. December 2025. https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/

## Why this submission

Most OpenEnv submissions ship a single domain, a single training pass, and a single opaque reward scalar. Strathos chose depth over breadth — one domain (regulated finance under SEC Reg BI) done end-to-end with documented honest findings, rather than a shallow tour across many domains.

What's in here that isn't in most submissions:

- **TaskEnv abstraction with cross-domain stubs** for medical triage and code review. Demonstrates the extension pattern; finance is V1.
- **7-component composable reward rubric** — `format_valid`, `not_spam`, `decision_correct`, `asi_class_correct`, `task_correct`, `investigation_bonus`, `optimal_tools_match`. Debug per-skill, not just total score.
- **Multi-step investigation upgrade**. Hidden context per scenario that requires tool calls (audit_trail_query, check_risk_tolerance, etc.) to surface before deciding. Backward-compatible with single-turn API.
- **Three sequential training stages with documented failure modes**: V2 over-refusal, V2-PLUS grounded discrimination fix, GRPO reward hacking + format collapse. Failure modes documented alongside the successes.
- **Operationalises a published industry taxonomy** ([OWASP ASI 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)) — not a custom benchmark made up for the hackathon.

Trade-offs we made and own:
- 8 of 10 OWASP ASI classes shipped (15 hand-authored scenarios). ASI08 and ASI10 scaffolded for future work.
- 15 hand-authored adversarial scenarios across 8 OWASP ASI 2026 classes. Production benchmark would need 100+ per class.
- Multi-step reward components (`investigation_bonus`, `optimal_tools_match`) are exposed but the deployed SFT model was trained on the single-turn dataset. RL training against multi-step rewards is the natural next step.
- GRPO failed gracefully (reward hacking on small data) — V2-PLUS SFT retained as deployed.

For the full engineering story including the parts that didn't work, see [BLOG.md](BLOG.md).

## Roadmap

Strathos v0.1 is the hackathon submission. Near-term, mid-term, and long-term direction is documented in [ROADMAP.md](ROADMAP.md):

- **Near-term (2-4 weeks)**: multi-step RL training, scenario expansion to 80+, ship ASI08/ASI10, tighten reward function.
- **Mid-term (1-3 months)**: cross-domain validation, frontier model leaderboard, reward hacking case study.
- **Long-term (3-12 months)**: NeurIPS Datasets & Benchmarks 2026 submission, OWASP collaboration, Microsoft AGT pairing for defence-in-depth.

See [ROADMAP.md](ROADMAP.md) for full detail including deliberate scoping decisions and integration opportunities.

## Figures

Three figures from the V2-PLUS evaluation and GRPO Stage 3 runs. PNG and source data live in [`figures/`](figures/).

![Operating points: precision-recall trade-off across model conditions](figures/strathos-pr-operating-points.png)

*Operating points across model conditions on the V2-PLUS evaluation set (15 scenarios: 10 adversarial across 5 OWASP ASI classes + 5 legitimate). Base Qwen 1.7B sits at the origin (0% recall, 0% precision). V2-PLUS basic prompt achieves perfect precision but only 40% recall. The format-hint prompt trades precision (0.83) for recall (1.00), shifting along the F1 frontier. The 15-scenario evaluation predates the expansion to 8 ASI classes and 15 hand-authored scenarios in the shipped env.*

![Per-condition metrics and ASI-class confusion matrix](figures/strathos-eval-comparison.png)

*(a) Precision, recall, F1 across the three model conditions on adversarial detection. Base Qwen scores 0 across the board; V2-PLUS basic prompt achieves F1 0.57; format-hint prompt achieves F1 0.91. (b) ASI-class confusion for V2-PLUS basic prompt — most errors collapse onto ASI01 (over-classification) or fall into "none/wrong-format" rather than misclassifying across attack classes.*

![GRPO Stage-3 reward trajectory and loss/KL curves](figures/strathos-grpo-trajectory.png)

*(a) GRPO reward trajectory across 14 training steps. Composable rubric mean reward grows from 0.31 to 0.66 (+110%). (b) GRPO advantage-based loss and KL divergence from the V2-PLUS reference policy. Note: the GRPO model exhibits documented reward hacking (substring exploit on `asi_class`); V2-PLUS SFT is retained as the deployed model. See the [GRPO model card](https://huggingface.co/kavyanshshakya/strathos-qwen17b-grpo) for failure mode discussion.*

## Limitations

- 15 hand-authored adversarial scenarios across 8 OWASP ASI 2026 classes. Production benchmark would need 100+ per class.
- ASI08 (cascading failures) and ASI10 (rogue agents) are scaffolded in `models.py` but not yet shipped as scenarios.
- GRPO model exhibits reward hacking and format collapse (documented in [BLOG.md](BLOG.md) and the GRPO model card). V2-PLUS SFT is the deployed model.
- Multi-step investigation reward (`investigation_bonus`, `optimal_tools_match`) is exposed but the deployed SFT model was trained on the single-turn dataset. RL training against the multi-step rewards is future work.

## Tests

11 tests covering scenario schema validity and live env contract:

```bash
# Schema tests only (no env needed)
pytest tests/

# Full suite against live HF Space
STRATHOS_ENV_URL=https://kavyanshshakya-strathos.hf.space pytest tests/ -v
```

Last run: 11/11 passed in 8.81s. See [`tests/last_run.txt`](tests/last_run.txt).

## Experimental tracking

Stage 2 SFT re-training (continued from deployed V2-PLUS) was traced with wandb:

- **Wandb run** (real loss curve, 3 epochs, 75 steps): [v2plus-traced-rerun](https://wandb.ai/kavyanshshakya-indian-institute-of-science-education-and/strathos-tracking/runs/g3cxonxb)
- **Wandb project** (overview + comparisons): [strathos-tracking](https://wandb.ai/kavyanshshakya-indian-institute-of-science-education-and/strathos-tracking)
- **Trained adapter**: [`strathos-qwen17b-sft-traced`](https://huggingface.co/kavyanshshakya/strathos-qwen17b-sft-traced)

The deployed model is V2-PLUS (`strathos-qwen17b-sft`); the traced adapter is published separately as a verifiable re-training reference.

## License

MIT
