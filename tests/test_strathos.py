"""
End-to-end tests for the Strathos OpenEnv environment + scenarios.

Run locally:    pytest tests/
Run against live env:  STRATHOS_ENV_URL=https://kavyanshshakya-strathos.hf.space pytest tests/
"""
import os
import json
import glob
import pytest
import requests
import yaml

ENV_URL = os.environ.get("STRATHOS_ENV_URL", "http://localhost:7860")
SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "..", "server", "scenarios")
SHIPPED_ASI_CLASSES = {"ASI01", "ASI02", "ASI03", "ASI04", "ASI05", "ASI06", "ASI07", "ASI09"}


# === Scenario schema tests (offline, no env required) ===

@pytest.fixture(scope="module")
def all_scenarios():
    """Load every scenario YAML in the repo."""
    scenarios = []
    for path in sorted(glob.glob(os.path.join(SCENARIO_DIR, "*.yaml"))):
        with open(path) as f:
            data = yaml.safe_load(f) or []
        if isinstance(data, list):
            scenarios.extend(data)
        elif isinstance(data, dict):
            scenarios.append(data)
    return scenarios


def test_scenarios_load(all_scenarios):
    """Repo ships at least 15 hand-authored scenarios."""
    assert len(all_scenarios) >= 15, f"expected >=15 scenarios, found {len(all_scenarios)}"


def test_scenarios_unique_ids(all_scenarios):
    """Every scenario has a unique id (no copy-paste duplicates)."""
    ids = [s["id"] for s in all_scenarios]
    duplicates = [x for x in ids if ids.count(x) > 1]
    assert not duplicates, f"duplicate ids: {set(duplicates)}"


def test_scenarios_required_fields(all_scenarios):
    """Each scenario has the fields the env wrapper depends on."""
    required = ["id", "asi_class", "domain", "client_opener",
                "expected_decision", "optimal_tools", "optimal_steps"]
    for s in all_scenarios:
        missing = [f for f in required if f not in s]
        assert not missing, f"scenario {s.get('id', '?')} missing fields: {missing}"


def test_scenarios_asi_class_valid(all_scenarios):
    """Every adversarial scenario uses a shipped ASI class."""
    for s in all_scenarios:
        assert s["asi_class"] in SHIPPED_ASI_CLASSES, \
            f"scenario {s['id']} uses unshipped class {s['asi_class']}"


def test_scenarios_decision_enum(all_scenarios):
    """expected_decision is one of the 4 allowed values."""
    allowed = {"refuse", "execute", "escalate", "verify_first"}
    for s in all_scenarios:
        assert s["expected_decision"] in allowed, \
            f"scenario {s['id']} has invalid decision {s['expected_decision']}"


def test_scenarios_optimal_tools_nonempty(all_scenarios):
    """Every scenario annotates at least one optimal investigation tool."""
    for s in all_scenarios:
        assert s["optimal_tools"], f"scenario {s['id']} has empty optimal_tools"
        assert len(s["optimal_tools"]) == s["optimal_steps"], \
            f"scenario {s['id']}: optimal_steps={s['optimal_steps']} but {len(s['optimal_tools'])} tools listed"


def test_eight_asi_classes_covered(all_scenarios):
    """Repo ships scenarios across at least 8 OWASP ASI 2026 classes."""
    classes_in_data = {s["asi_class"] for s in all_scenarios}
    assert len(classes_in_data) >= 8, f"only {len(classes_in_data)} ASI classes covered: {classes_in_data}"


# === Live env contract tests (require running env) ===

def _env_running():
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _env_running(), reason="env not reachable at STRATHOS_ENV_URL")
def test_env_health():
    r = requests.get(f"{ENV_URL}/health", timeout=10)
    assert r.status_code == 200


@pytest.mark.skipif(not _env_running(), reason="env not reachable at STRATHOS_ENV_URL")
def test_env_reset_returns_observation():
    """POST /reset returns a valid StrathosObservation."""
    r = requests.post(f"{ENV_URL}/reset", json={}, timeout=15)
    assert r.status_code == 200
    body = r.json()
    obs = body.get("observation") or body
    assert "client_message" in obs
    assert obs["client_message"]["content"]
    assert "episode_meta" in obs
    ep = obs["episode_meta"]
    assert "scenario_id" in ep
    assert "is_adversarial" in ep


@pytest.mark.skipif(not _env_running(), reason="env not reachable at STRATHOS_ENV_URL")
def test_env_step_with_refuse_on_adversarial():
    """POST /step with a refuse decision on adversarial scenario yields positive reward."""
    # Pull an adversarial scenario (retry until we get one)
    obs = None
    for _ in range(10):
        r = requests.post(f"{ENV_URL}/reset", json={}, timeout=15)
        body = r.json()
        candidate = body.get("observation") or body
        if candidate["episode_meta"].get("is_adversarial"):
            obs = candidate
            break
    assert obs is not None, "could not sample adversarial scenario in 10 tries"

    ep = obs["episode_meta"]
    action = {
        "action": {
            "decision": "refuse",
            "tool_call": None,
            "confidence": 0.95,
            "asi_class": ep["asi_class_hint"],
            "reasoning": "test refusal with correct class",
            "compliance_flag": True,
            "metadata": {"scenario_id": ep["scenario_id"], "is_adversarial": True},
        }
    }
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=15)
    assert r.status_code == 200
    body = r.json()
    reward = body.get("reward")
    assert reward is not None
    assert reward >= 0.5, f"refuse + correct class on adversarial should reward >=0.5, got {reward}"


@pytest.mark.skipif(not _env_running(), reason="env not reachable at STRATHOS_ENV_URL")
def test_env_diverse_scenarios_served():
    """Live env serves at least 6 distinct ASI classes across 50 resets."""
    classes = set()
    for _ in range(50):
        try:
            r = requests.post(f"{ENV_URL}/reset", json={}, timeout=10).json()
            ep = (r.get("observation") or r)["episode_meta"]
            cls = ep.get("asi_class_hint")
            if cls:
                classes.add(cls)
        except Exception:
            pass
    assert len(classes) >= 6, f"expected >=6 ASI classes in 50 resets, saw {classes}"
