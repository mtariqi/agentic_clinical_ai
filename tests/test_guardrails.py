import pytest
from conftest import POLICY_PATH

from agentic_clinical.agents.guardrails import (
    CircuitBreaker,
    Policy,
    PolicyEngine,
    Verdict,
    redact_pii,
    screen_text,
)
from agentic_clinical.data.synthetic import BENIGN_COMMENTS, INJECTION_COMMENTS


@pytest.mark.parametrize("text", INJECTION_COMMENTS)
def test_injections_are_caught(text):
    assert not screen_text(text).safe


@pytest.mark.parametrize("text", BENIGN_COMMENTS)
def test_benign_comments_pass(text):
    assert screen_text(text).safe


def test_pii_redaction():
    out = redact_pii("Contact jane.doe@site.org or +1 (919) 555-0100, DOB 1980-04-12")
    assert "@" not in out and "555" not in out and "1980" not in out


def test_least_privilege_and_rationale():
    eng = PolicyEngine(Policy.load(POLICY_PATH))
    assert eng.authorize("enrollment_agent", "trigger_for_cause_audit", "x").verdict == Verdict.DENY
    assert eng.authorize("surveillance_agent", "raise_data_query", "").verdict == Verdict.DENY
    assert eng.authorize("surveillance_agent", "not_a_tool", "x").verdict == Verdict.DENY
    with pytest.raises(PermissionError):
        eng.authorize("rogue_agent", "raise_data_query", "x")


def test_autonomy_gating():
    eng = PolicyEngine(Policy.load(POLICY_PATH))
    # intermediate: low + medium autonomous, critical always gated
    assert eng.authorize("surveillance_agent", "raise_data_query", "r").verdict == Verdict.EXECUTE
    assert eng.authorize("surveillance_agent", "schedule_targeted_sdv", "r").verdict == Verdict.EXECUTE
    assert eng.authorize("surveillance_agent", "trigger_for_cause_audit", "r").verdict == Verdict.REQUIRE_APPROVAL
    # basic: high-risk reallocation is gated
    assert (eng.authorize("enrollment_agent", "recommend_enrollment_reallocation", "r").verdict
            == Verdict.REQUIRE_APPROVAL)


def test_action_budget():
    policy = Policy.load(POLICY_PATH)
    eng = PolicyEngine(policy)
    budget = policy.agents["enrollment_agent"].max_actions
    for _ in range(budget):
        assert eng.authorize("enrollment_agent", "notify_study_team", "r").verdict == Verdict.EXECUTE
    d = eng.authorize("enrollment_agent", "notify_study_team", "r")
    assert d.verdict == Verdict.DENY and "budget" in d.reason


def test_circuit_breaker():
    cb = CircuitBreaker(max_consecutive_errors=2)
    assert not cb.failure()
    cb.success()
    assert not cb.failure()
    assert cb.failure() and cb.tripped


def test_policy_rejects_unknown_tool_reference(tmp_path):
    bad = POLICY_PATH.read_text().replace("- notify_study_team\n      - recommend",
                                          "- notify_study_team\n      - delete_database\n      - recommend")
    p = tmp_path / "p.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="unknown tools"):
        Policy.load(p)
