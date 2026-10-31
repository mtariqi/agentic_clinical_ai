import json

from conftest import make_context

from agentic_clinical.agents.audit import verify_chain
from agentic_clinical.agents.core import Planner, ProposedAction
from agentic_clinical.agents.enrollment_agent import EnrollmentAgent
from agentic_clinical.agents.llm import parse_actions
from agentic_clinical.agents.oversight import triage
from agentic_clinical.agents.surveillance_agent import SurveillanceAgent
from agentic_clinical.agents.tools import ToolError, ToolSpec


def test_end_to_end_governed_run(tmp_path, study, model_outputs):
    _, truth = study
    ctx = make_context(tmp_path, study, model_outputs)
    s = SurveillanceAgent().run(ctx)
    e = EnrollmentAgent().run(ctx)

    # every injection was quarantined, and the offending text never reached the log
    assert {q["entity"] for q in s.quarantined} == set(truth["prompt_injection_sites"])
    log_text = (tmp_path / "audit.jsonl").read_text().lower()
    assert "ignore all previous instructions" not in log_text

    # critical / high-risk actions are never executed autonomously
    executed_tools = {a["tool"] for a in s.executed + e.executed}
    assert "trigger_for_cause_audit" not in executed_tools
    assert "recommend_enrollment_reallocation" not in executed_tools
    pending_tools = {a["tool"] for a in s.pending_approval + e.pending_approval}
    assert "recommend_enrollment_reallocation" in pending_tools

    # every executed action carries a rationale and the audit chain is intact
    assert all(a["rationale"] for a in s.executed + e.executed)
    assert verify_chain(tmp_path / "audit.jsonl")["valid"]


def test_memory_escalates_persistent_issues(tmp_path, study, model_outputs):
    mem = tmp_path / "mem.json"
    SurveillanceAgent().run(make_context(tmp_path / "r1", study, model_outputs, "wk30", memory_path=mem))
    r2 = SurveillanceAgent().run(make_context(tmp_path / "r2", study, model_outputs, "wk31", memory_path=mem))
    queries = [a for a in r2.executed if a["tool"] == "raise_data_query"]
    assert queries and all(a["severity"] == "high" for a in queries)
    assert all(a["evidence"]["persistent"] for a in queries)
    entries = json.loads(mem.read_text())
    assert {e["run_id"] for e in entries} == {"wk30", "wk31"}


def test_rerun_same_run_id_does_not_self_escalate(tmp_path, study, model_outputs):
    mem = tmp_path / "mem.json"
    SurveillanceAgent().run(make_context(tmp_path / "a", study, model_outputs, "same", memory_path=mem))
    r = SurveillanceAgent().run(make_context(tmp_path / "b", study, model_outputs, "same", memory_path=mem))
    assert not any(a["evidence"].get("persistent") for a in r.executed)


class ExplodingPlanner(Planner):
    name = "exploding"

    def plan(self, agent, observation, ctx):
        raise RuntimeError("LLM unavailable")


def test_planner_failure_falls_back_to_rules(tmp_path, study, model_outputs):
    ctx = make_context(tmp_path, study, model_outputs, planner=ExplodingPlanner())
    res = EnrollmentAgent().run(ctx)
    assert res.planner == "rules(fallback)" and res.executed
    assert "planner_fallback" in (tmp_path / "audit.jsonl").read_text()


def test_circuit_breaker_stops_cascading_failures(tmp_path, study, model_outputs):
    ctx = make_context(tmp_path, study, model_outputs)

    def broken(**_):
        raise ToolError("EDC API timeout")

    for name in ctx.registry.names():  # simulate a downstream system outage
        spec = ctx.registry.get(name)
        ctx.registry.register(ToolSpec(spec.name, spec.description, spec.args, broken))
    res = SurveillanceAgent().run(ctx)
    assert res.circuit_breaker_tripped
    assert len(res.failed) == ctx.policy_engine.policy.max_consecutive_tool_errors
    assert res.deferred  # remaining work handed back, not silently dropped


def test_llm_output_is_strictly_validated(tmp_path, study, model_outputs):
    ctx = make_context(tmp_path, study, model_outputs)
    site = model_outputs[0]["site_id"].iloc[0]
    text = json.dumps({"actions": [
        {"tool": "raise_data_query", "args": {"site_id": site, "kri": "query_rate", "detail": "d"},
         "rationale": "z=4", "entity": site, "severity": "medium", "confidence": 0.7},
        {"tool": "delete_all_data", "args": {}, "rationale": "x", "entity": site},
        {"tool": "raise_data_query", "args": {"site_id": site}, "rationale": "x", "entity": site},
    ]})
    allowed = ctx.policy_engine.identity("surveillance_agent").allowed_tools
    actions, rejected = parse_actions("```json\n" + text + "\n```", ctx, allowed)
    assert len(actions) == 1 and len(rejected) == 2


def test_oversight_never_hides_decisions_or_security(tmp_path, study, model_outputs):
    ctx = make_context(tmp_path, study, model_outputs)
    results = [SurveillanceAgent().run(ctx), EnrollmentAgent().run(ctx)]
    t = triage(results, max_items=1, min_priority=0.99)
    queued = {e["entity"] for e in t["review_queue"]}
    for r in results:
        assert {a["entity"] for a in r.pending_approval} <= queued
        assert {q["entity"] for q in r.quarantined} <= queued
    assert t["stats"]["review_items"] < t["stats"]["raw_signals"]


def test_proposed_action_validates_severity():
    import pytest

    with pytest.raises(ValueError):
        ProposedAction("notify_study_team", {}, "r", "US", severity="urgent")
