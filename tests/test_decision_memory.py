"""Human decisions are remembered across runs and not re-requested."""

import json
from datetime import UTC, datetime, timedelta

from conftest import make_context

from agentic_clinical.agents.audit import AuditLog, verify_chain
from agentic_clinical.agents.enrollment_agent import EnrollmentAgent
from agentic_clinical.agents.memory import InstitutionalMemory
from agentic_clinical.agents.oversight import triage
from agentic_clinical.agents.surveillance_agent import SurveillanceAgent
from agentic_clinical.cli import main


def _decide(mem_path, action, decision, ts=None, args=None):
    return InstitutionalMemory(mem_path).record_decision(
        entity=action["entity"], tool=action["tool"], args=args or action["args"],
        decision=decision, approver="Tester", reason="unit test", run_id="wk30", ts=ts)


def test_rejected_request_is_not_requeued(tmp_path, study, model_outputs):
    mem = tmp_path / "mem.json"
    r1 = SurveillanceAgent().run(make_context(tmp_path / "r1", study, model_outputs, "wk30",
                                              memory_path=mem))
    audit_req = next(a for a in r1.pending_approval if a["tool"] == "trigger_for_cause_audit")
    _decide(mem, audit_req, "reject")

    r2 = SurveillanceAgent().run(make_context(tmp_path / "r2", study, model_outputs, "wk31",
                                              memory_path=mem))
    pending = {(a["tool"], a["entity"]) for a in r2.pending_approval}
    assert ("trigger_for_cause_audit", audit_req["entity"]) not in pending
    sup = [s for s in r2.suppressed if s["entity"] == audit_req["entity"]]
    assert sup and "already rejected by Tester" in sup[0]["reason"]
    # other, undecided audit requests are still queued
    others = {a["entity"] for a in r1.pending_approval
              if a["tool"] == "trigger_for_cause_audit"} - {audit_req["entity"]}
    assert others <= {a["entity"] for a in r2.pending_approval}
    assert "approval_suppressed_by_memory" in (tmp_path / "r2" / "audit.jsonl").read_text()


def test_approval_only_covers_identical_request(tmp_path, study, model_outputs):
    mem = tmp_path / "mem.json"
    r1 = EnrollmentAgent().run(make_context(tmp_path / "r1", study, model_outputs, "wk30",
                                            memory_path=mem))
    realloc = next(a for a in r1.pending_approval
                   if a["tool"] == "recommend_enrollment_reallocation")

    _decide(mem, realloc, "approve")
    r2 = EnrollmentAgent().run(make_context(tmp_path / "r2", study, model_outputs, "wk31",
                                            memory_path=mem))
    assert realloc["args"] not in [a["args"] for a in r2.pending_approval]
    assert realloc["args"] in [s["args"] for s in r2.suppressed]

    # a newer approval for *different* arguments no longer covers this request
    _decide(mem, realloc, "approve", args={**realloc["args"], "n_subjects": 1})
    r3 = EnrollmentAgent().run(make_context(tmp_path / "r3", study, model_outputs, "wk32",
                                            memory_path=mem))
    assert realloc["args"] in [a["args"] for a in r3.pending_approval]


def test_expired_decision_is_asked_again(tmp_path, study, model_outputs):
    mem = tmp_path / "mem.json"
    r1 = SurveillanceAgent().run(make_context(tmp_path / "r1", study, model_outputs, "wk30",
                                              memory_path=mem))
    audit_req = next(a for a in r1.pending_approval if a["tool"] == "trigger_for_cause_audit")
    _decide(mem, audit_req, "reject", ts=datetime.now(UTC) - timedelta(days=30))

    r2 = SurveillanceAgent().run(make_context(tmp_path / "r2", study, model_outputs, "wk31",
                                              memory_path=mem))  # default TTL: 14 days
    assert ("trigger_for_cause_audit", audit_req["entity"]) in {
        (a["tool"], a["entity"]) for a in r2.pending_approval}


def test_persistent_issues_rank_higher(tmp_path, study, model_outputs):
    mem = tmp_path / "mem.json"
    r1 = SurveillanceAgent().run(make_context(tmp_path / "r1", study, model_outputs, "wk30",
                                              memory_path=mem))
    r2 = SurveillanceAgent().run(make_context(tmp_path / "r2", study, model_outputs, "wk31",
                                              memory_path=mem))
    p1 = {e["entity"]: e["priority"] for e in triage([r1], max_items=100, min_priority=0)["review_queue"]}
    p2 = {e["entity"]: e["priority"] for e in triage([r2], max_items=100, min_priority=0)["review_queue"]}
    common = set(p1) & set(p2)
    assert common
    assert all(p2[e] >= p1[e] for e in common)
    assert any(p2[e] > p1[e] for e in common)


def test_cli_decide_writes_memory_and_keeps_chain_valid(tmp_path, study, model_outputs):
    tables, _ = study
    agdir, datadir, mem = tmp_path / "agents", tmp_path / "data", tmp_path / "mem" / "m.json"
    datadir.mkdir()
    tables["sites"].to_csv(datadir / "sites.csv", index=False)
    ctx = make_context(agdir, study, model_outputs, "wk30", memory_path=mem)
    ctx.audit = AuditLog(agdir / "audit_log.jsonl", run_id="wk30", policy_version="1.0.0")
    for f in ("outbox.jsonl",):
        (agdir / f).touch()
    EnrollmentAgent().run(ctx)

    main(["decide", "--agentsdir", str(agdir), "--datadir", str(datadir), "--memory", str(mem),
          "--index", "0", "--decision", "reject", "--approver", "QA", "--reason", "not now"])

    decisions = json.loads((mem.parent / "human_decisions.json").read_text())
    assert decisions[-1]["decision"] == "reject" and decisions[-1]["approver"] == "QA"
    assert verify_chain(agdir / "audit_log.jsonl")["valid"]
