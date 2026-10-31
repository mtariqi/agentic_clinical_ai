import json

import pytest

from agentic_clinical.agents.audit import AuditLog, read_log, verify_chain
from agentic_clinical.agents.tools import Outbox, ToolError, ToolRegistry


def _log(tmp_path, n=5):
    log = AuditLog(tmp_path / "a.jsonl", run_id="r", policy_version="1")
    for i in range(n):
        log.record("agent", "evt", {"i": i})
    return tmp_path / "a.jsonl"


def test_chain_valid_and_resumable(tmp_path):
    path = _log(tmp_path)
    AuditLog(path, run_id="r", policy_version="1").record("agent", "evt", {"i": 99})  # reopen
    res = verify_chain(path)
    assert res["valid"] and res["entries"] == 6


def test_tampered_payload_detected(tmp_path):
    path = _log(tmp_path)
    lines = path.read_text().splitlines()
    e = json.loads(lines[2])
    e["payload"]["i"] = 1000
    lines[2] = json.dumps(e)
    path.write_text("\n".join(lines) + "\n")
    res = verify_chain(path)
    assert not res["valid"] and res["first_invalid_seq"] == 3


def test_deleted_entry_detected(tmp_path):
    path = _log(tmp_path)
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")
    assert not verify_chain(path)["valid"]


def test_entries_are_attributable(tmp_path):
    for e in read_log(_log(tmp_path)):
        assert e["actor"] and e["ts"] and e["policy_version"] == "1"


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(Outbox(tmp_path / "o.jsonl"), {"US-001"}, {"US", "DE"})


def test_tool_arg_validation(registry):
    with pytest.raises(ToolError, match="missing"):
        registry.execute("raise_data_query", {"site_id": "US-001"})
    with pytest.raises(ToolError, match="unexpected"):
        registry.execute("notify_study_team", {"topic": "t", "message": "m", "severity": "low", "x": 1})
    with pytest.raises(ToolError, match="must be"):
        registry.execute("schedule_targeted_sdv", {"site_id": "US-001", "scope": "s", "percent": True})
    with pytest.raises(ToolError, match="unknown site"):
        registry.execute("raise_data_query", {"site_id": "ZZ-1", "kri": "k", "detail": "d"})
    with pytest.raises(ToolError):
        registry.execute("recommend_enrollment_reallocation",
                         {"from_country": "US", "to_country": "US", "n_subjects": 3})


def test_tool_effects_are_staged(registry, tmp_path):
    rec = registry.execute("raise_data_query", {"site_id": "US-001", "kri": "k", "detail": "d"})
    staged = [json.loads(x) for x in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert staged[0]["action_id"] == rec["action_id"] and staged[0]["status"] == "staged"
