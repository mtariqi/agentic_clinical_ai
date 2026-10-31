"""Command-line entry points. Each sub-command is one Snakemake rule.

    python -m agentic_clinical.cli <command> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml


def _cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _write_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def _read_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["flags"] = df["flags"].apply(lambda s: json.loads(s) if isinstance(s, str) else [])
    return df


# ---------------------------------------------------------------------------
def cmd_generate(a) -> None:
    from agentic_clinical.data.synthetic import StudyConfig, generate_study, write_study

    cfg = _cfg(a.config)
    tables, truth = generate_study(StudyConfig.from_dict(cfg["study"]), seed=cfg["seed"])
    write_study(tables, truth, a.outdir)
    print(f"generated {sum(len(t) for t in tables.values())} rows across {len(tables)} tables")


def cmd_validate(a) -> None:
    from agentic_clinical.data.quality import validate_tables

    rep = validate_tables(a.datadir)
    _write_json(rep.to_dict(), a.out)
    for w in rep.warnings:
        print(f"WARNING: {w}")
    if not rep.passed:
        for e in rep.errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)  # stop the pipeline: agents never run on unvalidated data
    print("validation passed")


def cmd_features(a) -> None:
    from agentic_clinical.data.quality import load_tables
    from agentic_clinical.features import build_site_kris

    kri = build_site_kris(load_tables(a.datadir))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kri.to_csv(a.out, index=False)


def cmd_surveillance(a) -> None:
    from agentic_clinical.models.surveillance import score_sites

    cfg = _cfg(a.config)
    s = cfg["surveillance"]
    scores = score_sites(pd.read_csv(a.kri), z_threshold=s["robust_z_threshold"],
                         n_estimators=s["isolation_forest"]["n_estimators"],
                         contamination=s["isolation_forest"]["contamination"], seed=cfg["seed"],
                         shrinkage_k=s.get("shrinkage_k", 10.0))
    scores["flags"] = scores["flags"].apply(json.dumps)
    scores.to_csv(a.out, index=False)


def cmd_forecast(a) -> None:
    from agentic_clinical.models.enrollment import forecast_enrollment

    cfg = _cfg(a.config)
    st, e = cfg["study"], cfg["enrollment"]
    d = Path(a.datadir)
    fc = forecast_enrollment(pd.read_csv(d / "enrollment_weekly.csv"), pd.read_csv(d / "sites.csv"),
                             current_week=st["current_week"], study_weeks=st["study_weeks"],
                             recent_window=e["recent_window_weeks"], n_sims=e["n_simulations"],
                             shortfall_threshold=e["shortfall_risk_threshold"], seed=cfg["seed"])
    fc.to_csv(a.out, index=False)


def cmd_agents(a) -> None:
    from agentic_clinical.agents.audit import AuditLog
    from agentic_clinical.agents.core import RunContext
    from agentic_clinical.agents.enrollment_agent import EnrollmentAgent
    from agentic_clinical.agents.guardrails import Policy, PolicyEngine
    from agentic_clinical.agents.llm import build_planner
    from agentic_clinical.agents.memory import InstitutionalMemory
    from agentic_clinical.agents.surveillance_agent import SurveillanceAgent
    from agentic_clinical.agents.tools import Outbox, ToolRegistry

    cfg = _cfg(a.config)
    policy = Policy.load(a.policy)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("outbox.jsonl", "approvals.jsonl", "audit_log.jsonl"):
        (out / f).unlink(missing_ok=True)  # each run starts a fresh chain and outbox
        (out / f).touch()

    sites = pd.read_csv(Path(a.datadir) / "sites.csv")
    registry = ToolRegistry(Outbox(out / "outbox.jsonl"), set(sites["site_id"]), set(sites["country"]))
    audit = AuditLog(out / "audit_log.jsonl", run_id=cfg["run_id"], policy_version=policy.version)
    audit.record("system", "pipeline_inputs", {"scores": a.scores, "forecast": a.forecast,
                                                "policy": a.policy, "planner": cfg["agents"]["planner"]})
    ctx = RunContext(
        run_id=cfg["run_id"], registry=registry, policy_engine=PolicyEngine(policy), audit=audit,
        memory=InstitutionalMemory(a.memory), approvals_path=out / "approvals.jsonl",
        planner=build_planner(cfg["agents"], policy.redact_pii_before_llm),
        data={"site_scores": _read_scores(a.scores),
              "enrollment_forecast": pd.read_csv(a.forecast),
              "site_comments": pd.read_csv(Path(a.datadir) / "site_comments.csv")},
    )
    agents = [SurveillanceAgent(z_threshold=cfg["surveillance"]["robust_z_threshold"]), EnrollmentAgent()]
    results = []
    for agent in agents:
        try:
            results.append(asdict(agent.run(ctx)))
        except Exception as exc:  # one agent failing must not take down the others
            audit.record(agent.agent_id, "agent_crashed", {"error": f"{type(exc).__name__}: {exc}"})
            raise
    _write_json(results, out / "agent_results.json")
    for r in results:
        print(f"{r['agent_id']}: executed={len(r['executed'])} pending={len(r['pending_approval'])} "
              f"denied={len(r['denied'])} quarantined={len(r['quarantined'])}")


def cmd_oversight(a) -> None:
    from agentic_clinical.agents.core import AgentRunResult
    from agentic_clinical.agents.oversight import triage

    cfg = _cfg(a.config)["oversight"]
    results = [AgentRunResult(**r) for r in json.loads(Path(a.results).read_text())]
    t = triage(results, max_items=cfg["max_review_items"], min_priority=cfg["min_priority"])
    _write_json(t, a.out)
    print(f"oversight: {t['stats']['raw_signals']} signals -> {t['stats']['review_items']} review items")


def cmd_verify_audit(a) -> None:
    from agentic_clinical.agents.audit import verify_chain

    res = verify_chain(a.log)
    _write_json(res, a.out)
    if not res["valid"]:
        print(f"AUDIT CHAIN INVALID at seq {res['first_invalid_seq']}: {res['reason']}", file=sys.stderr)
        sys.exit(1)
    print(f"audit chain valid ({res['entries']} entries)")


def cmd_evaluate(a) -> None:
    from agentic_clinical.evaluation import evaluate

    m = evaluate(json.loads(Path(a.truth).read_text()), _read_scores(a.scores), pd.read_csv(a.forecast),
                 json.loads(Path(a.results).read_text()), json.loads(Path(a.triage).read_text()))
    _write_json(m, a.out)
    print(json.dumps({k: {n: v.get("recall") if isinstance(v, dict) else v for n, v in d.items()}
                      for k, d in m.items() if k != "oversight"}, indent=1))


def cmd_report(a) -> None:
    from agentic_clinical.reporting import build_report

    policy = yaml.safe_load(Path(a.policy).read_text())
    html = build_report(
        run_id=_cfg(a.config)["run_id"], truth=json.loads(Path(a.truth).read_text()),
        policy_version=str(policy.get("policy_version")), scores=_read_scores(a.scores),
        forecast=pd.read_csv(a.forecast), results=json.loads(Path(a.results).read_text()),
        triage=json.loads(Path(a.triage).read_text()), audit=json.loads(Path(a.audit).read_text()),
        metrics=json.loads(Path(a.metrics).read_text()),
        lineage=json.loads(Path(a.validation).read_text())["lineage"])
    Path(a.out).write_text(html)
    print(f"report written to {a.out}")


def cmd_decide(a) -> None:
    """Human-in-the-loop: approve or reject a queued action, signed into the audit chain."""
    from agentic_clinical.agents.audit import AuditLog, read_log, verify_chain
    from agentic_clinical.agents.tools import Outbox, ToolRegistry

    agdir = Path(a.agentsdir)
    pending = [json.loads(x) for x in (agdir / "approvals.jsonl").read_text().splitlines() if x.strip()]
    decisions_path = agdir / "decisions.jsonl"
    decided = set()
    if decisions_path.exists():
        decided = {json.loads(x)["index"] for x in decisions_path.read_text().splitlines() if x.strip()}
    if a.list or a.index is None:
        for i, p in enumerate(pending):
            state = "decided" if i in decided else "PENDING"
            print(f"[{i}] {state:8s} {p['tool']} {p['args']}  <- {p['requested_by']}: {p['rationale']}")
        return
    i = int(a.index)
    if not 0 <= i < len(pending):
        sys.exit(f"no queued action with index {i}")
    if i in decided:
        sys.exit(f"action {i} already decided")
    if not a.approver or not a.decision or not a.reason:
        sys.exit("--approver, --decision and --reason are required to record a decision")
    if not verify_chain(agdir / "audit_log.jsonl")["valid"]:
        sys.exit("audit chain invalid - refusing to append decisions")

    item = pending[i]
    first = read_log(agdir / "audit_log.jsonl")[0]
    audit = AuditLog(agdir / "audit_log.jsonl", run_id=first["run_id"],
                     policy_version=first["policy_version"])
    actor = f"human:{a.approver}"
    record = {"index": i, "tool": item["tool"], "args": item["args"], "decision": a.decision,
              "approver": a.approver, "reason": a.reason}
    if a.decision == "approve":
        sites = pd.read_csv(Path(a.datadir) / "sites.csv")
        reg = ToolRegistry(Outbox(agdir / "outbox.jsonl"), set(sites["site_id"]), set(sites["country"]))
        record["action_id"] = reg.execute(item["tool"], item["args"])["action_id"]
    audit.record(actor, f"human_{a.decision}", record)
    with open(decisions_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"{'approved' if a.decision == 'approve' else 'rejected'} action {i} ({item['tool']}) as {actor}")


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic-clinical", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, *args):
        sp = sub.add_parser(name)
        for arg in args:
            sp.add_argument(f"--{arg}", required=True)
        sp.set_defaults(func=fn)

    add("generate", cmd_generate, "config", "outdir")
    add("validate", cmd_validate, "datadir", "out")
    add("features", cmd_features, "datadir", "out")
    add("surveillance", cmd_surveillance, "config", "kri", "out")
    add("forecast", cmd_forecast, "config", "datadir", "out")
    add("agents", cmd_agents, "config", "policy", "datadir", "scores", "forecast", "memory", "outdir")
    add("oversight", cmd_oversight, "config", "results", "out")
    add("verify-audit", cmd_verify_audit, "log", "out")
    add("evaluate", cmd_evaluate, "truth", "scores", "forecast", "results", "triage", "out")
    dp = sub.add_parser("decide", help="list / approve / reject queued high-risk actions")
    dp.add_argument("--agentsdir", required=True)
    dp.add_argument("--datadir", required=True)
    dp.add_argument("--list", action="store_true")
    dp.add_argument("--index", type=int)
    dp.add_argument("--decision", choices=["approve", "reject"])
    dp.add_argument("--approver")
    dp.add_argument("--reason")
    dp.set_defaults(func=cmd_decide)
    add("report", cmd_report, "config", "policy", "truth", "scores", "forecast", "results", "triage",
        "audit", "metrics", "validation", "out")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
