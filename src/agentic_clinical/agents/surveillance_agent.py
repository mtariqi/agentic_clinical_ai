"""Clinical Data Surveillance agent.

Objective: keep site-level data-quality risk under control with the least
intrusive effective action. Codified decision logic (the rule planner) is kept
explicit on purpose - "processes live in specifications, not in people's heads".
"""

from __future__ import annotations

import pandas as pd

from agentic_clinical.agents.core import Agent, AgentRunResult, ProposedAction, RunContext


class SurveillanceAgent(Agent):
    agent_id = "surveillance_agent"

    def __init__(self, z_threshold: float = 3.0):
        self.z = z_threshold

    def observe(self, ctx: RunContext) -> dict:
        scores: pd.DataFrame = ctx.data["site_scores"]
        relevant = scores[(scores["risk_tier"].isin(["medium", "high"])) |
                          (scores["flags"].apply(len) > 0)]
        sites = [{
            "site_id": r.site_id, "country": r.country, "n_subjects": int(r.n_subjects),
            "risk_score": float(r.risk_score), "risk_tier": r.risk_tier,
            "if_outlier": bool(r.if_outlier), "flags": r.flags,
        } for r in relevant.itertuples()]
        all_sites = scores["site_id"].tolist()
        return {"objective": "Contain site-level data-quality risk",
                "z_threshold": self.z, "sites": sites, "entities": all_sites,
                "n_sites_monitored": len(scores),
                "n_insufficient_data": int((scores["risk_tier"] == "insufficient_data").sum())}

    def untrusted_texts(self, ctx: RunContext) -> list[tuple[str, str]]:
        comments: pd.DataFrame = ctx.data["site_comments"]
        return list(zip(comments["site_id"], comments["comment"].astype(str)))

    def rule_plan(self, obs: dict, ctx: RunContext) -> list[ProposedAction]:
        z2 = 2 * self.z
        acts: list[ProposedAction] = []
        for s in obs["sites"]:
            sid, persistent = s["site_id"], obs["memory"].get(s["site_id"], 0) > 0
            note = " Persistent: flagged in a previous cycle." if persistent else ""
            for f in s["flags"]:
                conf = min(1.0, f["z"] / z2)
                ev = {"kri": f["kri"], "value": f["value"], "z": f["z"], "persistent": persistent}
                if f["kri"] == "lab_variance_ratio":
                    acts.append(ProposedAction(
                        "flag_site_for_review",
                        {"site_id": sid, "reason": f"Implausibly low lab variability (z={f['z']})",
                         "risk_score": s["risk_score"]},
                        rationale=f"{f['description']} = {f['value']} is {f['z']} robust SD in the risky "
                                  f"direction; pattern consistent with data fabrication.{note}",
                        entity=sid, severity="high", confidence=conf, evidence=ev))
                    if f["z"] >= z2 or persistent:
                        acts.append(ProposedAction(
                            "trigger_for_cause_audit",
                            {"site_id": sid, "reason": "Suspected data integrity issue: lab variance"},
                            rationale=f"Extreme lab-variance anomaly (z={f['z']} >= {z2}){note}",
                            entity=sid, severity="high", confidence=conf, evidence=ev))
                else:
                    acts.append(ProposedAction(
                        "raise_data_query",
                        {"site_id": sid, "kri": f["kri"],
                         "detail": f"{f['description']}: {f['value']} (z={f['z']})"},
                        rationale=f"KRI {f['kri']} breached threshold z>={self.z}.{note}",
                        entity=sid, severity="high" if (f["z"] >= z2 or persistent) else "medium",
                        confidence=conf, evidence=ev))
            if s["risk_tier"] == "high":
                kris = ", ".join(f["kri"] for f in s["flags"]) or "multivariate pattern"
                acts.append(ProposedAction(
                    "schedule_targeted_sdv",
                    {"site_id": sid, "scope": kris, "percent": 100 if s["risk_score"] >= 0.8 else 50},
                    rationale=f"Composite risk {s['risk_score']:.2f} (high tier); verify source data "
                              f"for: {kris}.{note}",
                    entity=sid, severity="high", confidence=s["risk_score"],
                    evidence={"risk_score": s["risk_score"], "flags": [f["kri"] for f in s["flags"]]}))
            elif s["risk_tier"] == "medium" and not s["flags"]:
                acts.append(ProposedAction(
                    "notify_study_team",
                    {"topic": f"Site {sid}: multivariate anomaly",
                     "message": f"Isolation-forest pattern flagged site {sid} "
                                f"(risk {s['risk_score']:.2f}) with no single KRI breach.",
                     "severity": "low"},
                    rationale="Unusual KRI combination without a single-threshold breach; inform, "
                              "do not intervene.",
                    entity=sid, severity="low", confidence=s["risk_score"],
                    evidence={"risk_score": s["risk_score"], "if_outlier": s["if_outlier"]}))

        for sec in obs["security_flags"]:
            acts.append(ProposedAction(
                "notify_study_team",
                {"topic": f"Security: suspicious free text at {sec['entity']}",
                 "message": "Site comment matched prompt-injection patterns "
                            f"({', '.join(sec['reasons'])}) and was quarantined. Content withheld.",
                 "severity": "high"},
                rationale="Untrusted input attempted to steer agent behaviour; human review needed.",
                entity=sec["entity"], severity="high", confidence=0.9,
                evidence={"reasons": sec["reasons"]}))
        return acts

    def learned(self, obs: dict, result: AgentRunResult) -> list[str]:
        out = []
        for s in obs["sites"]:
            if obs["memory"].get(s["site_id"], 0) > 0:
                out.append(f"{s['site_id']} remains at risk across cycles; prior actions insufficient.")
        if obs["security_flags"]:
            out.append(f"{len(obs['security_flags'])} injection attempt(s) seen in site comments.")
        return out
