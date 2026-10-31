"""Predictive Enrollment Intelligence agent.

Objective: protect the enrollment timeline. Detects countries whose recent rate
signals a shortfall *before* it shows in cumulative reports, alerts the team,
and proposes a target reallocation to countries with headroom. Reallocation is
HIGH risk, so under a BASIC autonomy policy it is always queued for a human.
"""

from __future__ import annotations

import math

import pandas as pd

from agentic_clinical.agents.core import Agent, AgentRunResult, ProposedAction, RunContext


class EnrollmentAgent(Agent):
    agent_id = "enrollment_agent"

    def observe(self, ctx: RunContext) -> dict:
        fc: pd.DataFrame = ctx.data["enrollment_forecast"]
        cols = ["country", "target", "enrolled_to_date", "rate_earlier", "rate_recent", "rate_ratio",
                "rate_drop_pvalue", "projected_median", "projected_p10", "projected_p90",
                "p_miss_target", "projected_shortfall", "headroom", "at_risk", "slowdown_detected"]
        countries = [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
                     for row in fc[cols].to_dict("records")]
        return {"objective": "Protect the enrollment timeline", "countries": countries,
                "entities": fc["country"].tolist(),
                "study_target": int(fc["target"].sum()),
                "study_projected_median": float(fc["projected_median"].sum())}

    def rule_plan(self, obs: dict, ctx: RunContext) -> list[ProposedAction]:
        acts: list[ProposedAction] = []
        donors = sorted([c for c in obs["countries"] if not c["at_risk"] and c["headroom"] > 0],
                        key=lambda c: -c["headroom"])
        capacity = {c["country"]: math.floor(c["headroom"]) for c in donors}

        for c in obs["countries"]:
            cc = c["country"]
            persistent = obs["memory"].get(cc, 0) > 0
            ev = {k: c[k] for k in ("rate_ratio", "rate_drop_pvalue", "p_miss_target",
                                    "projected_shortfall", "projected_median", "target")}
            if c["at_risk"]:
                acts.append(ProposedAction(
                    "notify_study_team",
                    {"topic": f"Enrollment risk: {cc}",
                     "message": (f"{cc} projected to reach {c['projected_median']:.0f}/{c['target']} "
                                 f"(80% PI {c['projected_p10']:.0f}-{c['projected_p90']:.0f}); "
                                 f"P(miss)={c['p_miss_target']:.0%}. Recent rate is "
                                 f"{c['rate_ratio']:.2f}x earlier rate."),
                     "severity": "high"},
                    rationale=f"Projected shortfall {c['projected_shortfall']:.0%} >= threshold with "
                              f"P(miss)={c['p_miss_target']:.0%}.",
                    entity=cc, severity="high", confidence=c["p_miss_target"], evidence=ev))
                gap = math.ceil(c["target"] - c["projected_median"])
                for d in donors:
                    if gap <= 0:
                        break
                    n = min(gap, capacity[d["country"]])
                    if n <= 0:
                        continue
                    capacity[d["country"]] -= n
                    gap -= n
                    acts.append(ProposedAction(
                        "recommend_enrollment_reallocation",
                        {"from_country": cc, "to_country": d["country"], "n_subjects": int(n)},
                        rationale=(f"{d['country']} projects {d['headroom']:.0f} subjects above target; "
                                   f"shift {n} to offset {cc} shortfall."
                                   + (" Persistent risk from previous cycle." if persistent else "")),
                        entity=cc, severity="high", confidence=c["p_miss_target"],
                        evidence={**ev, "donor": d["country"], "donor_headroom": d["headroom"]}))
            elif c["slowdown_detected"]:
                acts.append(ProposedAction(
                    "notify_study_team",
                    {"topic": f"Early warning: enrollment slowdown in {cc}",
                     "message": (f"Recent weekly rate {c['rate_recent']} vs {c['rate_earlier']} earlier "
                                 f"(p={c['rate_drop_pvalue']:.1e}); not yet projected to miss target."),
                     "severity": "medium"},
                    rationale="Statistically significant slowdown; early signal only.",
                    entity=cc, severity="medium", confidence=0.6, evidence=ev))
        return acts

    def learned(self, obs: dict, result: AgentRunResult) -> list[str]:
        return [f"{c['country']}: recent/earlier rate ratio {c['rate_ratio']}, "
                f"P(miss)={c['p_miss_target']}" for c in obs["countries"] if c["at_risk"]]
