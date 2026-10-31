"""Self-contained HTML run report (figures embedded as base64 PNG)."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def enrollment_figure(fc: pd.DataFrame) -> str:
    fc = fc.sort_values("country")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = range(len(fc))
    colors = ["#d1495b" if r else "#2e86ab" for r in fc["at_risk"]]
    ax.bar(x, fc["projected_median"], color=colors, alpha=0.85)
    ax.errorbar(x, fc["projected_median"],
                yerr=[fc["projected_median"] - fc["projected_p10"], fc["projected_p90"] - fc["projected_median"]],
                fmt="none", ecolor="#333", capsize=4)
    ax.scatter(x, fc["target"], marker="_", s=900, color="#111", zorder=3)
    ax.set_xticks(list(x), fc["country"])
    ax.set_ylabel("Subjects at study end")
    ax.set_title("Projected final enrollment by country (red = at risk)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(color="#2e86ab", alpha=0.85, label="Projected median (on track)"),
        Patch(color="#d1495b", alpha=0.85, label="Projected median (at risk)"),
        Line2D([], [], color="#333", marker="|", markersize=10, linestyle="None",
               label="80% interval"),
        Line2D([], [], color="#111", linewidth=2, label="Target"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_b64(fig)


def risk_figure(scores: pd.DataFrame, top: int = 20) -> str:
    s = scores.dropna(subset=["risk_score"]).head(top).iloc[::-1]
    cmap = {"high": "#d1495b", "medium": "#edae49", "low": "#66a182"}
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(s) + 1))
    ax.barh(s["site_id"], s["risk_score"], color=[cmap.get(t, "#999") for t in s["risk_tier"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Composite risk score")
    ax.set_title(f"Top {len(s)} sites by data-quality risk")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_b64(fig)


TEMPLATE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic Clinical AI run report - {{ run_id }}</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--muted:#666;--card:#f6f7f9;--line:#e3e5e8;--bad:#c0392b;--ok:#1e8449;--warn:#b9770e}
@media (prefers-color-scheme: dark){:root{--bg:#141517;--fg:#ececec;--muted:#a0a0a0;--card:#1e2023;--line:#2e3135;--bad:#ff7b6b;--ok:#6fd08c;--warn:#f5c46b}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:0 auto;padding:24px}
h1{margin-bottom:4px}h2{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:36px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:var(--card);border-radius:10px;padding:14px}.big{font-size:26px;font-weight:600}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.bad{color:var(--bad);font-weight:600}.ok{color:var(--ok);font-weight:600}.warn{color:var(--warn);font-weight:600}
img{max-width:100%;background:#fff;border-radius:8px}code{font-size:12px}
</style></head><body>
<h1>Agentic Clinical AI - run report</h1>
<p class="muted">Run <code>{{ run_id }}</code> · study {{ truth.study_id }} · week {{ truth.current_week }}/{{ truth.study_weeks }} · policy v{{ policy_version }} · planner <code>{{ planner }}</code></p>

<div class="grid">
 <div class="card"><div class="muted">Sites monitored</div><div class="big">{{ n_sites }}</div></div>
 <div class="card"><div class="muted">Actions executed</div><div class="big">{{ n_exec }}</div></div>
 <div class="card"><div class="muted">Awaiting human approval</div><div class="big warn">{{ n_pending }}</div></div>
 <div class="card"><div class="muted">Inputs quarantined</div><div class="big">{{ n_quar }}</div></div>
 <div class="card"><div class="muted">Review items (from {{ triage.stats.raw_signals }} signals)</div><div class="big">{{ triage.stats.review_items }}</div></div>
 <div class="card"><div class="muted">Audit chain</div><div class="big {{ 'ok' if audit.valid else 'bad' }}">{{ 'VALID' if audit.valid else 'BROKEN' }}</div><div class="muted">{{ audit.entries }} entries</div></div>
</div>

<h2>1 · Human review queue (oversight meta-agent)</h2>
<div class="scroll"><table><tr><th>#</th><th>Entity</th><th>Priority</th><th>Decision needed</th><th>Signals</th><th>Headline</th></tr>
{% for e in triage.review_queue %}<tr><td>{{ loop.index }}</td><td><b>{{ e.entity }}</b></td><td>{{ '%.2f'|format(e.priority) }}</td>
<td>{% if e.requires_decision %}<span class="warn">approval</span>{% elif e.security_event %}<span class="bad">security</span>{% else %}-{% endif %}</td>
<td>{{ e.n_signals }}</td><td>{{ e.headline }}</td></tr>{% endfor %}</table></div>
<p class="muted">{{ triage.stats.digest_items }} lower-priority entities moved to the digest.</p>

<h2>2 · Predictive enrollment intelligence</h2>
<img alt="Enrollment projection" src="data:image/png;base64,{{ fig_enroll }}">
<div class="scroll"><table><tr><th>Country</th><th>Target</th><th>Enrolled</th><th>Rate recent / earlier</th><th>Projected (80% PI)</th><th>P(miss)</th><th>Shortfall</th><th>Status</th></tr>
{% for r in forecast %}<tr><td>{{ r.country }}</td><td>{{ r.target }}</td><td>{{ r.enrolled_to_date }}</td>
<td>{{ r.rate_recent }} / {{ r.rate_earlier }}</td><td>{{ '%.0f'|format(r.projected_median) }} ({{ '%.0f'|format(r.projected_p10) }}-{{ '%.0f'|format(r.projected_p90) }})</td>
<td>{{ '%.0f%%'|format(100*r.p_miss_target) }}</td><td>{{ '%.0f%%'|format(100*r.projected_shortfall) }}</td>
<td>{% if r.at_risk %}<span class="bad">at risk</span>{% elif r.slowdown_detected %}<span class="warn">slowdown</span>{% else %}<span class="ok">on track</span>{% endif %}</td></tr>{% endfor %}</table></div>

<h2>3 · Clinical data surveillance</h2>
<img alt="Site risk" src="data:image/png;base64,{{ fig_risk }}">
<div class="scroll"><table><tr><th>Site</th><th>Country</th><th>Subjects</th><th>Risk</th><th>Tier</th><th>KRI breaches</th></tr>
{% for r in top_sites %}<tr><td>{{ r.site_id }}</td><td>{{ r.country }}</td><td>{{ r.n_subjects }}</td><td>{{ '%.2f'|format(r.risk_score) }}</td><td>{{ r.risk_tier }}</td>
<td>{% for f in r.flags %}{{ f.kri }} (z={{ f.z }}){% if not loop.last %}, {% endif %}{% endfor %}</td></tr>{% endfor %}</table></div>

<h2>4 · Agent actions</h2>
{% for res in results %}<h3>{{ res.agent_id }} <span class="muted">· planner {{ res.planner }}</span></h3>
<p>Executed {{ res.executed|length }} · pending approval {{ res.pending_approval|length }} · denied {{ res.denied|length }} · failed {{ res.failed|length }} · deferred {{ res.deferred|length }}{% if res.circuit_breaker_tripped %} · <span class="bad">circuit breaker tripped</span>{% endif %}</p>
{% if res.pending_approval %}<div class="scroll"><table><tr><th>Pending approval</th><th>Entity</th><th>Arguments</th><th>Rationale</th><th>Why gated</th></tr>
{% for a in res.pending_approval %}<tr><td>{{ a.tool }}</td><td>{{ a.entity }}</td><td><code>{{ a.args }}</code></td><td>{{ a.rationale }}</td><td>{{ a.reason }}</td></tr>{% endfor %}</table></div>{% endif %}
{% endfor %}

<h2>5 · Detection quality vs. ground truth</h2>
<div class="scroll"><table><tr><th>Signal</th><th>Precision</th><th>Recall</th><th>Missed</th><th>False positives</th></tr>
{% for area, sigs in metrics.items() if area != 'oversight' %}{% for name, m in sigs.items() if m is mapping %}
<tr><td>{{ area }} · {{ name }}</td><td>{{ m.precision }}</td><td>{{ m.recall }}</td><td>{{ m.missed|join(', ') }}</td><td>{{ m.false_positives|join(', ') }}</td></tr>{% endfor %}{% endfor %}</table></div>
<p>Coerced actions executed from injected text: <b class="{{ 'ok' if metrics.security.coerced_actions_executed == 0 else 'bad' }}">{{ metrics.security.coerced_actions_executed }}</b></p>

<h2>6 · Data lineage</h2>
<div class="scroll"><table><tr><th>Table</th><th>Rows</th><th>SHA-256</th></tr>
{% for name, l in lineage.items() %}<tr><td>{{ name }}</td><td>{{ l.rows }}</td><td><code>{{ l.sha256 }}</code></td></tr>{% endfor %}</table></div>
<p class="muted">Synthetic data only. Generated by the agentic-clinical Snakemake pipeline.</p>
</body></html>""")


def build_report(*, run_id: str, truth: dict, policy_version: str, scores: pd.DataFrame,
                 forecast: pd.DataFrame, results: list[dict], triage: dict, audit: dict,
                 metrics: dict, lineage: dict) -> str:
    planners = sorted({r["planner"] for r in results})
    return TEMPLATE.render(
        run_id=run_id, truth=truth, policy_version=policy_version, planner=", ".join(planners),
        n_sites=len(scores),
        n_exec=sum(len(r["executed"]) for r in results),
        n_pending=sum(len(r["pending_approval"]) for r in results),
        n_quar=sum(len(r["quarantined"]) for r in results),
        triage=triage, audit=audit, metrics=metrics, lineage=lineage, results=results,
        forecast=forecast.sort_values("projected_shortfall", ascending=False).to_dict("records"),
        top_sites=scores.dropna(subset=["risk_score"]).head(15).to_dict("records"),
        fig_enroll=enrollment_figure(forecast), fig_risk=risk_figure(scores),
    )
