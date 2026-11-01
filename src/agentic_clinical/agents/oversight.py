"""Oversight meta-agent: fights oversight fatigue.

Agents escalate defensively; if every alert reaches a human, reviews become
shallow. This meta-agent consolidates all agent outputs per entity, scores
priority, and builds a short, ranked review queue. Rules:
  * Anything awaiting a human decision (approval queue) is ALWAYS shown -
    it can be ranked but never suppressed.
  * Security events (quarantined input) are always shown.
  * Remaining items fill the queue up to ``max_items`` if priority >= ``min_priority``;
    the rest go to a digest that is still fully auditable.
  * Issues that persist from an earlier cycle get a priority bonus, so they are
    not buried under new alerts of the same strength.
  * Requests a human already decided (suppressed by memory) are listed
    separately for transparency, but do not ask for a decision again.
"""

from __future__ import annotations

from collections import defaultdict

from agentic_clinical.agents.core import AgentRunResult

SEVERITY_W = {"low": 0.3, "medium": 0.6, "high": 1.0}
PERSISTENCE_BONUS = 0.1


def _item_priority(item: dict, kind: str) -> float:
    sev = SEVERITY_W.get(item.get("severity", "medium"), 0.6)
    conf = float(item.get("confidence", 0.5))
    kind_w = {"pending_approval": 1.0, "security": 1.0, "executed": 0.7}[kind]
    bonus = PERSISTENCE_BONUS if item.get("evidence", {}).get("persistent") else 0.0
    return min(1.0, sev * (0.5 + 0.5 * conf) * kind_w + bonus)


def triage(results: list[AgentRunResult], max_items: int = 10, min_priority: float = 0.25) -> dict:
    groups: dict[str, dict] = defaultdict(lambda: {"items": [], "agents": set(),
                                                   "requires_decision": False, "security": False})
    raw = 0
    for res in results:
        for kind, items in (("executed", res.executed), ("pending_approval", res.pending_approval)):
            for it in items:
                raw += 1
                g = groups[it["entity"]]
                g["agents"].add(res.agent_id)
                g["items"].append({"kind": kind, "tool": it["tool"], "severity": it["severity"],
                                   "rationale": it["rationale"],
                                   "persistent": bool(it.get("evidence", {}).get("persistent")),
                                   "priority": round(_item_priority(it, kind), 4)})
                g["requires_decision"] |= kind == "pending_approval"
        for q in res.quarantined:
            raw += 1
            g = groups[q["entity"]]
            g["agents"].add(res.agent_id)
            g["security"] = True
            g["items"].append({"kind": "security", "tool": "input_screen", "severity": "high",
                               "rationale": f"Quarantined input: {', '.join(q['reasons'])}",
                               "priority": round(_item_priority({"severity": "high",
                                                                 "confidence": 0.9}, "security"), 4)})

    entities = []
    for entity, g in groups.items():
        prios = sorted((i["priority"] for i in g["items"]), reverse=True)
        # corroboration bonus: several independent signals on one entity
        priority = min(1.0, prios[0] + 0.05 * (len(prios) - 1))
        entities.append({
            "entity": entity,
            "priority": round(priority, 4),
            "requires_decision": g["requires_decision"],
            "security_event": g["security"],
            "n_signals": len(g["items"]),
            "agents": sorted(g["agents"]),
            "persistent": any(i.get("persistent") for i in g["items"]),
            "headline": max(g["items"], key=lambda i: i["priority"])["rationale"],
            "items": sorted(g["items"], key=lambda i: -i["priority"]),
        })
    entities.sort(key=lambda e: (not e["requires_decision"], not e["security_event"], -e["priority"]))

    mandatory = [e for e in entities if e["requires_decision"] or e["security_event"]]
    optional = [e for e in entities if e not in mandatory and e["priority"] >= min_priority]
    queue = mandatory + optional[: max(0, max_items - len(mandatory))]
    queued = {e["entity"] for e in queue}
    digest = [e for e in entities if e["entity"] not in queued]
    suppressed = [{"agent": r.agent_id, "entity": s["entity"], "tool": s["tool"],
                   "reason": s["reason"]}
                  for r in results for s in getattr(r, "suppressed", [])]
    return {
        "review_queue": queue,
        "digest": digest,
        "suppressed_by_memory": suppressed,
        "stats": {
            "raw_signals": raw,
            "entities": len(entities),
            "review_items": len(queue),
            "digest_items": len(digest),
            "mandatory_items": len(mandatory),
            "suppressed_by_memory": len(suppressed),
            "reduction_vs_raw": round(1 - len(queue) / raw, 4) if raw else 0.0,
        },
    }
