"""Institutional memory: every agent records what it did, what it learned and
what it deferred, building a collective memory across runs.

Memory is advisory: agents use it to escalate *persistent* issues (a site that
was flagged last cycle and is still flagged) rather than re-raising the same
query every week - one of the main drivers of oversight fatigue.

Human decisions are remembered too (in ``human_decisions.json`` next to the
memory file). A request a human already rejected, or an identical request a
human already approved, is not put back in the approval queue while the
decision is still fresh (``decision_ttl_days``).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path


class InstitutionalMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.path.with_name("human_decisions.json")
        self.entries: list[dict] = []
        self.decisions: list[dict] = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text() or "[]")
        if self.decisions_path.exists():
            self.decisions = json.loads(self.decisions_path.read_text() or "[]")

    # ---- agent memory --------------------------------------------------------
    def prior_flags(self, entity: str, exclude_run_id: str | None = None) -> int:
        """How many earlier runs flagged this entity (site or country)."""
        return sum(1 for e in self.entries
                   if entity in e.get("flagged_entities", []) and e.get("run_id") != exclude_run_id)

    def remember(self, agent_id: str, run_id: str, did: list[dict], learned: list[str],
                 deferred: list[dict], flagged_entities: list[str]) -> dict:
        entry = {
            "agent_id": agent_id,
            "run_id": run_id,
            "did": did,
            "learned": learned,
            "deferred": deferred,
            "flagged_entities": sorted(set(flagged_entities)),
            "action_counts": dict(Counter(d["tool"] for d in did)),
        }
        # one entry per (agent, run): re-running a run replaces, never duplicates
        self.entries = [e for e in self.entries
                        if not (e["agent_id"] == agent_id and e["run_id"] == run_id)]
        self.entries.append(entry)
        self.path.write_text(json.dumps(self.entries, indent=2, default=str))
        return entry

    # ---- human decisions -----------------------------------------------------
    def record_decision(self, *, entity: str, tool: str, args: dict, decision: str,
                        approver: str, reason: str, run_id: str,
                        ts: datetime | None = None) -> dict:
        if decision not in ("approve", "reject"):
            raise ValueError("decision must be 'approve' or 'reject'")
        rec = {
            "entity": entity, "tool": tool, "args": args, "decision": decision,
            "approver": approver, "reason": reason, "run_id": run_id,
            "ts": (ts or datetime.now(UTC)).isoformat(),
        }
        self.decisions.append(rec)
        self.decisions_path.write_text(json.dumps(self.decisions, indent=2, default=str))
        return rec

    def find_decision(self, entity: str, tool: str, args: dict, ttl_days: float,
                      now: datetime | None = None) -> dict | None:
        """Latest still-valid human decision that settles this request, if any.

        * a rejection settles any request for the same tool on the same entity
          (the human said "not this action here"), whatever the arguments;
        * an approval only settles an *identical* request (same arguments) -
          if the agent now proposes something different, a human must look again.
        """
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(days=ttl_days)
        for d in reversed(self.decisions):  # newest first
            if d["entity"] != entity or d["tool"] != tool:
                continue
            if datetime.fromisoformat(d["ts"]) < cutoff:
                return None  # newest relevant decision has expired
            if d["decision"] == "reject" or d["args"] == args:
                return d
            return None  # approved, but for different arguments
        return None
