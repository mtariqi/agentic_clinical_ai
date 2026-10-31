"""Institutional memory: every agent records what it did, what it learned and
what it deferred, building a collective memory across runs.

Memory is advisory: agents use it to escalate *persistent* issues (a site that
was flagged last cycle and is still flagged) rather than re-raising the same
query every week - one of the main drivers of oversight fatigue.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class InstitutionalMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text() or "[]")

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
