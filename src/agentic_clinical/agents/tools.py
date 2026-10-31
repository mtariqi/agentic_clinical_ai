"""Tool registry: the only way agents can affect the outside world.

Every tool has a typed argument schema that is validated before execution
(LLM-proposed calls with wrong or extra arguments are rejected). Effects are
written to a staged *outbox* (JSONL) rather than to production systems, so each
action is traceable and reversible: a downstream integration job commits the
outbox to CTMS/EDC only after the run has been reviewed.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ArgSchema = dict[str, type | tuple[type, ...]]


class ToolError(Exception):
    """Raised for invalid arguments or failed execution."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: ArgSchema
    fn: Callable[..., dict]

    def validate(self, args: dict[str, Any]) -> None:
        missing = set(self.args) - set(args)
        extra = set(args) - set(self.args)
        if missing:
            raise ToolError(f"{self.name}: missing arguments {sorted(missing)}")
        if extra:
            raise ToolError(f"{self.name}: unexpected arguments {sorted(extra)}")
        for k, typ in self.args.items():
            types = typ if isinstance(typ, tuple) else (typ,)
            v = args[k]
            if not isinstance(v, types) or (isinstance(v, bool) and bool not in types):
                names = "|".join(t.__name__ for t in types)
                raise ToolError(f"{self.name}: argument '{k}' must be {names}, got {type(v).__name__}")

    def describe(self) -> dict:
        def tname(t):
            return "|".join(x.__name__ for x in t) if isinstance(t, tuple) else t.__name__
        return {"name": self.name, "description": self.description,
                "args": {k: tname(t) for k, t in self.args.items()}}


class Outbox:
    """Append-only staging area for tool effects."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def stage(self, tool: str, record: dict) -> dict:
        rec = {"action_id": str(uuid.uuid4()), "tool": tool, "status": "staged", **record}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec


class ToolRegistry:
    SEVERITIES = frozenset({"low", "medium", "high"})

    def __init__(self, outbox: Outbox, known_sites: set[str], known_countries: set[str]):
        self.outbox = outbox
        self.sites = known_sites
        self.countries = known_countries
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    # ---- public API --------------------------------------------------------
    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(f"unknown tool '{name}'")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalogue(self, allowed: set[str] | frozenset[str]) -> list[dict]:
        return [self._tools[n].describe() for n in sorted(allowed) if n in self._tools]

    def validate(self, name: str, args: dict) -> None:
        self.get(name).validate(args)

    def execute(self, name: str, args: dict) -> dict:
        spec = self.get(name)
        spec.validate(args)
        return spec.fn(**args)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    # ---- domain checks -----------------------------------------------------
    def _site(self, site_id: str) -> None:
        if site_id not in self.sites:
            raise ToolError(f"unknown site '{site_id}'")

    def _country(self, c: str) -> None:
        if c not in self.countries:
            raise ToolError(f"unknown country '{c}'")

    def _severity(self, s: str) -> None:
        if s not in self.SEVERITIES:
            raise ToolError(f"severity must be one of {sorted(self.SEVERITIES)}")

    # ---- tool implementations ----------------------------------------------
    def _register_defaults(self) -> None:
        def raise_data_query(site_id: str, kri: str, detail: str) -> dict:
            self._site(site_id)
            return self.outbox.stage("raise_data_query", {"site_id": site_id, "kri": kri, "detail": detail})

        def notify_study_team(topic: str, message: str, severity: str) -> dict:
            self._severity(severity)
            return self.outbox.stage("notify_study_team",
                                     {"topic": topic, "message": message, "severity": severity})

        def flag_site_for_review(site_id: str, reason: str, risk_score: float) -> dict:
            self._site(site_id)
            if not 0 <= risk_score <= 1:
                raise ToolError("risk_score must be in [0, 1]")
            return self.outbox.stage("flag_site_for_review",
                                     {"site_id": site_id, "reason": reason, "risk_score": risk_score})

        def schedule_targeted_sdv(site_id: str, scope: str, percent: int) -> dict:
            self._site(site_id)
            if not 1 <= percent <= 100:
                raise ToolError("percent must be in [1, 100]")
            return self.outbox.stage("schedule_targeted_sdv",
                                     {"site_id": site_id, "scope": scope, "percent": percent})

        def recommend_enrollment_reallocation(from_country: str, to_country: str, n_subjects: int) -> dict:
            self._country(from_country)
            self._country(to_country)
            if from_country == to_country or n_subjects <= 0:
                raise ToolError("reallocation needs two different countries and n_subjects > 0")
            return self.outbox.stage("recommend_enrollment_reallocation",
                                     {"from_country": from_country, "to_country": to_country,
                                      "n_subjects": n_subjects})

        def trigger_for_cause_audit(site_id: str, reason: str) -> dict:
            self._site(site_id)
            return self.outbox.stage("trigger_for_cause_audit", {"site_id": site_id, "reason": reason})

        for spec in [
            ToolSpec("raise_data_query", "Open a data-management query for a site KRI breach",
                     {"site_id": str, "kri": str, "detail": str}, raise_data_query),
            ToolSpec("notify_study_team", "Send a notification to the study team",
                     {"topic": str, "message": str, "severity": str}, notify_study_team),
            ToolSpec("flag_site_for_review", "Flag a site for CRA/central-monitor review",
                     {"site_id": str, "reason": str, "risk_score": (int, float)}, flag_site_for_review),
            ToolSpec("schedule_targeted_sdv", "Schedule targeted source-data verification",
                     {"site_id": str, "scope": str, "percent": int}, schedule_targeted_sdv),
            ToolSpec("recommend_enrollment_reallocation",
                     "Recommend moving enrollment targets between countries",
                     {"from_country": str, "to_country": str, "n_subjects": int},
                     recommend_enrollment_reallocation),
            ToolSpec("trigger_for_cause_audit", "Request a for-cause GCP audit of a site",
                     {"site_id": str, "reason": str}, trigger_for_cause_audit),
        ]:
            self.register(spec)
