"""Agent runtime: perceive -> screen -> plan -> authorize -> act -> reflect.

Agents never call tools directly. Every proposed action passes through:
    1. argument validation against the tool schema
    2. the PolicyEngine (least privilege, budget, risk-vs-autonomy gate)
    3. execution with retry + circuit breaker, or the human approval queue
and every step is written to the hash-chained audit log.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_clinical.agents.audit import AuditLog
from agentic_clinical.agents.guardrails import CircuitBreaker, PolicyEngine, Verdict, screen_text
from agentic_clinical.agents.memory import InstitutionalMemory
from agentic_clinical.agents.tools import ToolError, ToolRegistry

SEVERITY_LEVELS = ("low", "medium", "high")


@dataclass
class ProposedAction:
    tool: str
    args: dict[str, Any]
    rationale: str
    entity: str                      # site_id or country the action concerns
    severity: str = "medium"         # low | medium | high
    confidence: float = 0.5          # 0..1
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError(f"severity must be one of {SEVERITY_LEVELS}")
        self.confidence = float(min(1.0, max(0.0, self.confidence)))


@dataclass
class RunContext:
    run_id: str
    registry: ToolRegistry
    policy_engine: PolicyEngine
    audit: AuditLog
    memory: InstitutionalMemory
    approvals_path: Path
    planner: Planner
    data: dict[str, Any]             # model outputs and tables the agent may perceive


@dataclass
class AgentRunResult:
    agent_id: str
    planner: str
    executed: list[dict] = field(default_factory=list)
    pending_approval: list[dict] = field(default_factory=list)
    denied: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    deferred: list[dict] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)
    circuit_breaker_tripped: bool = False

    def summary(self) -> dict:
        return {k: (len(v) if isinstance(v, list) else v) for k, v in asdict(self).items()}


class Planner(ABC):
    name = "abstract"

    @abstractmethod
    def plan(self, agent: Agent, observation: dict, ctx: RunContext) -> list[ProposedAction]:
        ...


class RulePlanner(Planner):
    """Deterministic, fully explainable planning using each agent's codified rules."""
    name = "rules"

    def plan(self, agent: Agent, observation: dict, ctx: RunContext) -> list[ProposedAction]:
        return agent.rule_plan(observation, ctx)


class Agent(ABC):
    agent_id: str = "abstract"

    # ---- to be implemented by concrete agents ------------------------------
    @abstractmethod
    def observe(self, ctx: RunContext) -> dict:
        """Build a structured observation from trusted model outputs."""

    @abstractmethod
    def rule_plan(self, observation: dict, ctx: RunContext) -> list[ProposedAction]:
        """Codified decision logic (also the fallback for LLM planning)."""

    def untrusted_texts(self, ctx: RunContext) -> list[tuple[str, str]]:
        """(entity, text) pairs of free text that must be screened. Default: none."""
        return []

    def learned(self, observation: dict, result: AgentRunResult) -> list[str]:
        return []

    # ---- runtime -------------------------------------------------------------
    def run(self, ctx: RunContext) -> AgentRunResult:
        ident = ctx.policy_engine.identity(self.agent_id)
        result = AgentRunResult(agent_id=self.agent_id, planner=ctx.planner.name)
        ctx.audit.record(self.agent_id, "run_started", {
            "autonomy": ident.autonomy.name, "allowed_tools": sorted(ident.allowed_tools),
            "max_actions": ident.max_actions, "planner": ctx.planner.name})

        observation = self.observe(ctx)
        observation["security_flags"] = self._screen(ctx, result)
        observation["memory"] = {
            e: ctx.memory.prior_flags(e, exclude_run_id=ctx.run_id)
            for e in observation.get("entities", [])
        }

        actions = self._plan(observation, ctx, result)
        breaker = CircuitBreaker(ctx.policy_engine.policy.max_consecutive_tool_errors)
        for i, action in enumerate(actions):
            if breaker.tripped:
                result.deferred.extend(self._as_record(a) for a in actions[i:])
                ctx.audit.record(self.agent_id, "actions_deferred",
                                 {"count": len(actions) - i, "reason": "circuit breaker tripped"})
                break
            self._act(action, ctx, result, breaker)
        result.circuit_breaker_tripped = breaker.tripped

        flagged = sorted({r["entity"] for r in result.executed + result.pending_approval})
        ctx.memory.remember(
            self.agent_id, ctx.run_id,
            did=[{"tool": r["tool"], "entity": r["entity"]} for r in result.executed],
            learned=self.learned(observation, result),
            deferred=[{"tool": r["tool"], "entity": r["entity"], "why": r.get("reason", "")}
                      for r in result.pending_approval + result.deferred],
            flagged_entities=flagged,
        )
        ctx.audit.record(self.agent_id, "run_completed", result.summary())
        return result

    # ---- internals -----------------------------------------------------------
    def _screen(self, ctx: RunContext, result: AgentRunResult) -> list[dict]:
        flags = []
        block = ctx.policy_engine.policy.block_on_prompt_injection
        for entity, text in self.untrusted_texts(ctx):
            s = screen_text(text)
            if not s.safe:
                rec = {"entity": entity, "reasons": list(s.reasons), "blocked": block}
                flags.append(rec)
                result.quarantined.append(rec)
                # the offending text itself is NOT logged or forwarded - only metadata
                ctx.audit.record(self.agent_id, "untrusted_input_quarantined", rec)
        return flags

    def _plan(self, observation: dict, ctx: RunContext, result: AgentRunResult) -> list[ProposedAction]:
        try:
            actions = ctx.planner.plan(self, observation, ctx)
            used = ctx.planner.name
        except Exception as exc:  # noqa: BLE001 - any planner failure must fall back to rules
            ctx.audit.record(self.agent_id, "planner_fallback",
                             {"planner": ctx.planner.name, "error": f"{type(exc).__name__}: {exc}"})
            actions = self.rule_plan(observation, ctx)
            used = "rules(fallback)"
        result.planner = used
        ctx.audit.record(self.agent_id, "plan_created", {
            "planner": used, "n_actions": len(actions),
            "actions": [{"tool": a.tool, "entity": a.entity, "severity": a.severity} for a in actions]})
        return actions

    @staticmethod
    def _as_record(a: ProposedAction, **extra) -> dict:
        return {"tool": a.tool, "args": a.args, "entity": a.entity, "severity": a.severity,
                "confidence": round(a.confidence, 3), "rationale": a.rationale,
                "evidence": a.evidence, **extra}

    def _act(self, a: ProposedAction, ctx: RunContext, result: AgentRunResult,
             breaker: CircuitBreaker) -> None:
        try:
            ctx.registry.validate(a.tool, a.args)
        except ToolError as exc:
            rec = self._as_record(a, reason=str(exc))
            result.denied.append(rec)
            ctx.audit.record(self.agent_id, "action_rejected_invalid", rec)
            if breaker.failure():
                ctx.audit.record(self.agent_id, "circuit_breaker_tripped",
                                 {"max_consecutive_errors": breaker.max_consecutive_errors})
            return

        decision = ctx.policy_engine.authorize(self.agent_id, a.tool, a.rationale)
        if decision.verdict == Verdict.DENY:
            rec = self._as_record(a, reason=decision.reason)
            result.denied.append(rec)
            ctx.audit.record(self.agent_id, "action_denied", rec)
            return
        if decision.verdict == Verdict.REQUIRE_APPROVAL:
            rec = self._as_record(a, reason=decision.reason, status="pending_approval",
                                  requested_by=self.agent_id, run_id=ctx.run_id)
            with open(ctx.approvals_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            result.pending_approval.append(rec)
            ctx.audit.record(self.agent_id, "approval_requested", rec)
            return

        last_err = None
        for attempt in (1, 2):  # one retry for transient failures
            try:
                outcome = ctx.registry.execute(a.tool, a.args)
                rec = self._as_record(a, action_id=outcome["action_id"], attempt=attempt)
                result.executed.append(rec)
                ctx.audit.record(self.agent_id, "action_executed", rec)
                breaker.success()
                return
            except ToolError as exc:
                last_err = exc
        rec = self._as_record(a, reason=str(last_err))
        result.failed.append(rec)
        ctx.audit.record(self.agent_id, "action_failed", rec)
        if breaker.failure():
            ctx.audit.record(self.agent_id, "circuit_breaker_tripped",
                             {"max_consecutive_errors": breaker.max_consecutive_errors})
