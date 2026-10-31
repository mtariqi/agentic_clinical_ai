"""Centralized guardrails: policy engine, injection screening, PII redaction,
action budgets and a circuit breaker.

The paper's autonomy-specific risks and where each is handled:
    prompt injection   -> InjectionScreen (untrusted text is quarantined, never forwarded)
    tool misuse        -> PolicyEngine (least-privilege allow-lists + arg validation in tools)
    unbounded actions  -> PolicyEngine budgets + risk/autonomy gating
    cascading failures -> CircuitBreaker
    data spillover     -> redact_pii before anything leaves for an LLM
    auditability gaps  -> every decision is written to the hash-chained AuditLog
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path

import yaml


class Risk(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Autonomy(IntEnum):
    """Capability-spectrum agency levels, ordered."""
    MINIMAL = 0       # human-led
    BASIC = 1         # conditional independence
    INTERMEDIATE = 2  # supervised initiation
    ADVANCED = 3      # continuous improvement


# Highest risk an agent may execute WITHOUT human approval, per autonomy level.
AUTONOMOUS_RISK_CEILING = {
    Autonomy.MINIMAL: 0,
    Autonomy.BASIC: Risk.LOW,
    Autonomy.INTERMEDIATE: Risk.MEDIUM,
    Autonomy.ADVANCED: Risk.HIGH,   # CRITICAL is never autonomous
}


class Verdict(str, Enum):
    EXECUTE = "execute"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str


@dataclass
class AgentIdentity:
    agent_id: str
    description: str
    autonomy: Autonomy
    max_actions: int
    allowed_tools: frozenset[str]


@dataclass
class Policy:
    version: str
    max_consecutive_tool_errors: int
    block_on_prompt_injection: bool
    redact_pii_before_llm: bool
    require_rationale: bool
    tool_risk: dict[str, Risk]
    agents: dict[str, AgentIdentity]

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        raw = yaml.safe_load(Path(path).read_text())
        g = raw.get("global", {})
        tool_risk = {name: Risk[spec["risk"].upper()] for name, spec in raw["tools"].items()}
        agents = {}
        for aid, spec in raw["agents"].items():
            unknown = set(spec["allowed_tools"]) - set(tool_risk)
            if unknown:
                raise ValueError(f"policy: agent {aid} references unknown tools {sorted(unknown)}")
            agents[aid] = AgentIdentity(
                agent_id=aid,
                description=spec.get("description", ""),
                autonomy=Autonomy[spec["autonomy"].upper()],
                max_actions=int(spec["max_actions"]),
                allowed_tools=frozenset(spec["allowed_tools"]),
            )
        return cls(
            version=str(raw.get("policy_version", "0")),
            max_consecutive_tool_errors=int(g.get("max_consecutive_tool_errors", 3)),
            block_on_prompt_injection=bool(g.get("block_on_prompt_injection", True)),
            redact_pii_before_llm=bool(g.get("redact_pii_before_llm", True)),
            require_rationale=bool(g.get("require_rationale", True)),
            tool_risk=tool_risk,
            agents=agents,
        )


@dataclass
class PolicyEngine:
    policy: Policy
    _used: dict[str, int] = field(default_factory=dict)

    def identity(self, agent_id: str) -> AgentIdentity:
        if agent_id not in self.policy.agents:
            raise PermissionError(f"unknown agent identity '{agent_id}'")
        return self.policy.agents[agent_id]

    def authorize(self, agent_id: str, tool: str, rationale: str | None) -> Decision:
        ident = self.identity(agent_id)
        if tool not in self.policy.tool_risk:
            return Decision(Verdict.DENY, f"tool '{tool}' is not registered")
        if tool not in ident.allowed_tools:
            return Decision(Verdict.DENY, f"'{agent_id}' lacks permission for '{tool}' (least privilege)")
        if self.policy.require_rationale and not (rationale and rationale.strip()):
            return Decision(Verdict.DENY, "action has no rationale")
        used = self._used.get(agent_id, 0)
        if used >= ident.max_actions:
            return Decision(Verdict.DENY, f"action budget exhausted ({ident.max_actions})")
        self._used[agent_id] = used + 1
        risk = self.policy.tool_risk[tool]
        ceiling = AUTONOMOUS_RISK_CEILING[ident.autonomy]
        if risk == Risk.CRITICAL or risk > ceiling:
            return Decision(Verdict.REQUIRE_APPROVAL,
                            f"risk {risk.name} exceeds autonomy {ident.autonomy.name}; human approval required")
        return Decision(Verdict.EXECUTE, f"risk {risk.name} within autonomy {ident.autonomy.name}")


# ---------------------------------------------------------------------------
# Untrusted-text screening
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    (r"\bignore (all |any )?(previous|prior|above) (instructions|rules|prompts?)", "instruction override"),
    (r"\b(disregard|forget|bypass|override)\b.{0,40}\b(policy|policies|rules|instructions|guardrails?)",
     "policy override"),
    (r"\bsystem( override| prompt)?\s*:", "fake system message"),
    (r"\b(you are now|act as|pretend to be)\b", "role reassignment"),
    (r"\badmin(istrator)? mode\b", "privilege escalation"),
    (r"</?\s*(data|system|assistant|user|tool)\s*>", "markup boundary injection"),
    (r"\bassistant\s*:", "fake assistant turn"),
    (r"\b(e-?mail|send|export|upload)\b.{0,40}\b(records|data|subjects?|patients?)\b", "exfiltration request"),
    (r"\b(approve|cancel|mark)\b.{0,40}\b(audit|compliant|reallocation)\b", "action coercion"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in _INJECTION_PATTERNS]


@dataclass(frozen=True)
class ScreenResult:
    safe: bool
    reasons: tuple[str, ...]


def screen_text(text: str) -> ScreenResult:
    reasons = tuple(sorted({label for rx, label in _COMPILED if rx.search(text or "")}))
    return ScreenResult(safe=not reasons, reasons=reasons)


_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[PHONE]"),
    (re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b"), "[DATE]"),
]


def redact_pii(text: str) -> str:
    for rx, token in _PII_PATTERNS:
        text = rx.sub(token, text)
    return text


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
@dataclass
class CircuitBreaker:
    max_consecutive_errors: int
    _errors: int = 0
    tripped: bool = False

    def success(self) -> None:
        self._errors = 0

    def failure(self) -> bool:
        """Register a failure; return True if the breaker has now tripped."""
        self._errors += 1
        if self._errors >= self.max_consecutive_errors:
            self.tripped = True
        return self.tripped
