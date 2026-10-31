# Architecture

## Layers

The system separates three concerns that are often blurred in "agentic" products.

**Detection (data science).** Deterministic, testable models turn data into findings. `features.py` computes site Key Risk Indicators, `models/surveillance.py` scores sites with credibility-weighted robust z-scores plus an Isolation Forest, and `models/enrollment.py` projects country enrollment with a Poisson-Gamma Monte Carlo and a binomial rate-change test. Nothing here takes action.

**Decision (agents).** Agents perceive model outputs and decide what to do. Planning is either codified rules (default, fully explainable) or Claude (optional). Either way, the output is a list of `ProposedAction`s with a rationale, severity, confidence and evidence.

**Control (governance).** Every proposed action passes through the same runtime: schema validation, the policy engine, execution with retry and a circuit breaker, or the human approval queue. All of it is written to the hash-chained audit log. Governance is configuration (`config/policies.yaml`), so changing what an agent may do is a reviewed config change, not a code change.

## Risk handling

| Risk | Control | Where |
|---|---|---|
| Prompt injection via free text | Pattern screen; offending text is quarantined and never logged or forwarded; a security notification is raised instead | `guardrails.screen_text`, `Agent._screen` |
| Tool misuse | Per-agent allow-list; strict argument schemas (including rejecting `bool` for `int`); domain checks (known sites/countries, ranges) | `PolicyEngine.authorize`, `ToolSpec.validate` |
| Unbounded decision-making | Action budgets per agent; risk tier vs autonomy level; `critical` always needs a human | `PolicyEngine.authorize` |
| Cascading failures | Circuit breaker after N consecutive tool errors; remaining actions are deferred and logged, not dropped | `CircuitBreaker`, `Agent.run` |
| Data spillover | PII redaction of every string before an LLM call; LLM never sees quarantined text | `redact_pii`, `llm._redact` |
| LLM hallucinated actions | Strict JSON parsing, allow-list and schema check per action; rejects are logged | `llm.parse_actions` |
| Auditability gaps | Append-only JSONL, SHA-256 chain, `verify_audit` gate in the pipeline, human decisions in the same chain | `agents/audit.py`, `cli decide` |
| Oversight fatigue | Per-entity consolidation, priority scoring with corroboration bonus, bounded queue, digest for the rest | `agents/oversight.py` |

## Autonomy model

| Level | Highest risk executed autonomously |
|---|---|
| minimal | none (recommend only) |
| basic | low |
| intermediate | medium |
| advanced | high |

`critical` tools are never autonomous at any level.

## Statistical notes

Surveillance z-scores use median/MAD and are multiplied by a credibility weight `n / (n + k)` (n = visits) so sites with very few visits do not dominate the alert list. Direction matters: low lab variance and low AE rates are the risky tails (possible fabrication or under-reporting).

Enrollment uses the last `recent_window_weeks` to estimate the current rate with a Gamma(k + 0.5, w) posterior, simulates final enrollment, and reports P(miss target), expected shortfall and an 80% interval. A country is *at risk* when shortfall ≥ threshold and P(miss) ≥ 0.8; a *slowdown* is flagged when the recent rate is < 0.7× the earlier rate with p < 0.01, which typically fires before cumulative reports show a problem.

## Scaling beyond this repository

The tool registry, policy engine, audit log and memory are shared services used by every agent, which is the "shared, governed backbone" the paper argues for. Moving to production means replacing the staged outbox with real integrations (CTMS, EDC, eTMF) behind the same `ToolSpec` interface, backing identities with your IAM, shipping the audit log to WORM storage, and putting the whole system under computer-system validation.
