from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_clinical.agents.audit import AuditLog  # noqa: E402
from agentic_clinical.agents.core import RunContext, RulePlanner  # noqa: E402
from agentic_clinical.agents.guardrails import Policy, PolicyEngine  # noqa: E402
from agentic_clinical.agents.memory import InstitutionalMemory  # noqa: E402
from agentic_clinical.agents.tools import Outbox, ToolRegistry  # noqa: E402
from agentic_clinical.data.synthetic import StudyConfig, generate_study  # noqa: E402
from agentic_clinical.features import build_site_kris  # noqa: E402
from agentic_clinical.models.enrollment import forecast_enrollment  # noqa: E402
from agentic_clinical.models.surveillance import score_sites  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "policies.yaml"


@pytest.fixture(scope="session")
def study():
    return generate_study(StudyConfig(), seed=7)


@pytest.fixture(scope="session")
def model_outputs(study):
    tables, _ = study
    scores = score_sites(build_site_kris(tables), seed=7)
    fc = forecast_enrollment(tables["enrollment_weekly"], tables["sites"],
                             current_week=30, study_weeks=52, n_sims=2000, seed=7)
    return scores, fc


def make_context(tmp_path: Path, study, model_outputs, run_id: str = "test",
                 planner=None, memory_path: Path | None = None) -> RunContext:
    tables, _ = study
    scores, fc = model_outputs
    sites: pd.DataFrame = tables["sites"]
    policy = Policy.load(POLICY_PATH)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "approvals.jsonl").touch()
    return RunContext(
        run_id=run_id,
        registry=ToolRegistry(Outbox(tmp_path / "outbox.jsonl"), set(sites["site_id"]),
                              set(sites["country"])),
        policy_engine=PolicyEngine(policy),
        audit=AuditLog(tmp_path / "audit.jsonl", run_id=run_id, policy_version=policy.version),
        memory=InstitutionalMemory(memory_path or tmp_path / "memory.json"),
        approvals_path=tmp_path / "approvals.jsonl",
        planner=planner or RulePlanner(),
        data={"site_scores": scores, "enrollment_forecast": fc,
              "site_comments": tables["site_comments"]},
    )
