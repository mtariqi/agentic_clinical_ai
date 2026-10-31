# =============================================================================
# Agentic Clinical AI - Snakemake workflow
# =============================================================================
#   generate -> validate -> features -> surveillance ─┐
#                       └─> forecast ─────────────────┴─> agents -> oversight
#                                                         agents -> verify_audit
#   (all of the above) -> evaluate -> report
#
# Usage:
#   snakemake --cores 4                         # full pipeline
#   snakemake --cores 4 --use-conda             # in an isolated conda env
#   snakemake --cores 1 --config run_id=wk31    # namespaced run
#   snakemake --dag | dot -Tsvg > docs/dag.svg  # visualise the DAG
# =============================================================================
import os

import yaml

configfile: "config/config.yaml"

# Make the package importable without installation (pip install -e . also works).
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in [os.path.abspath("src"), os.environ.get("PYTHONPATH", "")] if p
)

RUN = config["run_id"]
DATA = f"results/{RUN}/data"
INT = f"results/{RUN}/intermediate"
AG = f"results/{RUN}/agents"
REP = f"results/{RUN}/report"
LOGS = f"logs/{RUN}"
CLI = "python -m agentic_clinical.cli"
MEMORY = config.get("memory_path", "results/memory/institutional_memory.json")

# Freeze the *effective* config (file + --config overrides) for this run, so every
# rule reads exactly the same settings and the run is reproducible later.
RUNCFG = f"results/{RUN}/run_config.yaml"
os.makedirs(os.path.dirname(RUNCFG), exist_ok=True)
_resolved = yaml.safe_dump(dict(config), sort_keys=False)
if not os.path.exists(RUNCFG) or open(RUNCFG).read() != _resolved:
    with open(RUNCFG, "w") as fh:
        fh.write(_resolved)

TABLES = ["sites", "enrollment_weekly", "subjects", "visits", "labs",
          "adverse_events", "site_comments"]

include: "workflow/rules/data.smk"
include: "workflow/rules/models.smk"
include: "workflow/rules/agents.smk"
include: "workflow/rules/report.smk"


rule all:
    input:
        f"{REP}/report.html",
        f"{REP}/metrics.json",
        f"{AG}/audit_verification.json",
