# ---- Data: synthetic generation + validation gate ---------------------------

rule generate_data:
    output:
        expand(f"{DATA}/{{t}}.csv", t=TABLES),
        truth=f"{DATA}/ground_truth.json",
    params:
        outdir=DATA,
    log:
        f"{LOGS}/generate_data.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} generate --config {RUNCFG} --outdir {params.outdir} > {log} 2>&1"


rule validate_data:
    """Schema, range and referential-integrity checks. Fails the run on any error."""
    input:
        expand(f"{DATA}/{{t}}.csv", t=TABLES),
    output:
        f"{INT}/validation.json",
    params:
        datadir=DATA,
    log:
        f"{LOGS}/validate_data.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} validate --datadir {params.datadir} --out {output} > {log} 2>&1"
