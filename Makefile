.PHONY: install test lint run dryrun dag clean

install:            ## editable install with dev tools
	pip install -e ".[dev]"

test:               ## unit tests
	pytest

lint:
	ruff check src tests

dryrun:             ## show what Snakemake would run
	snakemake -n --cores 1

run:                ## full pipeline
	snakemake --cores 4

dag:                ## render the workflow DAG (needs graphviz)
	snakemake --dag | dot -Tsvg > docs/dag.svg

clean:
	rm -rf results logs .snakemake
