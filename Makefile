PYTHON ?= python3

.PHONY: test validate-benchmark-v1
test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

validate-benchmark-v1:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m evaluation.benchmark_release --repo-root .
