PYTHON ?= python3

.PHONY: test validate-benchmark-v1 validate-scaled-benchmark
test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

validate-benchmark-v1:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m evaluation.benchmark_release --repo-root .

validate-scaled-benchmark:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m evaluation.scaled_release --repo-root .
