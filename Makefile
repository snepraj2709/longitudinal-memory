PYTHON ?= python3

.PHONY: test test-storage test-ingestion test-temporal test-temporal-eval test-conflict-candidates test-conflict-relations test-belief-resolution test-conflict-eval test-sessionization validate-benchmark-v1 validate-scaled-benchmark validate-load-corpus analyze-atomic-v2 dry-run-atomic-safety
test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

test-storage:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_storage_contracts tests.unit.test_ingestion_contracts \
		tests.integration.test_phase4_storage \
		tests.integration.test_ingestion_service -v

test-temporal:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_temporal_contracts \
		tests.integration.test_temporal_service -v

test-temporal-eval:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_temporal_evaluation \
		tests.integration.test_temporal_evaluation -v

test-conflict-candidates:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_conflict_candidates \
		tests.unit.test_conflict_candidate_evaluation \
		tests.integration.test_conflict_candidates -v

test-conflict-relations:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_conflict_classifier \
		tests.unit.test_conflict_relations \
		tests.unit.test_conflict_relation_evaluation \
		tests.integration.test_conflict_relations \
		tests.integration.test_conflict_relation_evaluation -v

test-belief-resolution:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_conflict_resolver \
		tests.unit.test_belief_resolution_persistence \
		tests.unit.test_belief_resolution_evaluation \
		tests.integration.test_belief_resolution \
		tests.integration.test_belief_resolution_evaluation -v

test-conflict-eval:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_phase5_conflict_evaluation \
		tests.integration.test_phase5_conflict_evaluation \
		tests.integration.test_conflict_candidates \
		tests.integration.test_conflict_relations \
		tests.integration.test_belief_resolution -v

test-sessionization:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_sessionization \
		tests.unit.test_sessionization_evaluation \
		tests.integration.test_sessionization -v

test-ingestion:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_ingestion_contracts -v

validate-benchmark-v1:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m evaluation.benchmark_release --repo-root .

validate-scaled-benchmark:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m evaluation.scaled_release --repo-root .

validate-load-corpus:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m load_testing.corpus --repo-root .

analyze-atomic-v2:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m extraction.failure_analysis --repo-root .

dry-run-atomic-safety:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m extraction.run_atomic --dry-run
