PYTHON ?= python3
DEMO_PYTHON ?= .venv-demo/bin/python

.PHONY: test test-storage test-ingestion test-temporal test-temporal-eval test-conflict-candidates test-conflict-relations test-belief-resolution test-conflict-eval test-sessionization test-grounded-summaries test-durative-claims test-retrieval-index test-retrieval-planning test-retrieval-baselines test-qwen-materialization test-qwen-v2 validate-benchmark-v1 validate-scaled-benchmark validate-load-corpus analyze-atomic-v2 dry-run-atomic-safety demo demo-data demo-build test-demo
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

test-grounded-summaries:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_grounded_summaries \
		tests.unit.test_summary_persistence \
		tests.unit.test_grounded_summary_evaluation \
		tests.integration.test_grounded_summary_persistence \
		tests.integration.test_grounded_summary_evaluation -v

test-durative-claims:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_durative_claims \
		tests.unit.test_durative_evaluation \
		tests.integration.test_durative_claim_persistence \
		tests.integration.test_durative_claim_evaluation -v

test-retrieval-index:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_retrieval_index \
		tests.unit.test_retrieval_index_evaluation \
		tests.integration.test_retrieval_index \
		tests.integration.test_retrieval_index_evaluation -v

test-retrieval-planning:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_retrieval_index \
		tests.unit.test_retrieval_index_evaluation \
		tests.integration.test_retrieval_index \
		tests.integration.test_retrieval_index_evaluation \
		tests.unit.test_retrieval_query_planning \
		tests.unit.test_retrieval_query_evaluation \
		tests.integration.test_retrieval_query_planning \
		tests.integration.test_retrieval_query_evaluation -v

test-retrieval-baselines:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_retrieval_index \
		tests.unit.test_retrieval_index_evaluation \
		tests.integration.test_retrieval_index \
		tests.integration.test_retrieval_index_evaluation \
		tests.unit.test_retrieval_query_planning \
		tests.unit.test_retrieval_query_evaluation \
		tests.integration.test_retrieval_query_planning \
		tests.integration.test_retrieval_query_evaluation \
		tests.unit.test_retrieval_baselines \
		tests.unit.test_retrieval_baseline_evaluation \
		tests.integration.test_retrieval_baselines \
		tests.integration.test_retrieval_baseline_evaluation -v

test-qwen-materialization:
	@set -eu; \
	trap 'docker compose down -v >/dev/null' EXIT; \
	docker compose up -d --wait storage-db; \
	STORAGE_DATABASE_URL=postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory \
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.integration.test_qwen_materialization -v

test-qwen-v2:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest \
		tests.unit.test_qwen_v2_contract \
		tests.unit.test_qwen_materialization \
		tests.unit.test_qwen_execution \
		tests.unit.test_qwen_scoring \
		tests.unit.test_qwen_lifecycle \
		tests.unit.test_qwen_pipeline \
		tests.unit.test_vllm_client -v
	$(MAKE) test-qwen-materialization PYTHON=$(PYTHON)

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

.venv-demo/.ready: requirements-demo.txt
	python3 -m venv .venv-demo
	.venv-demo/bin/pip install -r requirements-demo.txt
	touch $@

web/node_modules/.ready: web/package.json web/package-lock.json
	npm --prefix web install
	touch $@

demo-data: .venv-demo/.ready
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(DEMO_PYTHON) -m demo.bundle --repo-root .

demo-build: web/node_modules/.ready
	npm --prefix web run build

demo: .venv-demo/.ready demo-build demo-data
	PYTHONPATH=src $(DEMO_PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port $${PORT:-8000}

test-demo: .venv-demo/.ready demo-data
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(DEMO_PYTHON) -m unittest \
		tests.demo.test_bundle tests.api.test_demo_api \
		tests.unit.test_vllm_client tests.unit.test_qwen_series \
		tests.unit.test_qwen_compatibility tests.unit.test_qwen_preflight \
		tests.unit.test_qwen_benchmark -v
