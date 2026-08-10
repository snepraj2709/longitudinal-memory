# Demo runbook

## 1. What the demonstration proves

This demo proves that the repository has a deterministic evaluation harness for longitudinal memory work. It can validate frozen benchmark data, build full-history smoke prompts, inspect prior B1 baseline results, and run a Docker-backed retrieval baseline gate.

It does not prove that a finished Personal AI memory system exists. Memory-based answer generation currently abstains on the development package set, and the B0-B7 frozen comparison is not complete.

## 2. Prerequisites

- macOS or Linux shell.
- Python environment with the repository dependencies. This checkout uses `.venv-storage/bin/python` for the commands below.
- Docker Desktop or Docker Engine for PostgreSQL-backed retrieval tests.
- No OpenAI key is required for the safe demo commands.

## 3. Installation steps

The repository currently has no README and no single documented install command. In this checkout, the working Python environment already exists at `.venv-storage/`.

For a fresh machine, add a README-backed setup before sharing externally. The likely setup is:

```bash
python3 -m venv .venv-storage
.venv-storage/bin/pip install -r requirements-storage.txt
```

This has not been verified from a clean machine during this audit.

## 4. Environment setup without exposing secrets

Do not print or commit API keys. The deterministic demo does not need `OPENAI_API_KEY`.

Provider-backed historical runners are approval-gated. The safe demo should use only validators, dry runs, and already committed result artifacts.

## 5. Smallest useful evaluation command

There is no current one-command founder demo.

Use these commands today:

```bash
make validate-scaled-benchmark PYTHON=.venv-storage/bin/python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv-storage/bin/python -m evaluation.smoke --dry-run
make test-retrieval-baselines PYTHON=.venv-storage/bin/python
```

Recommended command to add:

```bash
make demo-audit
```

`make demo-audit` should run the scaled validator, smoke dry-run, retrieval summary check, and a read-only metrics summary from existing result files. It should not call a provider.

## 6. Expected output

`make validate-scaled-benchmark` should print a JSON object with:

- `status`: `passed`
- `benchmark_version`: `scaled_v1`
- `users`: `10`
- `sources`: `100`
- `qa`: `500`
- `summaries`: `50`
- `interactive_scenarios`: `20`

`evaluation.smoke --dry-run` should print five case rows:

- `extraction_001`
- `temporal_003`
- `conflict_004`
- `user_modeling_001`
- `abstention_001`

Each row should show `observation_count: 72`.

`make test-retrieval-baselines` should run 132 tests and end with `OK`. It starts a disposable PostgreSQL container and removes it afterward.

`results/retrieval/retrieval-quality-development-v1/scorecard.json` reports 8 queries, 24 retrieval results, 0 runtime failures, and 63 quality rows.

`results/answering/memory-answer-quality-development-v1/checks.json` reports `abstained_count: 24`, `citation_count: 0`, `factual_prediction_count: 0`, and `non_null_metric_count: 0`, showing the current memory-answer path is structurally abstaining.

## 7. Demonstration queries

Use these pilot cases when explaining the system:

| Case | Query | What it shows |
| --- | --- | --- |
| `temporal_003` | "What is the corrected date for Aryan's job start in Bengaluru?" | Correct answer requires preserving the old May 11 report and Aryan's May 18 correction. |
| `conflict_002` | "Did Pravin reject Maya's first Kids Spark project?" | Correct answer must distinguish Maya's initial feeling from Pravin's later clarification. |
| `user_modeling_001` | "How did Maya's career direction change from April to the end of September?" | The baseline partially answers but misses intermediate evidence and makes a wrong date claim. |

## 8. Inspect retrieved evidence and failure labels

Use these files:

- `results/pilot/b1-full-history/failure_analysis.jsonl` for per-case B1 failure labels, missed evidence, wrong-date notes, and reviewer explanations.
- `results/pilot/b1-full-history/manual_review.jsonl` for reviewed B1 answers and cited evidence.
- `data/pilot/evaluation/eval_answer.jsonl` for expected answers and gold evidence.
- `results/retrieval/retrieval-quality-development-v1/per-query.jsonl` for B2-B4 retrieval metrics per query.
- `results/retrieval/retrieval-quality-development-v1/scorecard.json` for aggregate retrieval scores.
- `results/answering/memory-answer-quality-development-v1/scorecard.json` for current memory-answer null metrics and structural abstentions.
- `results/abstention/b7-evaluation-development-v1/scorecard.json` for B6/B7 coverage and over-abstention metrics.

## 9. Known limitations

- No README, `.env.example`, UI, hosted demo, or one-command demo exists.
- Full `make test` currently fails six drift-sensitive tests in this worktree.
- The current worktree contains uncommitted Step 10.3 source/result changes.
- Docker is required for the retrieval gate.
- OpenAI-backed runs are intentionally approval-gated and should not be part of the default demo.
- B1 is a full-history baseline, not a memory architecture.
- B2-B4 retrieval works on development data, but memory-answer generation currently abstains structurally.
- B7 does not currently improve coverage over B6.

## 10. Troubleshooting

If Docker access fails with a socket permission error, start Docker Desktop and rerun:

```bash
make test-retrieval-baselines PYTHON=.venv-storage/bin/python
```

If `make test` fails with allowlist or protected-hash assertions, inspect the current worktree:

```bash
git status --short
```

Those tests are sensitive to uncommitted result and source drift. Do not fix them by deleting or reverting files unless the owner approves the cleanup.

If `make analyze-atomic-v2` fails with "refusing to overwrite non-empty output", that is expected in this checkout because `results/phase3/atomic-extraction-v2-failure-analysis-v1` already exists. Use the existing result files or add a new output directory before rerunning.
