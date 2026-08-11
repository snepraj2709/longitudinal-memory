# Demo runbook

## What this demo shows

The app shows how the benchmark represents source history, changing memory state, retrieved evidence, answers, abstentions, and review failures. It replays committed pilot and development artifacts. It does not run a model, query a database, or claim that the frozen B0-B7 comparison is complete.

## Local start

From the repository root:

```bash
make demo
```

Open `http://127.0.0.1:8000`. No `.env`, API key, Docker daemon, GPU, or database is required. The first run installs the demo dependencies and compiles the web app.

Run focused verification separately:

```bash
make test-demo
```

## Ten-minute walkthrough

1. Open **A corrected date**. Replay the source timeline and show how May 11 becomes historical after Aryan directly states May 18. The answer is correct, but review notes that its citation omitted the earlier report.
2. Open **Experience versus fact**. Show that Maya's interpretation is preserved while Pravin's clarification resolves the project status.
3. Open **A changing career direction**. Show the April, June, July, and September states. The B1 answer gets the direction right but skips the intermediate trajectory and says the role started in April instead of May 4.
4. Open **Evidence is absent**. Show that the system abstains instead of inferring Maya's college subject.
5. Open **Scorecard**. Compare B1 answer quality with B2-B4 retrieval metrics, then point out B6/B7 over-abstention. Keep the interrupted OpenAI run and unrun Qwen series separate.

## Honest labels

- `openai-gpt41-v1`: `interrupted_not_scored`. Historical cost is `$0.4399284`. Extraction completed, but B0 produced no valid prediction or score.
- `qwen35-27b-fp8-v1`: `configured_not_run`. Its current cost and request count are zero.
- B1: scored on the 25-case pilot.
- B2-B4: retrieval-only development scores.
- B6-B7: partial four-case abstention evaluation.
- B5: not scored in the demo bundle.

There is no composite score and no direct model-superiority claim.

## Docker

```bash
docker compose up --build demo
```

The image serves the same app at `http://127.0.0.1:8000`. Its runtime contains the sanitized bundle, compiled React assets, and FastAPI only. It does not include benchmark gold, oracle data, review files, source corpora, provider credentials, or a write endpoint.

## Hosted Railway demo

The hosted demo is available at [longitudinal-memory-benchmark.up.railway.app](https://longitudinal-memory-benchmark.up.railway.app). `Dockerfile` and `railway.toml` define the single service and its `/healthz` check. The service needs no volume, database, GPU, or environment secret.

Verify the deployment with:

```bash
curl --fail https://longitudinal-memory-benchmark.up.railway.app/healthz
curl --fail https://longitudinal-memory-benchmark.up.railway.app/api/demo/cases
```

Check that responses include `X-Robots-Tag: noindex, nofollow, noarchive`, unsupported API methods return 405, and Railway has no provider credentials configured.

## Rebuild the public bundle

```bash
make demo-data
```

The deterministic builder reads versioned local results and writes `results/demo/demo-v1/bundle.json`. Tests verify byte-stable output and reject oracle IDs, API-key names, and reviewer identifiers in the bundle.

## Troubleshooting

- If port 8000 is occupied: `PORT=8010 make demo`.
- If the frontend says artifacts could not be loaded, run `make demo-data` and restart.
- If web dependencies drift, remove `web/node_modules/.ready` and rerun `make demo-build`.
- Historical OpenAI runners remain closed. Do not add an API key to make the demo work.
