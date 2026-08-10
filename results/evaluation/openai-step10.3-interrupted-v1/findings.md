# Interrupted OpenAI Step 10.3 recovery

Status: preserved historical failure, not a benchmark result.

The OpenAI extraction batch completed 100 requests and produced 100 accepted extraction records. It cost $0.203612. The new Qwen comparison will not reuse those records. It will regenerate extraction so the scaled Qwen run uses one model family end to end.

The first B0 QA attempt made 20 requests. Every request failed and the prediction file is empty. Early requests were recorded as generic provider failures. Later responses showed that the serialized input did not contain the provider's required JSON instruction. This attempt has no score.

The corrected B0 QA attempt made four requests and cost $0.004876. All four responses failed the frozen answer contract with `output_status_or_body_invalid`. Its prediction file is also empty. This attempt has no score.

Historical OpenAI spend is $0.4399284, including $0.2314404 before Step 10.3. No OpenAI request was made after the Qwen recovery goal began.

Future OpenAI execution is closed. The historical runner refuses its built-in live client before it can read an environment file or create output. Tests can still pass an explicit fake client. The existing completed extraction directory refuses overwrite, and a completed resume returns its checkpoint without a provider call.

The next comparison series is `qwen35-27b-fp8-v1`. It will use new output paths and regenerate extraction. OpenAI artifacts remain available only for audit and cost history.
