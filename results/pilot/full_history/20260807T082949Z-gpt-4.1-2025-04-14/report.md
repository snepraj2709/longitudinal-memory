# Full-history smoke-test report

Provider: OpenAI
Model: gpt-4.1-2025-04-14
Settings: `{"api": "responses", "max_output_tokens": 1000, "store": false, "temperature": 0.0, "text_format": "json_object"}`

| Case | Valid JSON | Valid contract | Exact evidence | Capability behavior | Notes |
| --- | --- | --- | --- | --- | --- |
| extraction_001 | yes | yes | yes | Not reviewed. Check whether it identifies the supported memories, role and cites its source. | Ready for manual capability review. |
| temporal_003 | yes | yes | yes | Not reviewed. Check whether it uses the later explicit correction instead of the older date. | Ready for manual capability review. |
| conflict_004 | yes | yes | yes | Not reviewed. Check whether it uses the supported reason and does not state Maya's assumption as fact. | Ready for manual capability review. |
| user_modeling_001 | yes | yes | yes | Not reviewed. Check whether it describes a career change over time, not a permanent preference. | Ready for manual capability review. |
| abstention_001 | yes | no | no | Not reviewed. Check whether it abstains with empty evidence and a non-empty reason. | prediction contract: Invalid baseline prediction: - answer must be a non-empty string |
