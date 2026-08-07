# Full-history smoke-test report

Provider: OpenAI
Model: gpt-4.1-2025-04-14
Settings: `{"api": "responses", "max_output_tokens": 1000, "store": false, "temperature": 0.0, "text_format": "json_object"}`

| Case | Valid JSON | Valid contract | Exact evidence | Capability behavior | Notes |
| --- | --- | --- | --- | --- | --- |
| extraction_001 | yes | yes | yes | Pass. Identifies the Marketing Associate role and cites supporting sources. | Matches the evaluation-side reference answer. |
| temporal_003 | yes | yes | yes | Pass. Uses Aryan's later correction to May 18. | Does not keep the older May 11 report. |
| conflict_004 | yes | yes | yes | Pass. Uses campaign coverage as the supported reason. | Does not present Maya's assumption about Pravin's motive as fact. |
| user_modeling_001 | yes | yes | yes | Pass with caveat. Describes a shift from marketing to product over time. | It compresses the intermediate changes and phrases the April offer as though the role started in April; the cited offer states a May 4 start. |
| abstention_001 | yes | yes | yes | Pass. Abstains because Maya's subject is not in the source history. | Returns empty evidence and a non-empty reason. |
