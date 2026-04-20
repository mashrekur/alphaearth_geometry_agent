# Evaluation Responses

Example system responses from the agentic evaluation described in Section 4 of the manuscript. These are a curated subset of 32 response records drawn from 6 queries (2 per complexity tier), selected to illustrate the key findings reported in the paper.

## File

`example_evaluation_responses.csv` — 32 rows, one per query–model–condition combination. Each row includes the full query text, system response text (truncated to 2000 characters), tool call count, and all five criterion scores plus the weighted composite.

## Columns

| Column | Description |
|---|---|
| `query_id` | Unique query identifier (T1_xx, T2_xx, T3_xx) |
| `tier` | Complexity tier: tier1 (single-location), tier2 (multi-step comparison), tier3 (counterfactual/geometric reasoning) |
| `query_text` | Natural language query |
| `intent` | Query intent category |
| `region` | Primary CONUS region |
| `system_model` | LLM used for tool planning and response generation |
| `condition` | Ablation condition (full, no_geometric, llm_only, paper1_deterministic) |
| `judge_model` | LLM used for scoring (Gemma-3-27B for all rows) |
| `response_text` | System response (truncated to 2000 characters) |
| `n_tool_calls` | Number of tool invocations in the reasoning chain |
| `grounding` | Score 1–5: Does the response cite retrieved embedding data? |
| `scientific_accuracy` | Score 1–5: Are interpretations consistent with validated relationships? |
| `completeness` | Score 1–5: Does it fully address the query? |
| `coherence` | Score 1–5: Is it well-structured and internally consistent? |
| `practical_utility` | Score 1–5: Is the information actionable? |
| `weighted_score` | Composite: 0.25·G + 0.25·A + 0.20·C + 0.15·H + 0.15·U |
| `geometric_grounding` | Score 1–5: Does the response reference manifold properties? (Tier 2–3 only) |

## Selected Queries

Two queries per tier, chosen to represent the findings in Section 5:

**Tier 1 — Single-location assessment**
- **T1_08**: Describe the land surface near Boise, Idaho.
- **T1_22**: What is the flood risk for New Orleans, Louisiana?

**Tier 2 — Multi-step comparison**
- **T2_04**: Compare the NDVI patterns of coastal California versus inland Nevada.
- **T2_12**: Compare the precipitation regimes of Seattle and Albuquerque.

**Tier 3 — Counterfactual and geometric reasoning**
- **T3_01**: If the Central Valley of California received Pacific Northwest levels of precipitation, what would the land surface most likely resemble?
- **T3_14**: Why might embedding-based retrieval be less reliable in the Rocky Mountain foothills compared to the Great Plains?

## Conditions Included

For each query, responses are provided under multiple ablation conditions:

- **full**: All nine tools (five retrieval-based + four geometry-aware)
- **no_geometric**: Only the five retrieval tools from Paper 1; geometry-aware tools removed
- **llm_only**: No tool access; the model responds from parametric knowledge alone
- **paper1_deterministic**: The deterministic pipeline from Paper 1 (Tier 1 only)

Where available, both Claude Sonnet 4.5 and Claude Opus 4.6 responses are included to illustrate the cross-model geometric grounding finding (Section 5.3).

## Notes

- Response text is truncated to 2000 characters in the evaluation checkpoint logs. Some longer responses may be incomplete.
- The full evaluation comprises 520 Sonnet responses (120 queries × 4 conditions + 40 deterministic) and 72 Opus responses (36 queries × 2 conditions). This file is a representative subset.
- Scoring was performed by Gemma-3-27B as a separate judge model. See Section 4.3 of the manuscript for the evaluation protocol.
