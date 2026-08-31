# Controlled Grounding-Prompt Tuning Report

This report compares one generic grounding-prompt revision against the preserved baseline. The same 15 evaluation cases, models, stored chunks, Top-K value, retrieval implementation, and generation settings were used in both runs.

Automated checks cover source-file retrieval and exact unsupported-question fallback behavior only. Semantic answer correctness remains a human-review judgment.

## Change under test

Only `GROUNDING_SYSTEM_INSTRUCTION` in `rag_service.py` changed. The revised instruction:

- permits faithful paraphrases when retrieved context directly supports an answer;
- says not to refuse merely because the question and context use different wording;
- prohibits inventing causal, explanatory, or logical relationships between separate facts;
- emphasizes that an unsupported answer must be exactly `The information is not available in the supplied context.` with no surrounding text;
- tells the model not to generate citations, references, source sections, filenames, or chunk labels because the application displays retrieval sources separately;
- requests direct, factual answers of one to three sentences.

Generation settings were not changed: temperature remains `0.0`, random seed remains `0`, and maximum output remains `96` tokens.

## Baseline vs tuned metrics

| Metric | Baseline | Tuned |
| --- | ---: | ---: |
| Retrieval Hit@1 | 9/10 (90%) | 9/10 (90%) |
| Retrieval Hit@3 | 10/10 (100%) | 10/10 (100%) |
| Exact unsupported fallback | 4/5 (80%) | 4/5 (80%) |
| Average latency | 10.641 s | 10.627 s |
| Median latency | 9.257 s | 10.507 s |
| Minimum latency | 7.930 s | 9.324 s |
| Maximum latency | 17.156 s | 14.408 s |

Latency was measured after model initialization. This round did not attempt to optimize performance, so the observed timing differences are descriptive only.

## Named weak-case comparison

### `prompt_01` - improved

Retrieved Top-3 in both runs:

1. `prompt_engineering.md`, chunk 2, similarity 0.818826
2. `rag.md`, chunk 2, similarity 0.761284
3. `rag.md`, chunk 1, similarity 0.669683

Baseline answer:

> The information is not available in the supplied context.

Tuned answer:

> The assistant should state that the information is not available in the supplied context.

The tuned answer uses the retrieved prompt-engineering rule instead of falsely treating the question itself as unsupported.

### `architecture_01` - improved

Retrieved Top-3 in both runs:

1. `architecture.md`, chunk 1, similarity 0.661299
2. `foundry_local.md`, chunk 2, similarity 0.646863
3. `foundry_local.md`, chunk 0, similarity 0.581816

Baseline answer:

> The context provided does not contain specific details on how a user's question travels through the local application before an answer is returned. The information is not available in the supplied context.

Tuned answer:

> The user's question is first accepted by the client interface. Then, the server or pipeline layer orchestrates the retrieval and generation processes. Finally, an answer is returned to the user.

The false refusal was removed, and the tuned answer uses the available architecture context.

### `unknown_02` - unchanged

Retrieved Top-3 in both runs:

1. `foundry_local.md`, chunk 0, similarity 0.645564
2. `foundry_local.md`, chunk 2, similarity 0.643927
3. `rag.md`, chunk 2, similarity 0.551299

Baseline answer:

> The information provided does not contain details about the company that owns the local assistant application.

Tuned answer:

> The provided context does not contain information about the company that owns the local assistant application. The information is not available in the supplied context.

The tuned answer includes the required sentence but still adds text before it, so the exact-fallback check remains a failure.

### `rag_02` - unchanged

Retrieved Top-3 in both runs:

1. `rag.md`, chunk 2, similarity 0.749524
2. `rag.md`, chunk 0, similarity 0.639584
3. `embeddings.md`, chunk 2, similarity 0.632405

Baseline answer:

> The project's RAG approach reduces hallucinations and produces answers grounded in the supplied documents by using the same embedding model for both stored document chunks and incoming queries. This ensures that the information provided is based on actual content from the documents rather than generating unsupported answers. The context does not provide further details on how exactly this mechanism reduces hallucinations.

Tuned answer:

> The project's RAG approach uses embeddings as the retrieval representation, which helps in grounding answers in supplied documents, thereby reducing the likelihood of unsupported answers.

The wording is shorter, but it still invents a causal relationship between the embedding fact and hallucination reduction. The grounding problem is unresolved.

### `embeddings_02` - partially improved but unresolved

Retrieved Top-3 in both runs:

1. `embeddings.md`, chunk 2, similarity 0.718440
2. `architecture.md`, chunk 2, similarity 0.690268
3. `sqlite.md`, chunk 2, similarity 0.676696

Baseline answer:

> The context suggests that the same embedding model is used for both stored document chunks and incoming queries to facilitate the retrieval process in the RAG system. This uniformity likely aids in consistency and efficiency during the retrieval and processing stages.
>
> Reference: [Source: embeddings.md | chunk: 2]
> [Source: architecture.md | chunk: 2]
> [Source: sqlite.md | chunk: 2]

Tuned answer:

> The context suggests that the same embedding model is used for both stored document chunks and incoming queries to maintain consistency in the retrieval representation for RAG. This ensures that the local LLM receives relevant chunks as augmented context during a query. The information is derived from the sources: embeddings.md, architecture.md, and sqlite.md.

The formatted reference block disappeared, but the model still generated filenames and unsupported explanatory links. The issue is only partially improved.

## Additional human-review observations

- `embeddings_01` regressed from a mostly responsive answer with a filename typo to the incomplete fragment: “Semantically similar text having similar vector representations.” Retrieval remained Hit@3, so this is a generation-quality regression.
- `foundry_01` improved by removing the baseline text defect `user'thy` while preserving the grounded meaning.
- No new false refusals appeared within the 10 answerable evaluation cases.
- The CLI demo remained functional and passed its required unsupported fallback assertion. Its supported question “What should the assistant do when retrieved information is insufficient?” still elicited the literal fallback rather than a description of the rule, showing that the false-refusal behavior remains sensitive to wording outside the exact evaluation case.

## Regression verification

- Flask API checks: 21/21 passed.
- `/api/health` and `/api/ask` behavior and response shape remained unchanged.
- Valid API responses still contain a non-empty answer and exactly three retrieval-source objects.
- Unsupported API questions remain HTTP 200 responses.
- The existing CLI RAG demo completed successfully.
- Both models unloaded successfully after evaluation, API verification, and CLI verification.

## Outcome

This controlled round is partially successful. It fixed the two measured false refusals (`prompt_01` and `architecture_01`) without changing retrieval, and Hit@3 remained 100%. It did not improve exact fallback accuracy, did not fully prevent unsupported relationships or generated source metadata, and introduced one answer-quality regression. No further prompt iteration was performed in this round.
