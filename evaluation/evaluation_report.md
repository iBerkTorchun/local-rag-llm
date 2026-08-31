# Baseline RAG Evaluation Report

This report measures source-file retrieval and exact unsupported-question fallback behavior. **Semantic answer correctness requires human review**; expected concepts are guidance and are not automatically scored.

## Test configuration

- Embedding model alias: `qwen3-embedding-0.6b`
- Selected embedding model ID: `qwen3-embedding-0.6b-generic-cpu:1`
- Chat model alias: `phi-3.5-mini`
- Selected chat model ID: `Phi-3.5-mini-instruct-generic-cpu:2`
- Knowledge-base chunks: 18
- Top-K: 3
- Answerable cases: 10
- Unanswerable cases: 5
- Query latency timing begins after model initialization.

## Retrieval metrics

- Hit@1: 9/10 (90.00%)
- Hit@3: 10/10 (100.00%)

## Grounding metrics

- Unsupported-question fallback success: 4/5 (80.00%)

## Performance

- Queries: 15
- Average latency: 10.641 seconds
- Median latency: 9.257 seconds
- Minimum latency: 7.930 seconds
- Maximum latency: 17.156 seconds

## Case-by-case results

### rag_01 - answerable

**Question:** What sequence does the system follow to turn relevant documents into a model response?

**Latency:** 11.818 seconds

**Expected source(s):** `rag.md`

**Expected concepts (human-review guidance):**

- Retrieve relevant information from the document collection
- Add the retrieved information to the model input as context
- Generate an answer using that context

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `rag.md` - chunk 1 - similarity 0.710427
2. `rag.md` - chunk 2 - similarity 0.669858
3. `prompt_engineering.md` - chunk 1 - similarity 0.632272

**Generated answer:**

> The system follows this sequence: Retrieve relevant information from documents, add it to the model input as context, and then generate the answer using that context. Additionally, it is instructed not to invent unsupported information.

---

### rag_02 - answerable

**Question:** How does the project's RAG approach make unsupported answers less likely?

**Latency:** 15.545 seconds

**Expected source(s):** `rag.md`, `prompt_engineering.md`

**Expected concepts (human-review guidance):**

- RAG helps reduce hallucinations
- Answers are grounded in supplied documents or retrieved context

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `rag.md` - chunk 2 - similarity 0.749524
2. `rag.md` - chunk 0 - similarity 0.639584
3. `embeddings.md` - chunk 2 - similarity 0.632405

**Generated answer:**

> The project's RAG approach reduces hallucinations and produces answers grounded in the supplied documents by using the same embedding model for both stored document chunks and incoming queries. This ensures that the information provided is based on actual content from the documents rather than generating unsupported answers. The context does not provide further details on how exactly this mechanism reduces hallucinations.

---

### foundry_01 - answerable

**Question:** Why can this assistant run model inference without sending application data to a cloud model?

**Latency:** 12.106 seconds

**Expected source(s):** `foundry_local.md`

**Expected concepts (human-review guidance):**

- Foundry Local runs language models on the user's device
- Application data does not need to be sent to a cloud model
- The approach is suitable for an offline local Q&A assistant

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `foundry_local.md` - chunk 2 - similarity 0.828089
2. `foundry_local.md` - chunk 0 - similarity 0.714499
3. `foundry_local.md` - chunk 1 - similarity 0.647443

**Generated answer:**

> Foundry Local enables large language models to run locally on a user'thy device, which allows applications to perform local model inference without sending application data to a cloud model.

---

### foundry_02 - answerable

**Question:** Which local component offers both a model runtime and a catalog of optimized models?

**Latency:** 7.930 seconds

**Expected source(s):** `foundry_local.md`

**Expected concepts (human-review guidance):**

- Foundry Local
- A lightweight runtime and SDK
- A curated catalog of optimized models

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `foundry_local.md` - chunk 1 - similarity 0.807529
2. `foundry_local.md` - chunk 2 - similarity 0.729952
3. `foundry_local.md` - chunk 0 - similarity 0.711007

**Generated answer:**

> Foundry Local

---

### embeddings_01 - answerable

**Question:** What representation allows the system to compare a question with passages by meaning?

**Latency:** 11.484 seconds

**Expected source(s):** `embeddings.md`

**Expected concepts (human-review guidance):**

- Text embeddings are numerical vector representations of meaning
- Embeddings enable semantic similarity search

**Automated retrieval:** Hit@1 = FAIL; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `rag.md` - chunk 2 - similarity 0.578521
2. `prompt_engineering.md` - chunk 0 - similarity 0.513356
3. `embeddings.md` - chunk 1 - similarity 0.502340

**Generated answer:**

> Semantically similar text should have similar vector representations, as stated in the context from embeddthy.md (chunk: 1). This representation allows the system to compare a question with passages by meaning.

---

### embeddings_02 - answerable

**Question:** Why should stored passages and new search questions be encoded with one embedding model?

**Latency:** 17.156 seconds

**Expected source(s):** `embeddings.md`

**Expected concepts (human-review guidance):**

- The same embedding model must be used for stored document chunks and incoming queries

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `embeddings.md` - chunk 2 - similarity 0.718440
2. `architecture.md` - chunk 2 - similarity 0.690268
3. `sqlite.md` - chunk 2 - similarity 0.676696

**Generated answer:**

> The context suggests that the same embedding model is used for both stored document chunks and incoming queries to facilitate the retrieval process in the RAG system. This uniformity likely aids in consistency and efficiency during the retrieval and processing stages.
>
> Reference: [Source: embeddings.md | chunk: 2]
> [Source: architecture.md | chunk: 2]
> [Source: sqlite.md | chunk: 2]

---

### sqlite_01 - answerable

**Question:** What makes SQLite practical for this local project without operating a database server?

**Latency:** 10.316 seconds

**Expected source(s):** `sqlite.md`

**Expected concepts (human-review guidance):**

- SQLite is lightweight, serverless, and self-contained
- The database can be stored in one local file
- No separate database server is required

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `sqlite.md` - chunk 1 - similarity 0.829211
2. `sqlite.md` - chunk 0 - similarity 0.813758
3. `sqlite.md` - chunk 2 - similarity 0.654055

**Generated answer:**

> SQLite is lightweight, serverless, and self-contained, which allows it to be stored as a single local file without the need for a separate database server.

---

### sqlite_02 - answerable

**Question:** Which retrieval data does the application keep in its SQLite file?

**Latency:** 8.618 seconds

**Expected source(s):** `sqlite.md`, `architecture.md`

**Expected concepts (human-review guidance):**

- Document chunks
- Embedding vectors

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `sqlite.md` - chunk 2 - similarity 0.711173
2. `sqlite.md` - chunk 1 - similarity 0.658238
3. `sqlite.md` - chunk 0 - similarity 0.642960

**Generated answer:**

> The application keeps document chunks and their embedding vectors in its SQLite file.

---

### prompt_01 - answerable

**Question:** How should the assistant respond when its retrieved material cannot support an answer?

**Latency:** 8.988 seconds

**Expected source(s):** `prompt_engineering.md`

**Expected concepts (human-review guidance):**

- Do not invent unsupported information
- Say that the information is not available when context is insufficient

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `prompt_engineering.md` - chunk 2 - similarity 0.818826
2. `rag.md` - chunk 2 - similarity 0.761284
3. `rag.md` - chunk 1 - similarity 0.669683

**Generated answer:**

> The information is not available in the supplied context.

---

### architecture_01 - answerable

**Question:** How does a user's question travel through the local application before an answer is returned?

**Latency:** 11.602 seconds

**Expected source(s):** `architecture.md`

**Expected concepts (human-review guidance):**

- A client accepts the question
- A server or pipeline coordinates retrieval and generation
- SQLite stores chunks and embeddings
- Foundry Local provides local inference
- Retrieved chunks are supplied to the local LLM as augmented context

**Automated retrieval:** Hit@1 = PASS; Hit@3 = PASS

**Semantic answer correctness:** Human review required.

**Retrieved sources:**

1. `architecture.md` - chunk 1 - similarity 0.661299
2. `foundry_local.md` - chunk 2 - similarity 0.646863
3. `foundry_local.md` - chunk 0 - similarity 0.581816

**Generated answer:**

> The context provided does not contain specific details on how a user's question travels through the local application before an answer is returned. The information is not available in the supplied context.

---

### unknown_01 - unanswerable

**Question:** Who wrote the documents in this knowledge base?

**Latency:** 8.800 seconds

**Automated fallback check:** PASS

**Retrieved sources:**

1. `rag.md` - chunk 2 - similarity 0.598415
2. `prompt_engineering.md` - chunk 1 - similarity 0.506288
3. `rag.md` - chunk 1 - similarity 0.495984

**Generated answer:**

> The information is not available in the supplied context.

---

### unknown_02 - unanswerable

**Question:** Which company owns this local assistant application?

**Latency:** 9.257 seconds

**Automated fallback check:** FAIL

**Retrieved sources:**

1. `foundry_local.md` - chunk 0 - similarity 0.645564
2. `foundry_local.md` - chunk 2 - similarity 0.643927
3. `rag.md` - chunk 2 - similarity 0.551299

**Generated answer:**

> The information provided does not contain details about the company that owns the local assistant application.

---

### unknown_03 - unanswerable

**Question:** How many vacation days do users of this application receive?

**Latency:** 8.471 seconds

**Automated fallback check:** PASS

**Retrieved sources:**

1. `foundry_local.md` - chunk 0 - similarity 0.540307
2. `rag.md` - chunk 2 - similarity 0.532965
3. `foundry_local.md` - chunk 2 - similarity 0.522682

**Generated answer:**

> The information is not available in the supplied context.

---

### unknown_04 - unanswerable

**Question:** On what date is this application scheduled to be deployed?

**Latency:** 8.719 seconds

**Automated fallback check:** PASS

**Retrieved sources:**

1. `foundry_local.md` - chunk 0 - similarity 0.523764
2. `foundry_local.md` - chunk 2 - similarity 0.519655
3. `prompt_engineering.md` - chunk 1 - similarity 0.474028

**Generated answer:**

> The information is not available in the supplied context.

---

### unknown_05 - unanswerable

**Question:** How much RAM and which processor are required to run the assistant?

**Latency:** 8.811 seconds

**Automated fallback check:** PASS

**Retrieved sources:**

1. `prompt_engineering.md` - chunk 2 - similarity 0.498514
2. `foundry_local.md` - chunk 2 - similarity 0.485945
3. `rag.md` - chunk 2 - similarity 0.483565

**Generated answer:**

> The information is not available in the supplied context.

---

## Cleanup

- Models unloaded successfully: yes
