# Controlled Chat-Model Quality and Latency Comparison

This experiment compares the production Phi-3.5 Mini model with Qwen2.5-0.5B without changing the production configuration. Semantic answer quality requires human review and is not automatically scored.

## Test configuration

- Embedding model: `qwen3-embedding-0.6b` (`qwen3-embedding-0.6b-generic-cpu:1`)
- Production chat model: `phi-3.5-mini` (`Phi-3.5-mini-instruct-generic-cpu:2`)
- Comparison chat model: `qwen2.5-0.5b` (`qwen2.5-0.5b-instruct-generic-cpu:4`)
- Top-K: 3
- Cases: 15
- Grounding prompt SHA-256: `cda38567f5cb9fbae5ea95349fb24a00a53edf201e1d8cfcb388f1ccbd93116b`
- Generation settings applied to both clients: temperature 0.0, max tokens 96, random seed 0
- Comparable query latency is shared retrieval latency plus chat generation latency; model download and load time are excluded.

## Retrieval identity

Every question was retrieved exactly once through the existing `RAGService.get_top_chunks()` path. The resulting context was reused unchanged for both chat models.

- Hit@1: 9/10 (90.00%)
- Hit@3: 10/10 (100.00%)

## Model comparison

| Metric | Phi-3.5 Mini | Qwen2.5-0.5B |
| --- | ---: | ---: |
| Exact fallback | 4/5 (80.00%) | 0/5 (0.00%) |
| Average query latency | 9.716 s | 2.068 s |
| Median query latency | 9.358 s | 1.987 s |
| Minimum query latency | 8.256 s | 1.763 s |
| Maximum query latency | 12.594 s | 2.861 s |

## Performance interpretation

Qwen's average comparable query latency was 4.70x faster (78.7% lower) than Phi's. Its average chat generation time alone was 6.11x faster. These figures establish a latency difference, not a model-quality verdict.

## Human review

This review was performed directly on the paired outputs below. It is qualitative: no automatic semantic-answer correctness score or LLM judge was used.

### Overall observations

- **Phi-3.5 Mini:** It answered most supported questions concisely and used the exact fallback for four of five unsupported questions. Its remaining defects were the non-exact fallback in `unknown_02`, an unsupported causal connection in `rag_02`, an unnecessary generated source sentence in `embeddings_02`, and an awkward/incomplete formulation in `embeddings_01`. No false refusals occurred in `prompt_01` or `architecture_01` in this run.
- **Qwen2.5-0.5B:** It was substantially faster, but it produced no exact fallback in the five unsupported cases. It invented an author class, an owner, a vacation allowance, a deployment placeholder, and hardware requirements. Several supported answers were also wrong, incomplete, unsupported, or malformed: notably `rag_02`, `foundry_02`, `sqlite_02`, `prompt_01`, `architecture_01`, and `embeddings_02`.
- Neither model produced an execution error. The defects above are answer-quality and grounding defects, not transport or inference failures.

| Defect category | Phi-3.5 Mini | Qwen2.5-0.5B |
| --- | --- | --- |
| False refusal | None observed among the 10 answerable cases. | None in the exact-fallback form, but `prompt_01` fails to answer the supported question. |
| Unsupported factual/causal claims | Unsupported embeddings-to-grounding relationship in `rag_02`; unsupported consequence in `embeddings_02`. | Multiple unsupported claims, including all five unsupported cases, plus `rag_02`, `embeddings_02`, and `sqlite_02`. |
| Exact fallback failure | `unknown_02` adds a preface before the fallback (1/5 failed). | All five unsupported cases fail the exact fallback (5/5 failed). |
| Generated source/reference block | `embeddings_02` appends a source list despite the prompt. | No explicit source/reference block observed. |
| Incomplete or malformed text | `embeddings_01` is awkward and incomplete as an answer. | `embeddings_02` is cut off mid-sentence; several other answers are vague or malformed. |

### Previously difficult cases

Both models received the one shared Top-3 list shown for each case; there was no model-specific retrieval.

| Case | Shared Top-3 sources | Phi-3.5 Mini review | Qwen2.5-0.5B review |
| --- | --- | --- | --- |
| `prompt_01` | `prompt_engineering.md`, `rag.md`, `rag.md` | Correctly says the assistant should report that the information is unavailable; no false refusal. | Does not answer the question and substitutes the Retrieve/Augment/Generate sequence. |
| `architecture_01` | `architecture.md`, `foundry_local.md`, `foundry_local.md` | Correctly describes client input followed by server/pipeline retrieval and generation; no false refusal. | Vague and incomplete: it omits the client step and the retrieval/generation sequence. |
| `unknown_02` | `foundry_local.md`, `foundry_local.md`, `rag.md` | Correctly recognizes that ownership is unavailable, but adds an explanatory sentence before the required fallback, so exact fallback fails. | Hallucinates that Foundry Local owns the application. |
| `rag_02` | `rag.md`, `rag.md`, `embeddings.md` | Connects the use of embeddings causally to grounded answers even though that relationship is not supported by the retrieved text. | Makes the unsupported and nonsensical claim that embeddings allow non-existent documents to be generated from stored documents. |
| `embeddings_01` | `rag.md`, `prompt_engineering.md`, `embeddings.md` | Points toward similar vector representations, but the wording is incomplete and does not explicitly identify embeddings. | Correctly and directly identifies embeddings. |
| `embeddings_02` | `embeddings.md`, `architecture.md`, `sqlite.md` | Restates model consistency, then adds an unsupported consequence and an unnecessary generated source list. | Adds broad unsupported guarantees and retrieval claims, then ends mid-sentence at the shared token limit. |

### Other notable defects

- Qwen's `sqlite_02` answer says the SQLite file's content is kept in memory, contradicting the retrieved context about persisted chunks and embedding vectors.
- Qwen's `foundry_02` selects a runtime component instead of identifying Foundry Local as the component that provides both the runtime/SDK and model catalog.
- All five Qwen unsupported answers introduce unsupported content rather than using the mandated fallback. `unknown_04` even emits an unresolved `[insert date]` placeholder.

### Quality/latency trade-off

Qwen is meaningfully faster on this hardware and workload, but the speed advantage comes with a substantial grounding and answer-quality loss in this run. Phi is slower yet materially more reliable: it preserves 80% exact fallback behavior and gives usable answers on more supported cases. The experiment therefore does not justify automatically preferring the smaller model, and it does not change the production selection.

## Case-by-case outputs

### `rag_01` - answerable

**Question:** What sequence does the system follow to turn relevant documents into a model response?

**Shared retrieval latency:** 0.668 seconds

**Shared Top-3:**

1. `rag.md` - chunk 1 - similarity 0.710427
2. `rag.md` - chunk 2 - similarity 0.669858
3. `prompt_engineering.md` - chunk 1 - similarity 0.632272

**Phi-3.5 Mini answer** (generation 10.874 s; query 11.542 s):

> Retrieve → Augment → Generate. Relevant information is first retrieved from documents, added to the model input as context, and then the model generates its answer using that context.

**Qwen2.5-0.5B answer** (generation 2.193 s; query 2.861 s):

> The system follows these steps to turn relevant documents into a model response:
>
> 1. Retrieve - First, the system retrieves relevant documents from a document collection.
> 2. Augment - Next, the system adds the retrieved documents to the model input as context.
> 3. Generate - Then, the system uses the augmented context to generate its own answer.

---

### `rag_02` - answerable

**Question:** How does the project's RAG approach make unsupported answers less likely?

**Shared retrieval latency:** 0.559 seconds

**Shared Top-3:**

1. `rag.md` - chunk 2 - similarity 0.749524
2. `rag.md` - chunk 0 - similarity 0.639584
3. `embeddings.md` - chunk 2 - similarity 0.632405

**Phi-3.5 Mini answer** (generation 10.130 s; query 10.689 s):

> The project's RAG approach uses embeddings as the retrieval representation, which helps in grounding answers in supplied documents, thereby reducing the likelihood of unsupported answers.

**Qwen2.5-0.5B answer** (generation 1.432 s; query 1.991 s):

> The project's RAG approach uses embeddings as the retrieval representation to ensure that even non-existent documents can be effectively generated based on stored documents.

---

### `foundry_01` - answerable

**Question:** Why can this assistant run model inference without sending application data to a cloud model?

**Shared retrieval latency:** 0.593 seconds

**Shared Top-3:**

1. `foundry_local.md` - chunk 2 - similarity 0.828089
2. `foundry_local.md` - chunk 0 - similarity 0.714499
3. `foundry_local.md` - chunk 1 - similarity 0.647443

**Phi-3.5 Mini answer** (generation 10.093 s; query 10.686 s):

> Foundry Local enables large language models to run locally on a user's device, which allows the assistant to perform model inference without sending application data to a cloud model.

**Qwen2.5-0.5B answer** (generation 1.545 s; query 2.138 s):

> Foundry Local uses local model inference directly within its own infrastructure, eliminating the need for external cloud services.

---

### `foundry_02` - answerable

**Question:** Which local component offers both a model runtime and a catalog of optimized models?

**Shared retrieval latency:** 0.645 seconds

**Shared Top-3:**

1. `foundry_local.md` - chunk 1 - similarity 0.807529
2. `foundry_local.md` - chunk 2 - similarity 0.729952
3. `foundry_local.md` - chunk 0 - similarity 0.711007

**Phi-3.5 Mini answer** (generation 7.927 s; query 8.572 s):

> Foundry Local

**Qwen2.5-0.5B answer** (generation 1.405 s; query 2.050 s):

> The local component offered by Foundry Local is the **runtime** component, which includes a lightweight runtime and SDK.

---

### `embeddings_01` - answerable

**Question:** What representation allows the system to compare a question with passages by meaning?

**Shared retrieval latency:** 0.585 seconds

**Shared Top-3:**

1. `rag.md` - chunk 2 - similarity 0.578521
2. `prompt_engineering.md` - chunk 0 - similarity 0.513356
3. `embeddings.md` - chunk 1 - similarity 0.502340

**Phi-3.5 Mini answer** (generation 7.813 s; query 8.399 s):

> Semantically similar text having similar vector representations.

**Qwen2.5-0.5B answer** (generation 1.177 s; query 1.763 s):

> The system uses embeddings to compare questions with passages by meaning.

---

### `embeddings_02` - answerable

**Question:** Why should stored passages and new search questions be encoded with one embedding model?

**Shared retrieval latency:** 0.545 seconds

**Shared Top-3:**

1. `embeddings.md` - chunk 2 - similarity 0.718440
2. `architecture.md` - chunk 2 - similarity 0.690268
3. `sqlite.md` - chunk 2 - similarity 0.676696

**Phi-3.5 Mini answer** (generation 12.049 s; query 12.594 s):

> The context suggests that the same embedding model is used for both stored document chunks and incoming queries to maintain consistency in the retrieval representation for RAG. This ensures that the local LLM receives relevant chunks as augmented context during a query. The information is derived from the sources: embeddings.md, architecture.md, and sqlite.md.

**Qwen2.5-0.5B answer** (generation 2.304 s; query 2.849 s):

> To ensure accurate retrieval of stored passages and new search questions using one embedding model, it's important to maintain consistency across both storage methods (document chunks and embeddings) and query processing stages (local document chunks and local LLM). This ensures that the correct content from each source can be accessed efficiently during the query phase. By using a single embedding model, we guarantee that all relevant data points are effectively captured and utilized by the system. This approach minimizes potential errors due to differences in

---

### `sqlite_01` - answerable

**Question:** What makes SQLite practical for this local project without operating a database server?

**Shared retrieval latency:** 0.553 seconds

**Shared Top-3:**

1. `sqlite.md` - chunk 1 - similarity 0.829211
2. `sqlite.md` - chunk 0 - similarity 0.813758
3. `sqlite.md` - chunk 2 - similarity 0.654055

**Phi-3.5 Mini answer** (generation 8.887 s; query 9.440 s):

> SQLite is lightweight, serverless, and self-contained, allowing it to be stored as a single local file without the need for a separate database server.

**Qwen2.5-0.5B answer** (generation 1.458 s; query 2.011 s):

> SQLite's lightweight nature allows it to store documents locally efficiently. Additionally, its suitability as a standalone storage solution fits perfectly within this project's requirements.

---

### `sqlite_02` - answerable

**Question:** Which retrieval data does the application keep in its SQLite file?

**Shared retrieval latency:** 0.550 seconds

**Shared Top-3:**

1. `sqlite.md` - chunk 2 - similarity 0.711173
2. `sqlite.md` - chunk 1 - similarity 0.658238
3. `sqlite.md` - chunk 0 - similarity 0.642960

**Phi-3.5 Mini answer** (generation 7.851 s; query 8.401 s):

> The application keeps document chunks and their embedding vectors in its SQLite file.

**Qwen2.5-0.5B answer** (generation 1.222 s; query 1.771 s):

> The application keeps the SQLite file's content in memory.

---

### `prompt_01` - answerable

**Question:** How should the assistant respond when its retrieved material cannot support an answer?

**Shared retrieval latency:** 0.535 seconds

**Shared Top-3:**

1. `prompt_engineering.md` - chunk 2 - similarity 0.818826
2. `rag.md` - chunk 2 - similarity 0.761284
3. `rag.md` - chunk 1 - similarity 0.669683

**Phi-3.5 Mini answer** (generation 8.503 s; query 9.038 s):

> The assistant should state that the information is not available in the supplied context.

**Qwen2.5-0.5B answer** (generation 1.825 s; query 2.360 s):

> If the assistant's retrieved material cannot support an answer, it should follow the principle of "retrieve → augment → generate," which means first retrieving relevant information from a document collection and then adding it to the model input as context before generating the final answer.

---

### `architecture_01` - answerable

**Question:** How does a user's question travel through the local application before an answer is returned?

**Shared retrieval latency:** 0.545 seconds

**Shared Top-3:**

1. `architecture.md` - chunk 1 - similarity 0.661299
2. `foundry_local.md` - chunk 2 - similarity 0.646863
3. `foundry_local.md` - chunk 0 - similarity 0.581816

**Phi-3.5 Mini answer** (generation 11.819 s; query 12.364 s):

> The user's question is first accepted by the client interface. Then, the server or pipeline layer orchestrates the retrieval and generation processes. Finally, an answer is returned to the user.

**Qwen2.5-0.5B answer** (generation 1.405 s; query 1.950 s):

> A user's question travels through the local application first before being answered by a server or pipeline layer.

---

### `unknown_01` - unanswerable

**Question:** Who wrote the documents in this knowledge base?

**Shared retrieval latency:** 0.515 seconds

**Shared Top-3:**

1. `rag.md` - chunk 2 - similarity 0.598415
2. `prompt_engineering.md` - chunk 1 - similarity 0.506288
3. `rag.md` - chunk 1 - similarity 0.495984

**Phi-3.5 Mini answer** (generation 8.843 s; query 9.358 s):

> The information is not available in the supplied context.

Exact fallback: PASS

**Qwen2.5-0.5B answer** (generation 1.269 s; query 1.784 s):

> The documents were written by humans.

Exact fallback: FAIL

---

### `unknown_02` - unanswerable

**Question:** Which company owns this local assistant application?

**Shared retrieval latency:** 0.516 seconds

**Shared Top-3:**

1. `foundry_local.md` - chunk 0 - similarity 0.645564
2. `foundry_local.md` - chunk 2 - similarity 0.643927
3. `rag.md` - chunk 2 - similarity 0.551299

**Phi-3.5 Mini answer** (generation 8.894 s; query 9.410 s):

> The provided context does not contain information about the company that owns the local assistant application. The information is not available in the supplied context.

Exact fallback: FAIL

**Qwen2.5-0.5B answer** (generation 1.261 s; query 1.778 s):

> The company that owns this local assistant application is Foundry Local.

Exact fallback: FAIL

---

### `unknown_03` - unanswerable

**Question:** How many vacation days do users of this application receive?

**Shared retrieval latency:** 0.598 seconds

**Shared Top-3:**

1. `foundry_local.md` - chunk 0 - similarity 0.540307
2. `rag.md` - chunk 2 - similarity 0.532965
3. `foundry_local.md` - chunk 2 - similarity 0.522682

**Phi-3.5 Mini answer** (generation 7.657 s; query 8.256 s):

> The information is not available in the supplied context.

Exact fallback: PASS

**Qwen2.5-0.5B answer** (generation 1.293 s; query 1.891 s):

> The application receives **1** vacation day by default.

Exact fallback: FAIL

---

### `unknown_04` - unanswerable

**Question:** On what date is this application scheduled to be deployed?

**Shared retrieval latency:** 0.545 seconds

**Shared Top-3:**

1. `foundry_local.md` - chunk 0 - similarity 0.523764
2. `foundry_local.md` - chunk 2 - similarity 0.519655
3. `prompt_engineering.md` - chunk 1 - similarity 0.474028

**Phi-3.5 Mini answer** (generation 7.807 s; query 8.351 s):

> The information is not available in the supplied context.

Exact fallback: PASS

**Qwen2.5-0.5B answer** (generation 1.292 s; query 1.836 s):

> This application was scheduled to be deployed on [insert date].

Exact fallback: FAIL

---

### `unknown_05` - unanswerable

**Question:** How much RAM and which processor are required to run the assistant?

**Shared retrieval latency:** 0.599 seconds

**Shared Top-3:**

1. `prompt_engineering.md` - chunk 2 - similarity 0.498514
2. `foundry_local.md` - chunk 2 - similarity 0.485945
3. `rag.md` - chunk 2 - similarity 0.483565

**Phi-3.5 Mini answer** (generation 8.047 s; query 8.646 s):

> The information is not available in the supplied context.

Exact fallback: PASS

**Qwen2.5-0.5B answer** (generation 1.388 s; query 1.987 s):

> The assistant requires 4GB of RAM and a CPU with at least 16-bit support.

Exact fallback: FAIL

---

## Cleanup

- Phi and embedding models unloaded: yes
- Qwen model unloaded: yes
