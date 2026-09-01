# Final Production CUDA Performance Report

## Final production configuration

Foundry Local is initialized with `Configuration(app_name="foundry_local_rag")`. Startup then attempts to register `CUDAExecutionProvider`, refreshes the catalog, and resolves both models only by alias. No concrete model ID, device constraint, execution-provider constraint, or `select_variant()` call is used.

If CUDA registration fails, startup prints a warning and continues to normal catalog/alias resolution so Foundry Local can use an available CPU variant.

| Role | Alias | Automatically selected ID | Device | Provider | Cached |
| --- | --- | --- | --- | --- | --- |
| Embedding | `qwen3-embedding-0.6b` | `qwen3-embedding-0.6b-cuda-gpu:1` | GPU | `CUDAExecutionProvider` | yes |
| Generation | `phi-3.5-mini` | `Phi-3.5-mini-instruct-cuda-gpu:2` | GPU | `CUDAExecutionProvider` | yes |

## One-time cache migration

Before migration, the production cache contained these selected CPU variants:

- `qwen3-embedding-0.6b-generic-cpu:1` — CPU / `CPUExecutionProvider`, cached, not loaded
- `Phi-3.5-mini-instruct-generic-cpu:2` — CPU / `CPUExecutionProvider`, cached, not loaded

The installed SDK's per-variant `remove_from_cache()` API removed only those two entries. The unrelated cached `qwen2.5-0.5b-instruct-generic-cpu:4` model was preserved. A subsequent clean process registered CUDA, refreshed the catalog, and automatically selected the uncached CUDA variants by alias before downloading them normally.

Cache removal is **not** part of application startup. It exists only in the one-time development migration diagnostic.

## Acquisition and startup costs

These timings are not query latency:

| Operation | Time |
| --- | ---: |
| Isolated first CUDA provider acquisition/registration | 312.877 s |
| Subsequent production CUDA registration | 4.931 s |
| One-time CUDA embedding-model download | 89.570 s |
| One-time CUDA Phi-model download | 392.613 s |
| Normal production service startup with cached models | 14.866 s |

Normal service startup includes provider registration, catalog refresh, both cached model loads, and the SQLite read. It excludes provider-component and model downloads.

## Query performance

The controlled A1/A2/A3 and B1/B2/B3 comparison was:

| Stage | CPU average | CUDA average | Reduction |
| --- | ---: | ---: | ---: |
| Query embedding | 1.185 s | 0.019 s | 98.44% |
| In-memory retrieval | 0.004 s | 0.003 s | — |
| Generation | 9.758 s | 0.562 s | 94.24% |
| Total | 10.947 s | 0.583 s | 94.67% |

The final 15-case production evaluation, using the regenerated CUDA corpus embeddings, measured:

- Average: 0.453 s
- Median: 0.341 s
- Minimum: 0.250 s
- Maximum: 0.982 s
- Average query embedding: 0.036 s
- Average retrieval: 0.003 s
- Average generation: 0.413 s

The first evaluated query included normal runtime warm-up. Live Flask requests after model loading completed in 1.079 s, 0.612 s, and 0.280 s. Headless-browser requests completed in 1.101 s, 0.666 s, and 0.331 s.

## Final evaluation

- Retrieval Hit@1: 9/10 (90%)
- Retrieval Hit@3: 10/10 (100%)
- Exact unsupported fallback: 4/5 (80%)
- Case errors: 0
- Top-1 unchanged from tuned CPU baseline: 15/15
- Top-3 set unchanged: 15/15
- Top-3 order unchanged: 15/15
- Maximum shared-source similarity-score difference: 0.001014
- Fallback-behavior changes: none

Only two generated answers differed from the tuned CPU run: `foundry_01` added the supported word “local,” and `foundry_02` produced a fuller supported answer. Semantic correctness remains human-reviewed.

The CLI demo still exposed one known Phi grounding weakness: for “How does RAG help reduce hallucinations?”, it connected embedding-model consistency to hallucination reduction even though that causal relationship was not explicitly supported. No prompt or model tuning was performed during this production-acceleration change.

## Embedding persistence

The production corpus was re-ingested once through `qwen3-embedding-0.6b-cuda-gpu:1`:

- Source documents: 6
- Stored chunks: 18
- Embedding dimension: 1024
- All values finite: yes
- Duplicate `(source, chunk_index)` keys: 0
- Previous database SHA-256: `2084C097D2B68C95DD44AFD33CC886FD8961F19C0F4E77CC0C141B7262250DFC`
- Final database SHA-256: `874089192771C1FAD7F0031613B8E85635E6C9967E911B12DAEB90E2F1917442`

## Regression results

- API test client: 21/21 checks passed
- `/api/health`: unchanged, HTTP 200 with `{"status":"ok"}`
- `/api/ask`: unchanged, valid answers include exactly three sources
- Unsupported question: unchanged, HTTP 200 with the exact grounded fallback
- CLI RAG demo: completed successfully; both models unloaded
- Real Flask server: page, CSS, JavaScript, health, and ask endpoints returned successfully
- Browser transcript: three independent turns retained; prior turns were not sent to the API
- Source disclosures: collapsed initially and exposed all three source rows when opened
- Theme control: light/dark switch persisted locally without clearing the transcript
- Desktop/mobile horizontal overflow: none detected
- Browser console-breaking errors: none

## Integrity

The grounding prompt, model aliases, Top-K, cosine retrieval, chunking, knowledge-base files, API contract, Flask routes, frontend, and evaluation cases were not changed. No dependency was added. The completed experiment's temporary model cache and the temporary browser profile were removed after their processes exited.
