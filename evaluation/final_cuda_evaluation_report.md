# Final Production CUDA Evaluation

## Runtime

- Service startup: 14.866 seconds
- Embedding model: `qwen3-embedding-0.6b-cuda-gpu:1`
- Embedding runtime: GPU / `CUDAExecutionProvider`
- Chat model: `Phi-3.5-mini-instruct-cuda-gpu:2`
- Chat runtime: GPU / `CUDAExecutionProvider`
- Cached models were reused: yes

## Metrics

- Hit@1: 9/10
- Hit@3: 10/10
- Exact fallback: 4/5
- Case errors: 0
- Query latency average/median/min/max: 0.453 / 0.341 / 0.250 / 0.982 seconds
- Embedding average: 0.036 seconds
- Retrieval average: 0.003 seconds
- Generation average: 0.413 seconds

## CPU-baseline retrieval comparison

- Same Top-1: 15/15
- Same Top-3 set: 15/15
- Same Top-3 order: 15/15
- Maximum shared-source score difference: 0.001014
- Fallback-behavior changes: none
- Semantic correctness and wording differences require human review.

## Cleanup

- Models unloaded: yes
