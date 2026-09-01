# Empty-Cache CUDA Alias-Selection Experiment

## Configuration

- SDK: `foundry-local-sdk-winml==1.2.4`
- Isolated model cache: `C:\Users\ayber\AppData\Local\Temp\foundry-local-empty-cache-51ft7x4r\models`
- Model cache was empty before Foundry Local initialization: yes
- Registered provider: `CUDAExecutionProvider` only
- Concrete model/device/provider selection was not forced

## Registration

- Success: `true`
- CUDA registered in the same process: `true`
- Status: EP registration complete: 1 bootstrapper(s) succeeded in 312877ms. Available EPs: CPUExecutionProvider, CUDAExecutionProvider

## Fresh-cache automatic selection

| Alias | Selected concrete ID | Device | Execution provider | Cached before download |
| --- | --- | --- | --- | --- |
| `qwen3-embedding-0.6b` | `qwen3-embedding-0.6b-cuda-gpu:1` | GPU | `CUDAExecutionProvider` | no |
| `phi-3.5-mini` | `Phi-3.5-mini-instruct-cuda-gpu:2` | GPU | `CUDAExecutionProvider` | no |

**Outcome:** Empty cache caused alias-based CUDA selection; cached CPU variants were influencing the previous selection.

## Performance

| Stage | CPU average | CUDA average | Reduction |
| --- | ---: | ---: | ---: |
| Query embedding | 1.185s | 0.019s | 98.44% |
| Retrieval | 0.004s | 0.003s | not a tuning target |
| Generation | 9.758s | 0.562s | 94.24% |
| Total | 10.947s | 0.583s | 94.67% |

## Retrieval consistency and grounding

- Embedding dimension: 1024
- Same Top-1: 15/15
- Same Top-3 set: 15/15
- Same Top-3 order: 15/15
- Material ranking difference: no
- Hit@1: 9/10
- Hit@3: 10/10
- Exact fallback: 4/5
- Semantic answer and grounding quality require human review.

## Integrity

- Production files unchanged: yes
- Production model-cache metadata unchanged: yes
- Temporary cache removed: no
- Experiment models unloaded: yes
