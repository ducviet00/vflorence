# fast-encdec

A small, readable CPU inference engine for **BART** and **Florence-2**, in the spirit of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Key Features

* **Fast offline CPU inference** - paged KV cache, continuous batching, greedy and beam search
* **Readable codebase** - minimal PyTorch + [vllm-cpu-attention](https://github.com/ducviet00/vllm-cpu-attention) kernels
* **Two models, one engine** - BART for summarisation, Florence-2 for vision-language tasks
* **FP32 and BF16** - frozen `torch.compile` weight packing (oneDNN / ACL) + SIMD/AMX paged attention

## Installation

```bash
uv pip install torch torchvision setuptools --index-url https://download.pytorch.org/whl/cpu
uv pip install --no-build-isolation git+https://github.com/ducviet00/vllm-cpu-attention.git
uv pip install --no-build-isolation -e .
```

## Quick Start

See `examples/` for runnable scripts. `LLM` takes `dtype` (`torch.float32` or `torch.bfloat16`),
`num_blocks`, `block_size`, `max_num_seqs`, `compile` (default `True`) and `compile_mm_encoder`:

```python
from fastencdec import LLM, SamplingParams

llm = LLM("facebook/bart-large-cnn")
outputs = llm.generate(
    "A car is parked in front of a building ...",
    SamplingParams(max_tokens=64, num_beams=4),
)
print(outputs[0].outputs[0].text)
```

Florence-2 takes an image plus a task prompt, and the answer is parsed with the
processor:

```python
import torch
from PIL import Image

from fastencdec import LLM, SamplingParams

llm = LLM("florence-community/Florence-2-base", dtype=torch.bfloat16)
image = Image.open("car.jpg")
outputs = llm.generate(
    {"prompt": "<OD>", "multi_modal_data": {"image": image}},
    SamplingParams(max_tokens=256, num_beams=3),
)
parsed = llm.processor.post_process_generation(
    outputs[0].outputs[0].text, task="<OD>", image_size=image.size
)
```

## How It Works

Reading order: `engine/sequence.py` -> `engine/block_manager.py` ->
`engine/scheduler.py` -> `engine/model_runner.py` -> `engine/llm_engine.py` ->
`layers/attention.py` -> `models/bart.py`. The step-by-step flow is in
[AGENTS.md section 4](AGENTS.md#4-how-a-step-works).

* **Paged KV cache.** Keys and values live in fixed blocks; beams share blocks and
  only copy one when they write into a shared block.
* **Continuous batching.** Requests join and leave the running batch freely.
* **No padding.** One token per sequence per step, and attention reads exactly the
  cached length of each sequence, so mixed lengths waste nothing.
* **Beam search** runs on the same cache, ranked by length-penalised score.
* **CPU attention.** Decoder self- and cross-attention call `vllm-cpu-attention`
  (`vllm_cpu_attn`) through opaque `torch.library.custom_op`s so the decoder,
  LM head, and encoder compile cleanly with Inductor weight freezing.

## Benchmark

Measured on GCP `c4a-standard-8` / `c4a-standard-16` (Arm Neoverse-V2, 8 / 16 cores) and
`c4-standard-8` / `c4-standard-16` (Intel Xeon 8581C, 4 / 8 cores, AMX) with
`torch 2.15.0+cpu`, `transformers 5.22.1`, and `vllm-cpu-attn 0.1.0`
(`benchmarks/benchmark.py` and `benchmarks/benchmark_florence2.py`):

| point | config | c4a-standard-8 | c4-standard-8 | c4a-standard-16 | c4-standard-16 |
|---|---|---:|---:|---:|---:|
| 128->512 | batch=1, beams=1, bf16 | 1.41x (127.2 / 90.4 tok/s) | 1.19x (73.4 / 61.7 tok/s) | 1.45x (166.5 / 114.8 tok/s) | 1.29x (118.6 / 91.9 tok/s) |
| 128->512 | batch=8, beams=1, bf16 | 2.02x (545.7 / 270.0 tok/s) | 1.64x (422.8 / 258.3 tok/s) | 1.94x (679.3 / 349.9 tok/s) | 1.65x (707.3 / 429.9 tok/s) |
| 128->512 | batch=8, beams=4, bf16 | 2.27x (267.4 / 117.8 tok/s) | 3.00x (240.8 / 80.2 tok/s) | 2.36x (369.3 / 156.6 tok/s) | 3.31x (403.4 / 121.8 tok/s) |
| 128->512 | batch=8, beams=1, fp32 | 1.49x (320.3 / 215.2 tok/s) | 1.73x (215.5 / 124.7 tok/s) | 1.62x (494.0 / 305.3 tok/s) | 1.67x (388.0 / 232.5 tok/s) |
| 512->128 | batch=1, beams=1, bf16 | 1.32x (113.7 / 85.9 tok/s) | 0.85x (49.0 / 57.9 tok/s) | 1.44x (155.4 / 108.2 tok/s) | 0.89x (77.0 / 86.9 tok/s) |
| 512->128 | batch=8, beams=1, bf16 | 1.64x (364.2 / 222.7 tok/s) | 1.38x (317.6 / 229.3 tok/s) | 1.76x (544.4 / 308.5 tok/s) | 1.48x (569.0 / 385.6 tok/s) |
| 512->128 | batch=8, beams=4, bf16 | 2.11x (210.2 / 99.8 tok/s) | 5.19x (206.0 / 39.7 tok/s) | 2.29x (329.4 / 143.8 tok/s) | 7.46x (380.0 / 50.9 tok/s) |
| 512->128 | batch=8, beams=1, fp32 | 1.20x (185.2 / 153.8 tok/s) | 1.18x (116.9 / 98.9 tok/s) | 1.08x (254.1 / 234.3 tok/s) | 1.14x (213.1 / 187.4 tok/s) |
| 256->256 | batch=1, beams=1, bf16 | 1.42x (127.3 / 89.5 tok/s) | 1.22x (71.7 / 58.5 tok/s) | 1.47x (166.8 / 113.5 tok/s) | 1.29x (116.1 / 89.9 tok/s) |
| 256->256 | batch=8, beams=1, bf16 | 1.85x (506.3 / 273.9 tok/s) | 1.56x (412.0 / 264.2 tok/s) | 1.88x (663.3 / 352.7 tok/s) | 1.48x (661.4 / 446.2 tok/s) |
| 256->256 | batch=8, beams=4, bf16 | 2.19x (260.7 / 118.9 tok/s) | 2.61x (230.3 / 88.1 tok/s) | 2.40x (377.2 / 157.4 tok/s) | 2.95x (398.5 / 134.9 tok/s) |
| 256->256 | batch=8, beams=1, fp32 | 1.36x (307.2 / 225.2 tok/s) | 1.55x (195.6 / 126.4 tok/s) | 1.49x (453.7 / 304.7 tok/s) | 1.52x (357.6 / 235.2 tok/s) |
| Florence-2 | batch=1, beams=1, bf16 | 2.02x (37.4 / 18.5 tok/s) | 3.20x (40.0 / 12.5 tok/s) | 2.63x (62.7 / 23.8 tok/s) | 3.87x (70.0 / 18.1 tok/s) |
| Florence-2 | batch=8, beams=1, bf16 | 1.42x (164.7 / 116.1 tok/s) | 2.52x (179.2 / 71.2 tok/s) | 1.48x (243.3 / 164.2 tok/s) | 2.40x (305.5 / 127.1 tok/s) |
| Florence-2 | batch=8, beams=4, bf16 | 2.11x (126.1 / 59.9 tok/s) | 3.97x (129.0 / 32.1 tok/s) | 2.15x (171.8 / 80.0 tok/s) | 3.87x (202.4 / 52.2 tok/s) |
| Florence-2 | batch=8, beams=1, fp32 | 1.22x (78.3 / 64.1 tok/s) | 1.73x (59.3 / 34.2 tok/s) | 1.19x (133.3 / 111.9 tok/s) | 1.88x (121.2 / 64.5 tok/s) |

## Tests & Benchmarks

```bash
PYTHONPATH=. python tests/test_parity.py            # vs transformers, tiny random BART
PYTHONPATH=. python tests/test_florence2.py         # vs transformers, tiny random Florence-2
PYTHONPATH=. python tests/test_beam.py              # beam candidate selection
PYTHONPATH=. python tests/test_logits_processor.py  # LogitsProcessor parity vs transformers
PYTHONPATH=. python tests/test_engine.py            # lifetime, cache budgets, validation, loading
PYTHONPATH=. python tests/test_benchmarks.py        # benchmark helpers
PYTHONPATH=. python benchmarks/verify_parity.py     # real checkpoints (BART + Florence-2)
PYTHONPATH=. python benchmarks/benchmark.py         # BART benchmark vs transformers
PYTHONPATH=. python benchmarks/benchmark_florence2.py  # Florence-2 benchmark vs transformers
```
