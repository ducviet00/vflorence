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

Measured on GCP `c4a-standard-8` (Axion / Neoverse-V2, 8 cores) and `c4-standard-8`
(Intel Xeon 8581C, 4 cores / 8 threads, AMX) with `torch 2.15.0+cpu`, `transformers 5.22.1`,
and `vllm-cpu-attn 0.1.0` (`benchmarks/benchmark.py` and `benchmarks/benchmark_florence2.py`):

| point | config | c4a-standard-8 (Axion) | c4-standard-8 (Xeon 8581C) |
|---|---|---:|---:|
| 128->512 | batch=1, beams=1, bf16 | 1.41x (127.2 / 90.4 tok/s) | 1.19x (73.4 / 61.7 tok/s) |
| 128->512 | batch=8, beams=1, bf16 | 2.02x (545.7 / 270.0 tok/s) | 1.64x (422.8 / 258.3 tok/s) |
| 128->512 | batch=8, beams=4, bf16 | 2.27x (267.4 / 117.8 tok/s) | 3.00x (240.8 / 80.2 tok/s) |
| 128->512 | batch=8, beams=1, fp32 | 1.49x (320.3 / 215.2 tok/s) | 1.73x (215.5 / 124.7 tok/s) |
| 512->128 | batch=1, beams=1, bf16 | 1.32x (113.7 / 85.9 tok/s) | 0.85x (49.0 / 57.9 tok/s) |
| 512->128 | batch=8, beams=1, bf16 | 1.64x (364.2 / 222.7 tok/s) | 1.38x (317.6 / 229.3 tok/s) |
| 512->128 | batch=8, beams=4, bf16 | 2.11x (210.2 / 99.8 tok/s) | 5.19x (206.0 / 39.7 tok/s) |
| 512->128 | batch=8, beams=1, fp32 | 1.20x (185.2 / 153.8 tok/s) | 1.18x (116.9 / 98.9 tok/s) |
| 256->256 | batch=1, beams=1, bf16 | 1.42x (127.3 / 89.5 tok/s) | 1.22x (71.7 / 58.5 tok/s) |
| 256->256 | batch=8, beams=1, bf16 | 1.85x (506.3 / 273.9 tok/s) | 1.56x (412.0 / 264.2 tok/s) |
| 256->256 | batch=8, beams=4, bf16 | 2.19x (260.7 / 118.9 tok/s) | 2.61x (230.3 / 88.1 tok/s) |
| 256->256 | batch=8, beams=1, fp32 | 1.36x (307.2 / 225.2 tok/s) | 1.55x (195.6 / 126.4 tok/s) |
| Florence-2 | batch=1, beams=1, bf16 | 2.02x (37.4 / 18.5 tok/s) | 3.20x (40.0 / 12.5 tok/s) |
| Florence-2 | batch=8, beams=1, bf16 | 1.42x (164.7 / 116.1 tok/s) | 2.52x (179.2 / 71.2 tok/s) |
| Florence-2 | batch=8, beams=4, bf16 | 2.11x (126.1 / 59.9 tok/s) | 3.97x (129.0 / 32.1 tok/s) |
| Florence-2 | batch=8, beams=1, fp32 | 1.22x (78.3 / 64.1 tok/s) | 1.73x (59.3 / 34.2 tok/s) |

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
