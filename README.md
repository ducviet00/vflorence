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
| 128->512 | batch=1, beams=1, bf16 | 1.49x (128.6 / 86.2 tok/s) | 1.21x (74.6 / 61.6 tok/s) |
| 128->512 | batch=8, beams=1, bf16 | 1.95x (545.1 / 279.7 tok/s) | 1.65x (428.6 / 260.2 tok/s) |
| 128->512 | batch=8, beams=4, bf16 | 2.20x (263.9 / 119.7 tok/s) | 2.91x (235.0 / 80.7 tok/s) |
| 128->512 | batch=8, beams=1, fp32 | 1.50x (334.6 / 223.0 tok/s) | 1.74x (220.6 / 127.0 tok/s) |
| 512->128 | batch=1, beams=1, bf16 | 1.28x (111.5 / 87.2 tok/s) | 1.16x (68.4 / 58.8 tok/s) |
| 512->128 | batch=8, beams=1, bf16 | 1.28x (294.6 / 229.8 tok/s) | 1.28x (301.9 / 235.3 tok/s) |
| 512->128 | batch=8, beams=4, bf16 | 1.95x (207.4 / 106.4 tok/s) | 4.49x (191.1 / 42.5 tok/s) |
| 512->128 | batch=8, beams=1, fp32 | 1.05x (156.6 / 149.9 tok/s) | 1.17x (118.8 / 101.9 tok/s) |
| 256->256 | batch=1, beams=1, bf16 | 1.40x (126.6 / 90.1 tok/s) | 1.19x (72.9 / 61.3 tok/s) |
| 256->256 | batch=8, beams=1, bf16 | 1.78x (506.5 / 284.3 tok/s) | 1.46x (388.1 / 266.0 tok/s) |
| 256->256 | batch=8, beams=4, bf16 | 2.17x (264.4 / 121.7 tok/s) | 2.51x (223.9 / 89.3 tok/s) |
| 256->256 | batch=8, beams=1, fp32 | 1.38x (290.3 / 209.9 tok/s) | 1.57x (202.6 / 128.7 tok/s) |
| Florence-2 | batch=1, beams=1, bf16 | 2.28x (36.1 / 15.8 tok/s) | 3.12x (40.5 / 13.0 tok/s) |
| Florence-2 | batch=8, beams=1, bf16 | 1.45x (156.2 / 107.9 tok/s) | 2.36x (175.6 / 74.4 tok/s) |
| Florence-2 | batch=8, beams=4, bf16 | 2.10x (121.9 / 57.9 tok/s) | 3.95x (129.0 / 32.3 tok/s) |
| Florence-2 | batch=8, beams=1, fp32 | 1.23x (76.0 / 62.0 tok/s) | 1.80x (63.2 / 35.1 tok/s) |

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
