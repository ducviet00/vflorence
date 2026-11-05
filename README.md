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
| 128->512 | batch=1, beams=1, bf16 | 1.44x (124.9 / 86.7 tok/s) | 1.22x (78.7 / 64.5 tok/s) |
| 128->512 | batch=8, beams=1, bf16 | 1.92x (539.9 / 280.5 tok/s) | 1.69x (456.4 / 270.5 tok/s) |
| 128->512 | batch=8, beams=4, bf16 | 2.21x (263.3 / 118.9 tok/s) | 2.96x (251.2 / 84.8 tok/s) |
| 128->512 | batch=8, beams=1, fp32 | 1.48x (304.7 / 206.1 tok/s) | 1.76x (230.9 / 130.9 tok/s) |
| 512->128 | batch=1, beams=1, bf16 | 0.83x (72.0 / 86.3 tok/s) | 0.85x (53.9 / 63.2 tok/s) |
| 512->128 | batch=8, beams=1, bf16 | 1.57x (353.3 / 224.6 tok/s) | 1.43x (347.7 / 243.1 tok/s) |
| 512->128 | batch=8, beams=4, bf16 | 1.96x (202.9 / 103.8 tok/s) | 4.72x (205.7 / 43.6 tok/s) |
| 512->128 | batch=8, beams=1, fp32 | 1.04x (148.7 / 143.3 tok/s) | 1.21x (125.4 / 103.3 tok/s) |
| 256->256 | batch=1, beams=1, bf16 | 1.40x (123.7 / 88.2 tok/s) | 1.14x (74.4 / 65.3 tok/s) |
| 256->256 | batch=8, beams=1, bf16 | 1.89x (508.4 / 269.2 tok/s) | 1.53x (433.2 / 282.4 tok/s) |
| 256->256 | batch=8, beams=4, bf16 | 2.23x (258.6 / 116.0 tok/s) | 2.65x (244.5 / 92.4 tok/s) |
| 256->256 | batch=8, beams=1, fp32 | 1.36x (273.9 / 200.9 tok/s) | 1.60x (213.3 / 133.4 tok/s) |
| Florence-2 | batch=1, beams=1, bf16 | 2.33x (34.3 / 14.7 tok/s) | 3.07x (40.7 / 13.3 tok/s) |
| Florence-2 | batch=8, beams=1, bf16 | 1.53x (149.2 / 97.8 tok/s) | 2.35x (183.7 / 78.3 tok/s) |
| Florence-2 | batch=8, beams=4, bf16 | 2.19x (115.0 / 52.5 tok/s) | 4.16x (143.7 / 34.1 tok/s) |
| Florence-2 | batch=8, beams=1, fp32 | 1.25x (74.7 / 59.6 tok/s) | 1.85x (66.4 / 35.9 tok/s) |

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
