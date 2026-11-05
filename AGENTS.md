# AGENTS.md — fast-encdec handover

Guidance for AI agents (and humans) working on this repository.

## 1. What this is

`fast-encdec` is a **small, readable CPU inference engine for BART and Florence-2**,
in the spirit of nano-vllm. It implements only what matters for offline
encoder-decoder decoding on CPU:

- **Paged KV cache** with block reference counting and copy-on-write.
- **Continuous batching** (requests join/leave the running batch freely).
- **No padding** — attention is per-sequence, so variable-length prompts and
  batches waste no compute.
- **Greedy and beam search** on top of the same paged cache.
- **FP32 and BF16** (`torch.float32`, `torch.bfloat16`).
- **Frozen `torch.compile` on CPU** paired with [`vllm-cpu-attention`](https://github.com/ducviet00/vllm-cpu-attention) (`vllm_cpu_attn`). One dynamic graph covers decoder + lm head and one the encoder; Inductor weight-freezing packs every GEMM into oneDNN's blocked layouts (BF16 `avx512_core_bf16` / AMX on modern x86, ACL on aarch64), while decoder self- and cross-attention run through opaque `torch.library.custom_op`s backed by `vllm_cpu_attn`.

The model (`fastencdec/models/bart.py`) is a near-verbatim copy of the
HuggingFace BART reference; the attention kernels differ, and the attention
q/k/v projections are fused into one GEMM (`utils/loader` concatenates the HF
parts).

**Florence-2** (`fastencdec/models/florence2.py`) reuses that paged BART as its
language model and pairs it with the DaViT vision encoder
(`Florence2VisionBackbone`) and `Florence2MultiModalProjector`, vendored in the
same `fastencdec/models/florence2.py` (an optimized copy of the `transformers`
reference). Visual tokens are spliced into the encoder input at the `<image>`
placeholders; the decoder is unchanged.

### Non-goals / do not add without discussion

- **CPU only** (`float32` and `bfloat16`), **BART and Florence-2 only**, **greedy and beam search only** (no multinomial sampling, no CUDA).
- No quantization, prefix caching or chunked prefill.
- **Single external kernel**: [`vllm-cpu-attention`](https://github.com/ducviet00/vllm-cpu-attention) (`vllm_cpu_attn`), wrapped in `fastencdec/layers/attention.py`. Do not add other external kernels or fallback backends without explicit approval.

## 2. Environment

- Python env: `.venv/bin/python` (Python >= 3.12, CPU PyTorch, `vllm-cpu-attn`, `transformers`, `safetensors`, `Pillow`).
- Dev CPU: AMD Ryzen 7 7840H (Zen 4) — has `avx512_bf16`, **no** AMX, **no** `avx512_fp16`. Prefer `bfloat16` or `float32`; never `float16`. Do not run heavy full benchmarks on this laptop (smoke tests / unit tests only); run full benchmarks on disposable GCP spot VMs (`spot-bench-gcp` skill: `bench-start axion` for `c4a-standard-8` and `bench-start xeon` for `c4-standard-8`, then `bench-stop` immediately when done).
- Install in a Python 3.12+ venv:
  ```bash
  uv pip install torch torchvision setuptools --index-url https://download.pytorch.org/whl/cpu
  uv pip install --no-build-isolation git+https://github.com/ducviet00/vllm-cpu-attention.git
  uv pip install --no-build-isolation -e .
  ```

## 3. Layout

Mirrors nano-vllm's package layout (`engine/`, `layers/`, `models/`, `utils/`).

```
fastencdec/
  __init__.py          package exports (LLM, SamplingParams, RequestOutput, ...)
  config.py            Config (engine hyperparameters)
  llm.py               unified LLM facade (BART + Florence-2, vLLM-style API)
  outputs.py           RequestOutput / CompletionOutput (vLLM-shaped)
  sampling_params.py   SamplingParams (greedy + beam search)
  engine/
    sequence.py        Sequence
    block_manager.py   paged blocks: allocate / fork / copy-on-write / free
    scheduler.py       continuous batching
    model_runner.py    batching, KV cache, encoder caching, forward
    llm_engine.py      LLMEngine: step loop, greedy, beam search
  layers/
    attention.py       vllm_cpu_attn custom ops (decoder self/cross attention + staging)
    logits_processor.py  pure-torch LogitsProcessor (built once per generate)
  models/
    bart.py            BART model (encoder / decoder / cross-attn)
    florence2.py       Florence-2 (vendored DaViT + projector + BART LM)
  utils/
    context.py         per-step Context read by the attention custom ops
    loader.py          HF checkpoint -> model weights (concatenates fused q/k/v)
examples/summarize.py
examples/florence2.py
benchmarks/benchmark.py
benchmarks/benchmark_vllm.py
benchmarks/benchmark_florence2.py
benchmarks/verify_parity.py
benchmarks/throughput.py
benchmarks/gemm.py
benchmarks/florence2_tasks.py
benchmarks/profile_florence2.py
tests/test_parity.py
tests/test_florence2.py
tests/test_beam.py
tests/test_engine.py
tests/test_benchmarks.py
tests/test_logits_processor.py
```

## 4. How a step works

```
Scheduler.schedule()              # admit waiting up to max_num_seqs; return ALL running
ModelRunner.run(seqs)
    fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
    _encode(fresh)                # batched encoder (grouped by length/image shape) -> dense cross-attn K/V
    _prepare(seqs)                # input_ids/positions/slot_mapping/self_metadata; ensure blocks/CoW
    set_context(...)              # Context: context_lens, block_tables, slot_mapping,
                                  #   query_start_loc, self_metadata, isa, cross_staged
    decoder forward               # compiled decoder + compute_logits; custom ops call vllm_cpu_attn
LLMEngine.step()                  # argmax (greedy) or beam-expand; append; free finished
```

`LLMEngine.run()` loops `step()` until `is_finished()`.
`LLMEngine.generate(requests, params)` is the offline entrypoint: it adds the
whole batch under a single `SamplingParams` and runs to completion.

`LLMEngine._postprocess` groups sequences by request, then batches every beam
request that shares a `(beam width, live beam count, decoder length)` into one
`_beam_group`: a single `log_softmax` + logits-processor call + `topk` over
`[num_requests, beam_count * vocab]`.

`models/bart.py` uses `F.scaled_dot_product_attention` for `BartEncoderSelfAttention`
(dense, no cache) and calls the opaque `decoder_self_attention` and
`decoder_cross_attention` custom ops from `layers/attention.py`. Before the
decoder forward, `ModelRunner.run` stages each layer's dense encoder K/V into a
paged cache (`attention.stage_cross` -> `Context.cross_staged`); the opaque
cross-attention op reads that per-step state at call time so it stays out of the
compiled graph.

## 5. Core invariants (read before changing anything)

1. **Token bookkeeping.** `Sequence.token_ids` = decoder prompt + generated
   tokens. `num_cached_tokens` = how many of them have K/V in the paged cache.
   The tokens fed next step are `token_ids[num_cached_tokens:]`. After
   `_prepare`, `num_cached_tokens == len(token_ids)`; a token is then
   appended, so between steps there is exactly one uncached token. **Every step
   feeds exactly one token per sequence.**
2. **Positions** are absolute indices into `token_ids` (0-based).
   `BartLearnedPositionalEmbedding` adds the `+2` BART offset internally.
3. **Paged KV cache.** Allocated via `vllm_cpu_attn.allocate_kv_cache` with shape
   determined by the selected ISA (`vec` / `amx` on x86, `neon` on arm64) and
   `block_size` (32 on x86, 16 on arm64; must be a multiple of 16).
   `head_dim` must be in `{32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512}`.
4. **Cross-attention staging.** Each `BartDecoderCrossAttention` holds
   `encoder_kv_cache: dict[request_id -> (k, v)]`, each
   `[enc_len, num_heads, head_dim]`, filled once in `_encode`, shared by all
   beams of a request and freed in `ModelRunner.clear_request`. `stage_cross`
   packs the active requests into a paged cache (plus scheduler metadata) memoized
   on `layer.cross_staged` keyed by `tuple(request_ids)`.
5. **Beam search = sequences + block sharing.** Every beam is an ordinary
   `Sequence` with the request's `request_id`. `BlockManager.fork` shares the
   parent's blocks (`refcount++`); `BlockManager.ensure_capacity` copy-on-writes
   any block that will be written while shared. Children are created **before**
   parents are freed.
6. **Scheduling.** `Scheduler.schedule()` returns *all* running sequences each
   step, so a request's beams are never split. Keep `max_num_seqs >= num_beams`.
7. **Global context.** `utils/context.set_context(...)` is called once per
   forward; it builds the step's `Context` and attention custom ops read it via
   `get_context()`. It is not thread-safe and assumes a single in-flight
   forward.

## 6. Running things

```bash
P=.venv/bin/python
export HF_HUB_OFFLINE=1

# unit / parity tests
PYTHONPATH=. $P tests/test_parity.py
PYTHONPATH=. $P tests/test_florence2.py
PYTHONPATH=. $P tests/test_beam.py
PYTHONPATH=. $P tests/test_logits_processor.py
PYTHONPATH=. $P tests/test_engine.py
PYTHONPATH=. $P tests/test_benchmarks.py

# real-checkpoint parity (fp32 exact; bf16 same length)
PYTHONPATH=. $P benchmarks/verify_parity.py --dtype bfloat16
TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH=. $P benchmarks/verify_parity.py --dtype float32

# examples
PYTHONPATH=. $P examples/summarize.py
PYTHONPATH=. $P examples/florence2.py

# benchmarks (run on GCP spot VMs via spot-bench-gcp, not on the dev laptop)
PYTHONPATH=. $P benchmarks/benchmark.py --json bart.json
PYTHONPATH=. $P benchmarks/benchmark_florence2.py --json florence2.json
```
