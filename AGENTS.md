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

- Python env: `.venv/bin/python` (Python >= 3.12, CPU PyTorch, `vllm-cpu-attn`, `transformers`, `safetensors`, `Pillow`, `xxhash`).
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
  __init__.py          package exports (LLM, Request, RequestOut, SamplingParams)
  config.py            Config (engine hyperparameters + HF/text/generation configs)
  llm.py               LLM facade (subclass of LLMEngine)
  sampling_params.py   SamplingParams (greedy + beam search)
  engine/
    sequence.py        Sequence
    block_manager.py   paged blocks: allocate / fork / copy-on-write / free
    scheduler.py       continuous batching
    model_runner.py    model weights, batching, KV cache, encoder caching, forward
    llm_engine.py      LLMEngine, Request, RequestOut: step loop, greedy, beam search
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
benchmarks/stage_latency.py        per-stage ms vs HF (vision / encode / decode step)
benchmarks/continuous_batching.py  mixed-length Florence-2 workload vs HF padded batches
benchmarks/utils.py                shared image loader and timer
tests/verify_parity.py             real-checkpoint parity vs HF
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
    _encode(fresh)                # batched encoder (grouped by length/image shape) -> paged cross-attn K/V
                                  #   vision encoder runs only on mm_cache misses
    _prepare(seqs)                # input_ids/positions/slot_mapping/self_metadata; ensure blocks/CoW
    set_context(...)              # Context: context_lens, block_tables, slot_mapping,
                                  #   query_start_loc, self_metadata, isa, cross_staged
    decoder forward               # compiled decoder + lm_head; custom ops call vllm_cpu_attn
LLMEngine.step()                  # argmax (greedy) or beam-expand; append; free finished
```

`LLMEngine.run()` loops `step()` until `is_finished()`.
`LLMEngine.generate(requests, params)` is the offline entrypoint: it adds the
whole batch under a single `SamplingParams` and runs to completion.

`ModelRunner._sample` runs `log_softmax` (beams only) and the logits processor
once over the whole batch, with histories left-padded by -1 and true lengths
passed in, so sequences of different decoder lengths share one call.
`_beam_search` then runs one `topk` per row over the vocab, adds each beam's
`cum_logprob` to those survivors only (it is constant per row, so a request's
top K always lies within its rows' top K), and merges each request's rows in
Python. `Scheduler._postprocess_beam` forks the surviving candidates.

`models/bart.py` uses `F.scaled_dot_product_attention` for `BartEncoderSelfAttention`
(dense, no cache) and calls the opaque `decoder_self_attention` and
`decoder_cross_attention` custom ops from `layers/attention.py`. Before the
decoder forward, `ModelRunner.run` points the step's sequences at their pages of
each layer's encoder K/V pool (`attention.stage_cross` -> `Context.cross_staged`); the opaque
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
3. **Paged KV cache.** Shape `[num_blocks, num_heads, block_size, head_dim]`
   with internal layout determined by the selected ISA (`vec` / `amx` on x86,
   `neon` on arm64) and `block_size` (32 on x86, 16 on arm64; must be a
   multiple of 16). `head_dim` must be in `{32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512}`.
4. **Cross-attention pool.** Each `BartDecoderCrossAttention` holds a paged
   encoder K/V pool `k_cache` / `v_cache`
   (`[num_pages, num_heads, block_size, head_dim]`), which starts empty and at
   least doubles when it runs out (it never shrinks). `_encode` writes a
   request's K/V once into pages from `ModelRunner._cross_free` and records
   `_cross_pages[request_id] = (pages, enc_len)`; all beams of a request share
   them and `ModelRunner.clear_request` returns them. `stage_cross` copies
   nothing: it builds the step's block table and scheduler metadata, memoized on
   `ModelRunner._cross_staged` keyed by `tuple(request_ids)`.
5. **Beam search = sequences + block sharing.** Every beam is an ordinary
   `Sequence` with the request's `request_id`. `BlockManager.fork` shares the
   parent's blocks (`refcount++`); `BlockManager.ensure_capacity` copy-on-writes
   any block that will be written while shared (it returns the `(src, dst)` pair;
   `ModelRunner._copy_blocks` copies all of a step's pairs at once). Children are created **before**
   parents are freed.
6. **Multimodal cache.** `ModelRunner.mm_cache` is an LRU (`Config.mm_cache_size`
   entries, 0 disables) of `Sequence.mm_hash -> image_embeds` (vision tower +
   projector output), kept across `generate` calls; `mm_cache_hits` /
   `mm_cache_misses` count lookups. `mm_hash` is the xxh3_128 of `pixel_values`
   (~0.1 ms for 768x768 fp32; not cryptographic); the processor still runs every time (~2.5 ms).
7. **Scheduling.** `Scheduler.schedule()` returns *all* running sequences each
   step, so a request's beams are never split. Keep `max_num_seqs >= num_beams`.
8. **Global context.** `utils/context.set_context(...)` is called once per
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
PYTHONPATH=. $P tests/verify_parity.py --dtype bfloat16
TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH=. $P tests/verify_parity.py --dtype float32

# examples
PYTHONPATH=. $P examples/summarize.py
PYTHONPATH=. $P examples/florence2.py

# benchmarks (run on GCP spot VMs via spot-bench-gcp, not on the dev laptop)
# one backend per run: repeat each with --backend hf
PYTHONPATH=. $P benchmarks/stage_latency.py --backend fast
PYTHONPATH=. $P benchmarks/stage_latency.py --backend fast --model facebook/bart-large-cnn --enc-len 512
PYTHONPATH=. $P benchmarks/continuous_batching.py --backend fast --beamsearch 1   # and 4; --mm-cache-size 0 disables the image cache
```

## 7. Known issues

- **CRITICAL: Florence-2-large produces garbage (open, not yet investigated).**
  `florence-community/Florence-2-large`, bf16, greedy, max num seq 32 on
  `c4a-standard-16` (2026-11-11, `benchmarks/continuous_batching.py`): 84 of 128
  requests decode only repeated `<s>` tokens until `max_tokens` (512). Every
  `<REFERRING_EXPRESSION_SEGMENTATION>` request fails, plus 28/32 `<OCR>` and 23/32
  `<OD>`; most `<DETAILED_CAPTION>` outputs look correct. Florence-2-base is fine
  on the same box. Not yet compared against HF (not checked whether HF large
  behaves, or whether fp32 / x86 differ). Start with one failing request, e.g.
  `<OD>` on COCO `000000002153.jpg`, fast vs HF in bf16 and fp32. Do not publish
  Florence-2-large numbers until this is fixed.
