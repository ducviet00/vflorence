# Why `torch.compile` + graph freezing does (does not) produce `mkldnn._linear_pointwise` weight prepacking

Findings, 2026-10-17, written on torch 2.14; **re-audited 2026-11-12 on torch
2.15.0+cpu** (the pinned version; Python 3.12) on the dev box (AMD Ryzen 7 7840H,
Zen 4, `avx512_bf16`, no AMX) and a `c4a-standard-8` (Neoverse-V2, ACL). Source
line numbers below are from the 2.15 wheel; the audit's corrections are inline
and its new measurements are in §6.3. The §4.2/§6.1 break-census numbers
were taken on a dev tree that also carried a (later unmerged) AMX attention
experiment; this branch carries everything else described here.

Answers for the three questions that motivated this doc:

1. **Does `torch.compile` + graph freezing emit `torch.ops.mkldnn._linear_pointwise`
   with prepacked weights?** On torch 2.15, **yes** — for bf16, for bf16 under
   `torch.amp.autocast`, and even with dynamic batch sizes. It also does it on
   *this engine*, graph breaks and all: the frozen decoder emits and executes
   **48 packed GEMMs per decoder forward** (runtime-verified). The prior note in
   AGENTS.md §7 ("[the recipe] does work, but only on models that are pure
   torch", "It does not pay off on this engine") was right about the outcome
   and wrong about the mechanism: the pack fires on this engine, graph breaks
   and all — what loses is the per-step Python overhead of a ~70-break
   shattered graph.
2. **Where does "no `_linear_pointwise`" actually come from?** Five concrete,
   silent traps (§3). The two that apply to us: freezing is *off by default* and
   in torch 2.15 the weight-pack pass exists **only inside the freezing
   pipeline** — `config.cpp.weight_prepack=True` alone does nothing; and
   compiling a *free function* over plain weight tensors (the old
   `enable_compile` helpers) never packs because those weights are **user
   arguments**, not frozen constants (`nn.Parameter` arguments do pack, §3).
3. **Do we need `torch.amp()`?** **No.** Autocast is *sufficient* but not
   *necessary*. The pass keys off (weight = frozen constant, dtype bf16/fp16, or
   fp32 with `mkldnn.matmul.fp32_precision` fast-math) — not off autocast.
   A module already cast to bf16 packs identically (verified).

---

## 1. The gate chain in torch 2.15 source

Weight prepacking on CPU is implemented as *freezing* FX patterns. This is why
the 2024 blogs' excerpt (`torch/_inductor/mkldnn.py:mkldnn_fuse_fx` /
`pack_module`) is stale: that entry point no longer exists in 2.15
(`grep -r mkldnn_fuse_fx torch/_inductor/` → 0 hits).

Decision tree, all paths pinned to the installed source
(`torch/_inductor/`, line numbers from the installed 2.15 wheel):

```
torch.compile(model)  ──  first forward call is the trigger
  │
  ├─ config.freezing == True AND torch.is_grad_enabled() == False   [*]
  │      → compile_fx.py picks fw_compiler_freezing
  │      → freezing._freeze() replaces every non-mutated parameter
  │        placeholder with a graph constant (`_frozen_paramN`)
  │      → fx_passes.freezing_patterns.freezing_passes() runs
  │            └─ lazy_init() registers _mkldnn_weight_pack_init()   [**]
  │                  └─ pattern aten.addmm / aten.mm (pass_number=1)
  │                        extra_check = _is_packable_linear         [***]
  │                        →  pack_linear_weight  (mkldnn._reorder_linear_weight,
  │                                             constant-folded offline during
  │                                             compile → another _frozen_param)
  │                        →  pack_linear (op choice, §2)
  │
  └─ otherwise: plain fw_compiler_base(is_inference=True)
         → the weight-pack pass is NEVER REGISTERED
         → all linears stay `extern_kernels.addmm` (verified on disk)
```

- `[*] compile_fx.py:3414`
  `if config.freezing and not torch.is_grad_enabled():` — grad mode is sampled
  **at trace time** (inside the first call). Note that the choice between
  `fw_compiler` (grad) and `inference_compiler` is made here; AOTAutograd later
  picks the inference compiler when the graph doesn't need grad.
- `[**] fx_passes/freezing_patterns.py:92-102` —
  `lazy_init()` registers `_mkldnn_weight_pack_init()` **only** if
  `torch._C._has_mkldnn and config.cpp.weight_prepack`; these patterns are
  registered into `pass_patterns[]` which is applied *only* from
  `freezing_passes()` (line 37-89), i.e. only inside `freezing._freeze()`
  (`freezing.py:77-147`).
- `[***] fx_passes/mkldnn_fusion.py:1288-1370` — `_is_packable_linear`:
  - weight arg must be a **graph constant** (`get_attr`) or a `cat` of constants
    (`is_const_or_cat_by_const`, lines 1293-1298) — the frozen params satisfy
    this, *user-supplied tensors do not*;
  - both args 2-D, on CPU, not fp64;
  - fp32 requires MKL (x86) or ACL (aarch64), lines 1332-1339;
  - bf16/fp16 require oneDNN's low-precision support
    (`is_mkldnn_bf16_supported`, `utils.py:5220`).

Defaults (`_inductor/config.py`): `cpp.weight_prepack` is **already
True** (line 1971); `freezing` is **False** (line 1821) — so of the pair in the
classic recipe, the relay that actually matters is `freezing`.

## 2. Which packed op you get, by dtype

From `mkldnn_fusion.py:109-159` (`pack_linear_weight` / `pack_linear`):

| weight dtype | condition | packed op |
|---|---|---|
| bf16 / fp16 | always | `torch.ops.mkldnn._linear_pointwise` |
| fp32 | `torch._C.has_mkl` and no fast-math | `torch.ops.mkl._mkl_linear` |
| fp32 | `torch.backends.mkldnn.matmul.fp32_precision ∈ {bf16, tf32}`* | `torch.ops.mkldnn._linear_pointwise` |
| fp32 on aarch64 | ACL supported | `torch.ops.mkldnn._linear_pointwise` |
| any dtype, dynamic batch (`has_free_symbols`) | — | `torch.ops.mkldnn._linear_pointwise` (weight reordered with `optimization_hint(batch)`); so fp32 uses MKL only in static graphs |

\* this is exactly the setting the AWS Graviton blog imports via
`DNNL_DEFAULT_FPMATH_MODE=BF16`; setting
`torch.backends.mkldnn.matmul.fp32_precision = "bf16"` switched our fp32 toy's
packed op from `mkl._mkl_linear` to `mkldnn._linear_pointwise` (verified by
grepping the emitted code).

So expecting `mkldnn._linear_pointwise` from an **fp32** recipe without
fast-math/autocast is itself a mistaken expectation — you get `mkl._mkl_linear`
(prepacking still happens, just under the MKL namespace).

## 3. The traps that silently yield *no* `_linear_pointwise`

Each one exercised on this box (see §4 evidence column):

1. **`config.freezing` not set** (default False, `config.py:1821`).
   The pack pass lives only in the freeze pipeline; `weight_prepack=True` alone
   does nothing, and the GEMMs compile to plain `extern_kernels.addmm`.
2. **The first (compiling) call made with grad enabled** — `freezing` silently
   degrades to the non-freezing compiler per the `[*]` gate. In our test the
   *call made under `no_grad`* afterwards recompiles with guards on grad mode
   and packs then, so you pay for two compiles instead of one (24-pack before
   and 24 after; useful first-call lesson, not a correctness trap).
3. **`torch.compile` over a free function whose weights are plain tensors**
   (passed as arguments or captured in a closure): they are never frozen
   constants, `is_const_or_cat_by_const` fails → **zero packing** (measured 0/2).
   `nn.Parameter`s passed as arguments *are* frozen and packed (measured 2/2 and
   24/24), so this is about the tensor type, not about free functions as such.
4. **Manual prepacking (`layers/packed_linear.py`) replaces `nn.Linear.forward`
   with a closure wrapped in `@torch.compiler.disable`.** Those GEMMs are
   invisible to Dynamo by construction — Inductor cannot pack weights it never
   sees, and the frozen pipeline cannot fuse around a disabled region. The two
   routes are mutually exclusive in the current wiring (they need not be: see
   §6 — and note that on torch 2.14 Inductor *can* consume the opaque blocked
   weight of an *explicit* `mkldnn._linear_pointwise` call: the "fails on
   `stride=[1, 0]`" observation in the old Florence-2 note did not reproduce on
   a chained-packed-forward toy compiled today; worth rechecking next time
   `compile_mm_encoder` is touched).
5. **dtype mismatch of expectations** (§2): fp32 without fast-math packs into
   `mkl._mkl_linear`; scanning `prof` for `mkldnn._linear_pointwise` in an fp32
   run finds none even though packing all happened.

Not a trap (we back this now with the engine run): **graph breaks don't stop
packing.** Each broken subgraph is frozen and packed independently.

## 4. Empirical evidence

### 4.1 Toy module — 12×[Linear(1024→4096), GELU, Linear(4096→1024)], batch 8

Metric: `torch._dynamo.utils.counters["inductor"]
["mkldnn_linear_weight_pack_matcher_count"]` — one per `aten.mm`/`aten.addmm`
node rewritten to `reorder_linear_weight (+ bias) + pointwise linear`.
24 GEMMs = 24/24 packed. Timings wall-clock per call, CPU.

| variant | packed | op in code | compiled ms | eager ms |
|---|---|---|---:|---:|
| bf16 (weights pre-cast, **no autocast**), `no_grad`, freezing | 24 | `_linear_pointwise` | 4.53 | 5.99 |
| bf16 but first call grad-enabled | 24 (2nd compile) | same | 4.92 | 6.40 |
| fp32, `no_grad`, freezing | 24 | `mkl._mkl_linear` | 17.80 | 21.60 |
| fp32 **+ `torch.amp.autocast("cpu", bfloat16)`, traced under amp** | 24 | `_linear_pointwise` | 4.56 | 38.18 (with autocast) |
| fp32 + `mkldnn.matmul.fp32_precision="bf16"` | 24 | `_linear_pointwise` | — | — |
| bf16 + `dynamic=True`, batch 8/1/3 | 24 | `_linear_pointwise` | 4.48 | 6.04 |
| bf16 + opaque-packaged weights + explicit `_linear_pointwise`, **no** `compiler.disable` | n/a | `_linear_pointwise` extern (60× in code) | 4.58 | 4.53 |
| `torch.compile` over a free function, weights = plain tensor args | **0** | plain `mm`/`addmm` externs | — | — |
| same, weights = `nn.Parameter` args | 24 | `_linear_pointwise` | 4.60 | 5.82 |

Blogs' environment check: the same recipe was measured on Graviton (ACL) and on
x86 with `DNNL_DEFAULT_FPMATH_MODE` (cf. §2); those posts show the packed op
`mkldnn::_linear_pointwise` in their torch.profiler / code tables.

### 4.2 The engine — `bart-large-cnn`, bf16, batch 8, 256→256 greedy,
manual prepack stubbed out (`packed_linear.apply ≡ 0`), `Config.compile_model=False`,
then `torch.compile(engine.runner.decoder, dynamic=False)`,
`torch._inductor.config.freezing = True`, decode runs under the engine's own
`torch.no_grad()` (`engine/model_runner.py:151`):

- compile-time: `mkldnn_linear_weight_pack_matcher_count = 48` — the 5 GEMMs per
  decoder layer × 12 (qkv, self-out, cross-q, fc1, fc2).
- emitted code (TORCH_COMPILE_DEBUG dump): 47 output modules; the packed lines
  look like
  `buf0 = torch.ops.mkldnn._linear_pointwise.default(arg2_1, _frozen_param3, _frozen_param1, 'none', [-1], '')`
  — the `_reorder_linear_weight` call was constant-folded at compile time into
  a new `_frozen_param`, i.e. *the same load-time reorder our `pack_weights`
  performs, done per compile instead of once at load*.
- runtime (torch.profiler, 24-token generate, 8 seqs):
  `mkldnn::_linear_pointwise count=1152` (= 48 × 24 decode steps), while the
  same generate *without* freezing had `aten::addmm 1788` (the frozen run drops
  it to 636: the remainder is the encoder + LM head + per-step staging GEMMs
  which live outside the compiled decoder).
- wall time (256→256, b8, best of 3): **compiled 7006 ms vs eager 5919 ms
  (0.84×)** — the *same* breakdown as the AGENTS.md table (7490 vs 6700 there),
  even though packing demonstrably happened. Compile time itself is tiny
  (~0.1 s per subgraph, a few s total), so the regression is entirely in the
  hot path. (After the §6.1 capability-hoist fix the same experiment reads
  6467 vs 5860 ms = 0.91×, with the break census below.)

Where the wall time goes (24-token generate profile; the break census is from
dynamo's compile report):

| overhead signal | count/generate |
|---|---:|
| graph breaks per decoder-forward trace (compile-time census) | ~70 pre-fix → ~53 post-fix §6.1 |
| `TorchDynamo Cache Lookup` (guard checks) | 8040 |
| `Pregraph bytecode` | 2640 |
| `AOTDispatcher Runtime Wrapper Prologue` | 2640 |
| `Torch-Compiled Region` dispatches | dozens per step (~12/layer) |
| `aten::clone` / `aten::contiguous` between subgraphs | 432 / 420 |

Of the ~70 breaks/forward, **52 are ours to remove today**: 39 from
`torch.cpu._is_avx512_supported` (+ its bf16 sibling) and 13 from
`torch.cpu._init_amx` — the latter only because the census tree still carried
the AMX attention experiment; this branch's `layers/cpu_attn.py`
`_x86_kernel_ok()` contains only the AVX-512 pair, so its per-call break total
is 39. They run inside `select_isa()` on the op-dispatch path on *every*
attention call (`layers/attention.py:42,168,284`). They are torch CPU-API calls
resolved
per process, so caching the answers as plain bools on the module removes the
breaks (`lru_cache` does not help under
Dynamo anyway — it re-fires per fake-tensor frame).
The remaining 17 (9 `attention_forward`, 8 `reshape_and_cache`) are the pybind
kernels that need `torch.library.custom_op` + `register_fake`, as AGENTS.md
§7 already diagnosed.

**Conclusion for the engine:** the recipe works; the engine's *cost structure*
(per-step Python dispatch across a shattered graph) is what loses to eager,
since our `pack_weights` reaches the identical oneDNN blocked-weight op with
zero compile machinery. Freezing reaches exactly the same packed weights — it
is not a different optimization we are missing.

## 5. Do you need `torch.amp`?

Short answer: **no**.

- The packing condition depends on frozen constants + GEMM dtype — not on
  autocast. bf16 module + `no_grad` + `freezing=True` packs 24/24 (§4.1 row 1).
- `torch.amp.autocast("cpu", torch.bfloat16)` is the recommended recipe in the
  blogs because *their models are fp32 modules* wanting bf16 GEMMs. Under
  compile, autocast's `weight` casts get constant-folded during freezing, so an
  fp32+autocast run packs too (§4.1 row 4). If you instead cast the module (as
  our engine does), you need no autocast at all.
- For fp32 kernels there is one autocast-*adjacent* switch worth knowing:
  `torch.backends.mkldnn.matmul.fp32_precision = "bf16"` (or
  `DNNL_DEFAULT_FPMATH_MODE=BF16` from the Graviton post) — it moves fp32 GEMMs
  (and their packed form) from `mkl._mkl_linear` to the oneDNN
  `_linear_pointwise` bf16-accelerated path (§2). It trades exact fp32 for
  reduced-precision accumulation — the engine's exact-fp32 parity (the fp32
  rows of `benchmarks/verify_parity.py`) would no longer hold.

## 6. What to actually do (if/when revisiting compile on this engine)

1. ~~**Freeze the capability checks.**~~ **Done** (same-day): `cpu_attn.py`
   resolves the AVX-512 capability pair once into module globals
   (`_resolve_isa`, minus the census tree's AMX probe)
   and caches the built extension in a plain global (`_BUILT`) instead of
   `lru_cache` — Dynamo ignores cache wrappers and was tracing `_build()` per
   call site (41 `posix.stat` breaks from `os.path.exists`). Post-fix break
   census on the engine: **53, all pybind** (`reshape_and_cache` ×40,
   `attention_forward` ×9, `build_metadata` ×4); compile+freezing now measures
   **6467 ms vs eager 5860 (0.91x, was 0.84x)** with packing unchanged (48),
   and the eager default is marginally faster too (the baseline table in
   AGENTS.md was refreshed). Tests pass (`test_parity`, `test_attn_backend`,
   `test_beam`, `test_florence2`).
2. ~~**Verdict on replacing the manual prepack with compile+freezing: no.**~~
   **Superseded:** the custom-op re-plumb *did* land, and it changed the
   verdict.  See §6.1 below.
3. Never expect packing from compiled *helpers* (trap §3.3); compile the
   module's own forward.

### 6.2 The landed state (`compile` is the CPU default; manual prepack removed)

The manual `packed_linear` path is gone: frozen torch.compile is the only CPU
GEMM-story now, on both dtypes, and it is on by default
(`compile_decoder = CPU + backend=="vllm"` in `ModelRunner.__init__`; fp32
compiles too and verifies **token-exact vs HF**).  What built on §6.1:

- **Two or three graphs per runner**, all wrapped in
  `torch._inductor.config.patch(freezing=True)` contexts:
  `model.forward` = decoder + lm head (packs inside the graph —
  73 `_linear_pointwise`, 0 `aten.addmm`; dynamic B: one artifact for every
  batch size >= 2 plus one for batch 1, Dynamo's 0/1 specialization),
  `model.encode` (48 encoder GEMMs, dynamic over batch + encoder length), and
  for Florence-2 the vision tower + projector
  (`get_image_features`, its own **static** graph — see below).
- **Semantics preserved**: the CUDA decode path is untouched (`use_cudagraph`
  captures the eager decoder); a batch of separate LLM instances co-exists —
  now with the Dynamo budget to match. `encode` and the vision tower are bound
  methods, so they share **one** Dynamo frame (`external_utils.inner`): per
  engine up to 4 encode graphs (batch 1 / >1, with / without image) plus one
  tower graph per image shape, next to 2 on `forward`. `ModelRunner._compile`
  raises `torch._dynamo.config.recompile_limit` to `8 * engines`; past the
  limit Dynamo silently runs eager.
- **The compile is once per *engine* and once per *process*.** Frozen graphs
  bypass the Inductor FX graph cache (`fxgraph_cache_bypass` on every graph), so
  only the C++ kernels are reused: bart-large bf16 measured 19.9 s + 11.8 s
  (batch>=2 pair, then batch-1 pair) in a second process against 27.7 s + 13.9 s
  cold; budget ~10-25 s per graph (Inductor compile_threads capped at 4 so the
  parallel C++ workers don't spike GBs on small boxes).
- Benchmark table: the new dev-box baseline in `AGENTS.md` ("Benchmark
  baseline") records this state.  The per-shape memory bound measured above
  (0.74 GB bf16 / 1.6 GB fp32 per STATIC artifact for bart-large) is real; the
  decode and all text-encoder graphs are therefore `dynamic=True` (one artifact
  across all batch sizes >= 2 / encoder lengths).  Only the Florence-2 tower compiles
  static, because dynamic convs mis-stride DaViT (`assert_tensor_metadata`
  failures in the generated kernel); `_image_embeds` therefore always calls it
  with **one image**, otherwise every distinct number of cache misses is its own
  ~20 s / ~230 MB graph. Batching buys nothing anyway: at 16 and 32 images a
  static batched graph was 1.2-1.9x slower per image on a Xeon 8581C and
  1.1-1.2x slower on Axion (Florence-2 base and large; only Axion fp32 base was
  4% faster), with bit-identical output. The runner keeps the tower out of the text
  graph (it passes precomputed `image_embeds` into `encode`) so the two do not
  contaminate each other.
- fp32 token-exactness: on x86 the freezing pack on fp32 lowers to
  `mkl._mkl_linear` only in the static batch-1 graph; the dynamic batch>=2
  graphs (and everything on aarch64) use oneDNN `_linear_pointwise` — verified
  token-for-token equal vs HF on BART beams=1/4,
  Florence beams=1/3 (`verify_parity.py --dtype float32`).
- `verify_parity.py` now runs each (model, beams) phase in its own subprocess
  (fp32's two engines + reference + artifacts only fit an 11 GB box one phase
  at a time) and snaps fp32 KV budgets down (bf16: `2048*16` → fp32:
  `2048*4`).

Everything below (§6.1) stays true about *how* the custom ops make the graph
single-holed and the FX-cache rules — only its last two bullets (the
`exclude=` parameter, "compile is bf16-only") are superseded: `packed_linear`
was deleted entirely and fp32 compiles now too.

A third round (the "clean tensor API" refactor) moved the op bodies out of
`layers/attention.py` into `layers/cpu_ops.py` and replaced the layer registry
with per-step state the runner stages; §6.1's description of *those* two
mechanisms is historical, and the note at the top of the section says so.

### 6.3 Audit measurements (2026-11-12, torch 2.15)

bart-large-cnn, 8 threads, 256-token encoder input, 64 greedy steps; decode
ms/step. The laptop numbers are smoke-scale.

| box, dtype | batch | eager | compiled (default) | frozen batch 1 | max-autotune, frozen batch 1 |
|---|---|---:|---:|---:|---:|
| Axion bf16 | 8 | 17.7 | 11.7 | 11.7 | 11.3 |
| Axion bf16 | 1 | 7.1 | 6.4 (unfrozen) | 9.3 | 9.3 |
| Axion fp32 | 8 | 24.4 | 19.2 | 19.8 | 19.3 |
| Axion fp32 | 1 | 9.1 | 8.0 (unfrozen) | 12.0 | 8.0 |
| Zen 4 laptop bf16 | 8 | 21.1 | 15.8 | same | 15.6 |
| Zen 4 laptop bf16 | 1 | 13.9 | 13.3 | same | 10.3 |

- **Unfrozen graphs** (`ModelRunner._inductor_ctx` in
  `engine/model_runner.py`; the rule table is in `AGENTS.md` §4). A frozen graph
  keeps its own packed weights and batch 1 is its own graph, so a graph compiles
  with `freezing=False` (plain weights, no copy) wherever packing does not pay:
  fp32 encode at every batch size, fp32 batch-1 decode, and aarch64 bf16 batch-1
  decode. Measured 2026-11-13 on `c4-standard-8` (Xeon 8581C, 4 threads) and
  `c4a-standard-8` (Axion, 8 threads), frozen -> unfrozen, median of 3-5 runs:

  | fp32 | Xeon F2-large | Xeon bart-large | Axion F2-large | Axion bart-large |
  |---|---:|---:|---:|---:|
  | encode, batch 1 (ms) | 342 -> 346 | 132 -> 143 | 368 -> 342 | 134 -> 143 |
  | encode, batch 8 (ms) | 2820 -> 2921 | 1062 -> 1052 | 2620 -> 2648 | 1061 -> 1063 |
  | decode, batch 1 (ms/step) | 20.1 -> 19.2 | 19.6 -> 19.0 | unfrozen already | unfrozen already |
  | decode, batch 8 (ms/step) | 36.4 -> 57.6 | 31.7 -> 49.3 | 22.5 -> 27.5 | 19.7 -> 23.5 |
  | vision, 1 image (ms) | 2754 -> 3193 | - | 2487 -> 2587 | - |
  | vision, 8 images (ms) | 23144 -> 23014 | - | 19592 -> 19897 | - |

  So batch >= 2 decode and the tower stay frozen. RSS after warmup,
  Florence-2-large fp32 at batch 8, everything frozen -> encode + decode
  unfrozen -> tower unfrozen too: 7.6-8.3 -> 6.0-6.3 -> 4.8 GB on the Xeon,
  8.6 -> 6.3 -> 4.8 GB on Axion.

  x86 bf16 batch 1 stays frozen (+ max-autotune on decode), frozen + max-autotune
  -> unfrozen: Florence-2-large encode 73.2 -> 86.4 ms, decode 11.71 -> 13.28
  ms/step; bart-large-cnn encode 29.8 -> 40.7 ms, decode 11.21 -> 12.86 ms/step.
  Unfrozen + max-autotune and eager were no faster than unfrozen in either
  dtype. aarch64 bf16 encode unfrozen is not measured.

  fp32 stays token-exact vs HF (`verify_parity.py --dtype float32`, 12/12 on
  both boxes), and equal-length prompts batched together give the single-request
  tokens (the batch >= 2 encode graph; `verify_parity`'s prompts differ in
  length, so it only reaches batch-1 encode).
- **`mode="max-autotune"`** (Inductor's C++ GEMM template, autotuned per GEMM
  against the packed op) does not replace that: torch 2.15 has NEON/SVE
  micro-kernels for fp32 only, and the Axion bf16 run selected zero template
  kernels. On Axion fp32 it matches the unfrozen graph; on x86 bf16 it helps
  batch 1 only (-22% on the laptop). It costs 2-4x the compile time (61 s +
  52 s against 20 s + 12 s on the laptop) and picks kernels by timing them, so
  it is on only where it measured a clear win: the **decode graph on x86 bf16**
  (`ModelRunner._compile`). On 16-vCPU boxes with Florence-2-large
  (`c4-standard-16` Xeon 8581C, `c4a-standard-16` Axion), default → max-autotune:

  | stage | Xeon bf16 | Xeon fp32 | Axion bf16 | Axion fp32 |
  |---|---:|---:|---:|---:|
  | vision, 1 image (ms) | 339 → 369 | 1514 → 1485 | 443 → 444 | 1317 → 1286 |
  | encode, batch 8 (ms) | 494 → 445 | 1311 → 1559 | 448 → 450 | 1309 → 1315 |
  | encode, batch 1 (ms) | 45 → 45 | 166 → 167 | 62 → 63 | 168 → 168 |
  | decode, batch 8 (ms/step) | 12.35 → 10.91 | 22.66 → 23.94 | 11.96 → 12.38 | 15.89 → 13.91 |
  | decode, batch 1 (ms/step) | 8.57 → 6.52 | 12.44 → 11.73 | 5.23 → 5.39 | 6.68 → 6.64 |

  (Axion batch 1 is the unfrozen graph; frozen + max-autotune there is 9.49 bf16
  / 5.84 fp32.) fp32 stays token-exact vs HF under max-autotune on both boxes.
- **Memory**: every frozen graph keeps its own packed copy of the weights.
  bart-large bf16 (~0.8 GB of weights) grew RSS by ~1.0 GB for the batch>=2
  graphs and ~0.7 GB more for the batch-1 graphs on x86.

### 6.1 The landed state (compile_model=True, bf16 CPU) — second round

> **Historical.** The cross-attention *layer registry* this section describes
> (`register_cross_layer`, the weakref table, the int handle) has since been
> removed, and the op bodies no longer live in `layers/attention.py`. The
> current design: the ops are in `src/layers/cpu_ops.py`, the cross op
> takes `(query, layer_idx, scale)`, and the staged encoder K/V it reads comes
> from `Context.cross_staged`, which `ModelRunner.run` builds with
> `attention.stage_cross` *before* the forward — outside the graph.  The
> "register at `__init__` time" rule below still holds; the runner staging
> satisfies it in a stronger way (no state is read from traced code at all).

The remaining 53 pybind breaks were removed by wrapping the whole vLLM CPU
attention branch in two opaque `torch.library.custom_op`s
(`vflorence::cpu_vllm_self_attention`, `::cpu_vllm_cross_attention`,
registered in `layers/attention.py`), each with a `register_fake` that just
returns `torch.empty_like(query)`:

- the **op bodies run at call time**, not at trace time — they read the
  module-level `Context`, call `_self_derived` (per-step metadata memo),
  run `_cross_staged` (cross staging incl. the weakref layer registry),
  and mutate the paged caches through the pybind module — all invisible to
  Dynamo.  The self op declares `mutates_args={"k_cache", "v_cache"}`;
  profiling confirmed Inductor adds **no cache copies** for this (per-step
  `aten::copy_` count and time are identical to eager).
- cross-attention layers register at **`__init__` time**
  (`register_cross_layer` from `BartDecoderCrossAttention.__init__`); the int
  handle bakes into the graph eagerly.  Registration used to happen lazily in
  traced Python and produced an `IndexError` under a **cached replay**: the
  Inductor FX graph cache (`/tmp/torchinductor_*`) replays compiled artifacts
  across processes **without running the trace**, so trace-time side effects
  are skipped.  Rule: no state mutation inside traced code.
- `decoder_cross_attention` takes the (un-serializable) attention module only
  where the backend is known non-traceable or in eager paths; the custom op
  receives `(query, scale, handle, block_size)` — scalars only.
- `ModelRunner.__init__` gates the whole-decoder compile on
  bf16 + CPU + backend `"vllm"` and wraps each decoder call in
  `torch._inductor.config.patch(freezing=True)`; `_prepare` bucket-pads the
  block tables (16-block buckets) so table growth does not recompile.
- `packed_linear.apply` gains an `exclude=` parameter; with the compiled
  decoder it skips `model.decoder` — those 84 GEMMs/step are packed inside the
  graph by freezing instead (verified in the artifact: 84
  `mkldnn._linear_pointwise`, 0 `aten.addmm`).  The encoder, lm-head and every
  fp32 linear keep the manual prepack.

Measured on the dev CPU box (`256->256, batch=8`), same-day eager vs compiled
(benchmark.py `--compile`, one run each of a back-to-back pair):

| row | eager prepack | compiled | Δ |
|---|---:|---:|---:|
| batch=1, beams=1, bf16 | 3739 | 3375 | **−9%** |
| batch=8, beams=1, bf16 | 5318 | 5147 | −5% |
| batch=8, beams=4, bf16 | 8490 | 7865 | −6% |
| batch=8, beams=1, fp32 | 9247 | 9276 | = (compile gated off) |

The win comes from pointwise fusion: eager dispatches ~1.9 ms/step of Python
aten kernels (`aten::add` 2618, `native_layer_norm` 2393, `gelu` 780 per 64
steps); the compiled graph fuses those chains into its own C++ loops.  GEMM
output is equal (`_linear_pointwise` either way).  Greedy tokens are identical
to eager (the same kernels, same order — checked by `tests/test_attn_backend`
compiled parity and `benchmark.py --compile`'s same-token-count guard vs HF).

Remaining costs / next steps, in order of impact: per-distinct-batch-size
recompiles (~10-13 s for the first size; fixed-size offline batches are the
intended use — segment/finetune bucket pins like `DecodeGraphRunner` if this
ever needs to be the default), and the still-eager `compute_logits` (one large
GEMM; compiling it measured nothing in a toy probe).

## 7. Repro

Minimal toy check (all signals in-process; ~10 s):

```python
import torch

torch._inductor.config.freezing = True  # cpp.weight_prepack is already True
from torch._dynamo.utils import counters

model = torch.nn.Sequential(
    *[
        m
        for _ in range(12)
        for m in (
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        )
    ]
).to(torch.bfloat16)  # no autocast needed
model = torch.compile(model)

with torch.no_grad():  # must wrap the FIRST call
    model(torch.randn(8, 1024, dtype=torch.bfloat16))

print(
    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"]
)  # 24 = every GEMM
# to see which packed op was emitted (per §2):
#   $ TORCH_LOGS=+output_code python this_script.py 2>&1 | grep -E "_linear_pointwise|_mkl_linear"
```

The original experiment scripts were not kept; expected results are in §4 and §6.3.

- Counters meanings: `mkldnn_linear_weight_pack_matcher_count` = one per
  `aten.mm`/`aten.addmm` node rewritten into
  `mkldnn._reorder_linear_weight` (+ bias) + `mkldnn._linear_pointwise`
  / `mkl._mkl_linear`.
- Emitted code on disk: `TORCH_COMPILE_DEBUG=1` writes
  `torch_compile_debug/run_*/torchinductor/model__*/output_code.py`
  (grep `_linear_pointwise`, `_frozen_param`, `extern_kernels.addmm`).
- Runtime op counts: `torch.profiler` — `mkldnn::_linear_pointwise` vs
  `aten::addmm`/`aten::linear`.


## 8. References

- PyTorch blog, *Accelerated CPU Inference with PyTorch Inductor* — the
  `weight_prepack`/oneDNN `_linear_pointwise` design and (at the time)
  `mkldnn_fuse_fx`: https://pytorch.org/blog/accelerated-cpu-inference/
- Intel version of the same post: https://www.intel.com/content/www/us/en/developer/articles/technical/accelerated-cpu-inference-with-pytorch-inductor.html
- PyTorch blog, *Accelerated PyTorch inference … on AWS Graviton* — recipe
  `config.cpp.weight_prepack=True; config.freezing=True` under
  `torch.no_grad()`, T5 profile showing `mkldnn::_linear_pointwise`, and
  `DNNL_DEFAULT_FPMATH_MODE=BF16`:
  https://pytorch.org/blog/accelerated-pytorch-inference/
- PyTorch tutorial, *Using Max-Autotune Compilation on CPU* — "only support
  frozen model with `torch.no_grad` or the inference mode … set
  `TORCHINDUCTOR_FREEZING=1` and ensure both compile and inference steps are
  executed within `torch.no_grad`", example wrapped in
  `torch.no_grad(), torch.cpu.amp.autocast(bf16)`:
  https://docs.pytorch.org/tutorials/unstable/max_autotune_on_CPU_tutorial.html
- Source (line numbers from the torch 2.15 wheel):
  - `torch/_inductor/config.py` (1821: `freezing` default False; 1971:
    `cpp.weight_prepack` default True)
  - `torch/_inductor/compile_fx.py:3414` — the `freezing and not grad_enabled`
    gate; `:2516` `fw_compiler_freezing`
  - `torch/_inductor/freezing.py:77-147` — `freeze`/`_freeze`, params→constants
  - `torch/_inductor/fx_passes/freezing_patterns.py:37-102` — the pass
    registration home of weight packing
  - `torch/_inductor/fx_passes/mkldnn_fusion.py` (109-159 op choice; 1288-1370
    `_is_packable_linear`; 1403 `_register_weight_pack_pass`, the `addmm`/`mm` patterns)
  - `torch/_dynamo/config.py:121` — `recompile_limit = 8`
