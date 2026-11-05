"""BART benchmark: this engine vs vLLM on CPU, scenarios x configs.

Methodology mirrors ``benchmarks/benchmark.py``: both engines run the same
dtype, prompts and output length, EOS suppressed, so the speedup is
engine-vs-engine.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from functools import partial

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark import CONFIGS, SCENARIOS, make_prompt

MODEL = "facebook/bart-large-cnn"


def timed(fn, warmup: int, repeats: int):
    """Return (best_seconds, per-repeat token counts)."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    toks = []
    for _ in range(repeats):
        start = time.perf_counter()
        toks.append(fn())
        best = min(best, time.perf_counter() - start)
    return best, toks


def dump(args, rows) -> None:
    """Write raw rows now: a later OOM must not lose the whole run."""
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


def blocks_for(configs, enc_tokens: int, out_tokens: int, block_size: int) -> int:
    """``num_blocks`` covering the widest config of one scenario."""
    worst = max(batch * beams for _, batch, beams, _ in configs)
    tokens = worst * (enc_tokens + out_tokens + 8)
    return (tokens + block_size - 1) // block_size + 8


def ours_run(llm, prompts, params) -> int:
    """One generate; returns the number of tokens the engine decoded."""
    outs = llm.generate(prompts, params)
    return sum(len(o.outputs[0].token_ids) for o in outs)


def vllm_run(llm, prompts, params) -> int:
    """One vLLM generate; returns the number of tokens decoded."""
    outs = llm.generate(prompts, params)
    return sum(len(o.outputs[0].token_ids) for o in outs)


def run_ours(args, tokenizer, scenarios):
    from fastencdec import LLM, SamplingParams
    from fastencdec.config import default_block_size

    block_size = default_block_size()
    max_seqs = max(b * n for _, b, n, _ in args.configs)
    rows = []
    for dtype_name in ("bfloat16", "float32"):
        configs = [c for c in args.configs if c[3] == dtype_name]
        if not configs:
            continue
        for name, enc_tokens, out_tokens in scenarios:
            prompt = make_prompt(tokenizer, enc_tokens)
            enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            llm = LLM(
                args.model,
                num_blocks=blocks_for(configs, enc_len, out_tokens, block_size),
                block_size=block_size,
                max_num_seqs=max_seqs,
                dtype=getattr(torch, dtype_name),
            )
            for label, batch, beams, _ in configs:
                params = SamplingParams(
                    max_tokens=out_tokens, num_beams=beams, min_length=out_tokens + 1
                )
                dt, toks = timed(
                    partial(ours_run, llm, [prompt] * batch, params),
                    args.warmup,
                    args.repeats,
                )
                assert len(set(toks)) == 1, f"{name}/{label}: ours decoded {set(toks)}"
                expect = batch * out_tokens
                assert toks[0] == expect, (
                    f"{name}/{label}: ours {toks[0]}, want {expect}"
                )
                row = {
                    "scenario": name,
                    "config": label,
                    "dtype": dtype_name,
                    "batch": batch,
                    "beams": beams,
                    "enc_tokens": enc_len,
                    "out_tokens": out_tokens,
                    "ms": dt * 1e3,
                    "tok": toks[0],
                    "tok_s": toks[0] / dt,
                }
                rows.append(row)
                print(
                    f"ours {name} {label}: {dt * 1e3:.0f} ms, {row['tok_s']:.1f} tok/s",
                    flush=True,
                )
                dump(args, rows)
            del llm
            torch._dynamo.reset()
    return rows


def run_vllm(args, tokenizer, scenarios):
    use_beam = any(beams > 1 for _, _, beams, _ in args.configs)
    if use_beam:
        raise SystemExit(
            "vLLM cannot run BART beam search on CPU: BeamSearchScheduler needs "
            "the V2 model runner with async scheduling, which vLLM gates on "
            "Triton.  Drop the beams>1 configs."
        )
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_PLUGINS", "bart")
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "8")
    import vllm_bart_plugin  # noqa: F401  registers BART with vLLM
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "max_model_len": 1024,
        "max_num_seqs": max(b * n for _, b, n, _ in args.configs),
    }

    rows = []
    for dtype_name in ("bfloat16", "float32"):
        configs = [c for c in args.configs if c[3] == dtype_name]
        if not configs:
            continue
        llm = LLM(dtype=dtype_name, **llm_kwargs)
        for name, enc_tokens, out_tokens in scenarios:
            prompt = make_prompt(tokenizer, enc_tokens)
            enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for label, batch, beams, _ in configs:
                prompts = [
                    {
                        "encoder_prompt": {
                            "prompt": "",
                            "multi_modal_data": {"text": prompt},
                        },
                        "decoder_prompt": "<s>",
                    }
                ] * batch
                params = SamplingParams(
                    temperature=0.0,
                    max_tokens=out_tokens,
                    min_tokens=out_tokens,
                )
                dt, toks = timed(
                    partial(vllm_run, llm, prompts, params), args.warmup, args.repeats
                )
                assert len(set(toks)) == 1, f"{name}/{label}: vllm decoded {set(toks)}"
                lo, hi = batch * out_tokens, batch * (out_tokens + 1)
                assert lo <= toks[0] <= hi, (
                    f"{name}/{label}: vllm {toks[0]} not in [{lo},{hi}]"
                )
                row = {
                    "scenario": name,
                    "config": label,
                    "dtype": dtype_name,
                    "batch": batch,
                    "beams": beams,
                    "enc_tokens": enc_len,
                    "out_tokens": out_tokens,
                    "ms": dt * 1e3,
                    "tok": toks[0],
                    "tok_s": toks[0] / dt,
                }
                rows.append(row)
                print(
                    f"vllm {name} {label}: {dt * 1e3:.0f} ms, {row['tok_s']:.1f} tok/s",
                    flush=True,
                )
                dump(args, rows)
        del llm
        gc.collect()
    return rows


def compare(ours_path: str, vllm_path: str) -> None:
    with open(ours_path) as f:
        ours = {(r["scenario"], r["config"]): r for r in json.load(f)}
    with open(vllm_path) as f:
        vllm = {(r["scenario"], r["config"]): r for r in json.load(f)}

    def order(row):
        batch = row.get("batch") or int(
            re.search(r"batch=(\d+)", row["config"]).group(1)
        )
        beams = row.get("beams", 1)
        return (batch, beams, 1 if row.get("dtype") == "float32" else 0)

    for name, enc_tokens, out_tokens in SCENARIOS:
        keys = sorted(
            (k for k in ours if k[0] == name and k in vllm),
            key=lambda k: order(ours[k]),
        )
        if not keys:
            continue
        print(f"\n### {name}: encoder {enc_tokens} -> {out_tokens} generated\n")
        print("| config | ours ms | ours tok/s | vllm ms | vllm tok/s | speedup |")
        print("|---|---:|---:|---:|---:|---:|")
        for k in keys:
            o, v = ours[k], vllm[k]
            print(
                f"| {k[1]:<22} | {o['ms']:>8.0f} | {o['tok_s']:>10.1f} "
                f"| {v['ms']:>7.0f} | {v['tok_s']:>10.1f} "
                f"| {v['ms'] / o['ms']:>7.2f}x |"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--engine", choices=("ours", "vllm"))
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="0 = torch default, i.e. the number of physical cores",
    )
    ap.add_argument("--scenarios", default="", help="comma list, default all")
    ap.add_argument(
        "--configs",
        default="",
        help="semicolon list (labels contain commas), default all",
    )
    ap.add_argument(
        "--batches",
        default="",
        help="greedy (beams=1) batch sweep, comma list; overrides --configs",
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default=None, help="write raw results to a JSON file")
    ap.add_argument("--compare", nargs=2, metavar=("OURS", "VLLM"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.engine:
        ap.error("--engine is required (or use --compare)")

    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def select(spec, all_names, label, sep=","):
        wanted = {s.strip() for s in spec.split(sep) if s.strip()}
        bad = wanted - set(all_names)
        if bad:
            raise SystemExit(
                f"unknown {label}(s) {sorted(bad)}; choose from {all_names}"
            )
        return wanted

    scen_wanted = select(args.scenarios, [s[0] for s in SCENARIOS], "scenario")
    scenarios = [s for s in SCENARIOS if not scen_wanted or s[0] in scen_wanted]

    if args.batches:
        batches = [int(b) for b in args.batches.split(",") if b.strip()]
        args.configs = [
            (f"batch={b}, beams=1, bf16", b, 1, "bfloat16") for b in batches
        ]
    else:
        cfg_wanted = select(args.configs, [c[0] for c in CONFIGS], "config", sep=";")
        args.configs = [c for c in CONFIGS if not cfg_wanted or c[0] in cfg_wanted]

    rows = (
        run_ours(args, tokenizer, scenarios)
        if args.engine == "ours"
        else run_vllm(args, tokenizer, scenarios)
    )
    if args.json:
        dump(args, rows)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
