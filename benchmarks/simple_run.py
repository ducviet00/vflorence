"""Per-stage timing on one batch: vision encoder, language encoder, decode step.

One backend per run (``--backend fast`` or ``--backend hf``); run twice and
compare.  Works for Florence-2 (all three stages) and BART (no vision stage).

* vision  -- DaViT + projector on ``--batch`` images (Florence-2 only)
* encode  -- BART encoder (Florence-2: image tokens spliced into the prompt)
* decode  -- one decoder step for the whole batch, measured as
             (generate N tokens - generate 1 token) / (N - 1) with EOS suppressed,
             so it includes the per-step logits processing / beam bookkeeping

  PYTHONPATH=. python benchmarks/simple_run.py --backend fast
  PYTHONPATH=. python benchmarks/simple_run.py --backend hf --model facebook/bart-large-cnn --enc-len 512
"""

import argparse

import torch

from benchmarks.utils import (
    FLORENCE,
    HF_DECODE,
    IMAGES,
    bart_prompt,
    blocks_for,
    load_image,
    timed,
)


def make_inputs(args, florence: bool):
    """Processor/tokenizer outputs for ``--batch`` identical-length requests."""
    from transformers import AutoProcessor, AutoTokenizer

    if florence:
        processor = AutoProcessor.from_pretrained(args.model)
        images = [load_image(IMAGES[i % len(IMAGES)][0]) for i in range(args.batch)]
        return processor(
            text=[args.prompt] * args.batch, images=images, return_tensors="pt"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = bart_prompt(tokenizer, args.enc_len - 2)  # + BOS/EOS
    return tokenizer([text] * args.batch, return_tensors="pt")


def fast_stages(args, inputs, florence, dtype):
    """Return {stage: callable} for fast-encdec."""
    from fastencdec import LLM, SamplingParams
    from fastencdec.config import default_block_size

    block_size = default_block_size()
    seqs = args.batch * args.beams
    llm = LLM(
        args.model,
        num_blocks=blocks_for(seqs, args.max_tokens, block_size),
        block_size=block_size,
        max_num_seqs=seqs,
        dtype=dtype,
    )
    runner = llm.engine.runner
    ids = inputs["input_ids"]
    px = inputs["pixel_values"].to(dtype) if florence else None

    @torch.no_grad()
    def vision():
        with runner._inductor_ctx():
            return runner.image_fn(px)

    image_embeds = vision() if florence else None

    @torch.no_grad()
    def encode():
        with runner._inductor_ctx():
            if florence:
                return runner.encode_fn(ids, image_embeds=image_embeds)
            return runner.encode_fn(ids)

    requests = [
        (row.tolist(), [llm.decoder_start_token_id], px[i] if florence else None)
        for i, row in enumerate(ids)
    ]

    def generate(tokens):
        # min_length > max_tokens masks EOS for the whole run -> exactly ``tokens``
        params = SamplingParams(
            max_tokens=tokens, num_beams=args.beams, min_length=tokens + 1
        )
        return lambda: llm.engine.generate(requests, params)

    stages = {"encode": encode}
    if florence:
        stages = {"vision": vision, **stages}
    return stages, generate


def hf_stages(args, inputs, florence, dtype):
    """Return {stage: callable} for HuggingFace transformers."""
    from transformers import AutoModelForSeq2SeqLM, Florence2ForConditionalGeneration

    cls = Florence2ForConditionalGeneration if florence else AutoModelForSeq2SeqLM
    model = cls.from_pretrained(args.model, dtype=dtype).eval()
    ids = inputs["input_ids"]
    px = inputs["pixel_values"].to(dtype) if florence else None

    @torch.no_grad()
    def vision():
        return model.model.get_image_features(px).pooler_output

    image_embeds = vision() if florence else None

    @torch.no_grad()
    def encode():
        if not florence:
            return model.model.encoder(input_ids=ids)
        # mirrors Florence2Model.forward: splice image features into prompt embeddings
        embeds = model.model.get_input_embeddings()(ids)
        mask = model.model.get_placeholder_mask(ids, embeds, image_embeds)
        embeds = embeds.masked_scatter(mask, image_embeds.to(embeds.dtype))
        return model.model.language_model.encoder(inputs_embeds=embeds)

    def generate(tokens):
        def run():
            with torch.no_grad():
                return model.generate(
                    **inputs,
                    max_new_tokens=tokens,
                    min_new_tokens=tokens,  # EOS suppressed -> exactly ``tokens``
                    num_beams=args.beams,
                    **HF_DECODE,
                )

        return run

    stages = {"encode": encode}
    if florence:
        stages = {"vision": vision, **stages}
    return stages, generate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["fast", "hf"])
    ap.add_argument("--model", default=FLORENCE)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beams", type=int, default=1, help="1 = greedy")
    ap.add_argument("--max-tokens", type=int, default=128, help="decode steps timed")
    ap.add_argument("--prompt", default="<DETAILED_CAPTION>", help="Florence-2 prompt")
    ap.add_argument("--enc-len", type=int, default=512, help="BART encoder tokens")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if (
        min(args.batch, args.beams, args.warmup + args.repeats) < 1
        or args.max_tokens < 2
    ):
        ap.error("need --batch/--beams >= 1, --max-tokens >= 2, warmup+repeats >= 1")

    import transformers
    from transformers import AutoConfig

    transformers.logging.set_verbosity_error()
    florence = AutoConfig.from_pretrained(args.model).model_type == "florence2"
    dtype = getattr(torch, args.dtype)
    inputs = make_inputs(args, florence)
    build = fast_stages if args.backend == "fast" else hf_stages
    stages, generate = build(args, inputs, florence, dtype)

    results = {
        name: timed(fn, args.warmup, args.repeats)[0] for name, fn in stages.items()
    }
    first, _ = timed(generate(1), args.warmup, args.repeats)
    full, _ = timed(generate(args.max_tokens), args.warmup, args.repeats)
    step = (full - first) / (args.max_tokens - 1)

    print(
        f"\nbackend={args.backend}  model={args.model}  dtype={args.dtype}  "
        f"threads={torch.get_num_threads()}  batch={args.batch}  beams={args.beams}  "
        f"enc_len={inputs['input_ids'].shape[1]}  max_tokens={args.max_tokens}\n"
    )
    print(f"{'stage':<26}{'ms':>10}")
    for name, seconds in results.items():
        print(f"{name:<26}{seconds * 1e3:>10.1f}")
    print(f"{'decode step':<26}{step * 1e3:>10.2f}")
    print(f"{'generate 1 token':<26}{first * 1e3:>10.1f}")
    print(f"{f'generate {args.max_tokens} tokens':<26}{full * 1e3:>10.1f}")
    print(
        f"\ndecode: {args.batch / step:.1f} tok/s  "
        f"(batch {args.batch} x beams {args.beams} sequences per step)"
    )


if __name__ == "__main__":
    main()
