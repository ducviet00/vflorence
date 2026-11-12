"""Per-stage timing on one batch: vision encoder, language encoder, decode step.

One backend per run (``--backend fast`` or ``--backend hf``); run twice and
compare.  Works for Florence-2 (all three stages) and BART (no vision stage).

* vision  -- DaViT + projector on ``--batch`` images (Florence-2 only)
* encode  -- BART encoder (Florence-2: image tokens spliced into the prompt)
* decode  -- one decoder step for the whole batch, measured as
             (generate N tokens - generate 1 token) / (N - 1) with EOS suppressed,
             so it includes the per-step logits processing / beam bookkeeping

  PYTHONPATH=. python benchmarks/stage_latency.py --backend fast
  PYTHONPATH=. python benchmarks/stage_latency.py --backend hf --model facebook/bart-large-cnn --enc-len 512
"""

import argparse

import torch

from benchmarks.utils import load_image, timed

PARAGRAPH = (
    "Scientists at the university announced on Monday that they had discovered a "
    "new species of butterfly in the Amazon rainforest. The insect has bright blue "
    "wings and a wingspan of nearly ten centimetres. The city council approved the "
    "new transit plan after a long debate. The plan adds two tram lines, extends "
    "the bus network to the outer districts and introduces a single ticket for all "
    "modes of transport from next year."
)


def prepare_inputs(args, florence, dtype):
    """Prepare token IDs and optional image pixels before any stage is timed."""
    from transformers import AutoProcessor, AutoTokenizer

    if florence:
        processor = AutoProcessor.from_pretrained(args.model)
        image_urls = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "http://images.cocodataset.org/val2017/000000000632.jpg",
            "http://images.cocodataset.org/val2017/000000000724.jpg",
            "http://images.cocodataset.org/val2017/000000000776.jpg",
            "http://images.cocodataset.org/val2017/000000000785.jpg",
            "http://images.cocodataset.org/val2017/000000000802.jpg",
        ]
        images = [
            load_image(image_urls[i % len(image_urls)]) for i in range(args.batch)
        ]
        inputs = processor(
            text=[args.prompt] * args.batch, images=images, return_tensors="pt"
        )
        return inputs["input_ids"], inputs["pixel_values"].to(dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Repeat token IDs directly, reserving space for BART's BOS and EOS tokens.
    paragraph_ids = tokenizer(PARAGRAPH, add_special_tokens=False)["input_ids"]
    content_length = args.enc_len - 2
    copies = (content_length + len(paragraph_ids) - 1) // len(paragraph_ids)
    token_ids = [
        tokenizer.bos_token_id,
        *(paragraph_ids * copies)[:content_length],
        tokenizer.eos_token_id,
    ]
    return torch.tensor([token_ids] * args.batch), None


def fast_stages(args, input_token_ids, pixels, dtype):
    """Return {stage: callable} for vflorence."""
    from vflorence import LLM, Request, SamplingParams

    llm = LLM(
        args.model,
        max_num_seqs=args.batch * args.beams,
        max_decoder_len=args.max_tokens,
        max_encoder_len=input_token_ids.shape[1],
        dtype=dtype,
        mm_cache_size=0,  # stages time the vision encoder on every call
    )
    runner = llm.runner

    @torch.no_grad()
    def vision():
        with runner._inductor_ctx():
            return runner.image_fn(pixels)

    image_embeds = vision() if pixels is not None else None

    @torch.no_grad()
    def encode():
        with runner._inductor_ctx():
            if pixels is not None:
                return runner.encode_fn(input_token_ids, image_embeds=image_embeds)
            return runner.encode_fn(input_token_ids)

    requests = [
        Request(row.tolist(), pixels[i] if pixels is not None else None)
        for i, row in enumerate(input_token_ids)
    ]

    @torch.no_grad()
    def generate(tokens):
        # min_length > max_tokens masks EOS for the whole run -> exactly ``tokens``
        params = SamplingParams(
            max_tokens=tokens, num_beams=args.beams, min_length=tokens + 1
        )
        return llm.generate(requests, params, use_tqdm=args.tqdm)

    stages = {"encode": encode}
    if pixels is not None:
        stages = {"vision": vision, **stages}
    return stages, generate


def hf_stages(args, input_token_ids, pixels, dtype):
    """Return {stage: callable} for HuggingFace transformers."""
    from transformers import AutoModelForSeq2SeqLM, Florence2ForConditionalGeneration

    model_class = (
        Florence2ForConditionalGeneration
        if pixels is not None
        else AutoModelForSeq2SeqLM
    )
    model = model_class.from_pretrained(args.model, dtype=dtype).eval()
    inputs = {"input_ids": input_token_ids}
    if pixels is not None:
        inputs["pixel_values"] = pixels

    @torch.no_grad()
    def vision():
        return model.model.get_image_features(pixels).pooler_output

    image_embeds = vision() if pixels is not None else None

    @torch.no_grad()
    def encode():
        if pixels is None:
            return model.model.encoder(input_ids=input_token_ids)
        # mirrors Florence2Model.forward: splice image features into prompt embeddings
        embeds = model.model.get_input_embeddings()(input_token_ids)
        mask = model.model.get_placeholder_mask(input_token_ids, embeds, image_embeds)
        embeds = embeds.masked_scatter(mask, image_embeds.to(embeds.dtype))
        return model.model.language_model.encoder(inputs_embeds=embeds)

    @torch.no_grad()
    def generate(tokens):
        return model.generate(
            **inputs,
            max_new_tokens=tokens,
            min_new_tokens=tokens,  # Suppress EOS so every request generates exactly N tokens.
            num_beams=args.beams,
            # Everything else is the checkpoint's generation config, as in vflorence.
            do_sample=False,
        )

    stages = {"encode": encode}
    if pixels is not None:
        stages = {"vision": vision, **stages}
    return stages, generate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["fast", "hf"])
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beams", type=int, default=1, help="1 = greedy")
    ap.add_argument("--max-tokens", type=int, default=128, help="decode steps timed")
    ap.add_argument("--prompt", default="<DETAILED_CAPTION>", help="Florence-2 prompt")
    ap.add_argument("--enc-len", type=int, default=512, help="BART encoder tokens")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--tqdm", action="store_true", help="show the fast backend progress bar"
    )
    args = ap.parse_args()
    if (
        min(args.batch, args.beams, args.repeats) < 1
        or args.warmup < 0
        or args.enc_len < 2
        or args.max_tokens < 2
    ):
        ap.error(
            "need --batch/--beams/--repeats >= 1, --max-tokens/--enc-len >= 2, "
            "--warmup >= 0"
        )

    import transformers
    from transformers import AutoConfig

    transformers.logging.set_verbosity_error()
    florence = AutoConfig.from_pretrained(args.model).model_type == "florence2"
    dtype = getattr(torch, args.dtype)
    input_token_ids, pixels = prepare_inputs(args, florence, dtype)
    if args.backend == "fast":
        stages, generate = fast_stages(args, input_token_ids, pixels, dtype)
    else:
        stages, generate = hf_stages(args, input_token_ids, pixels, dtype)

    results = {
        name: timed(fn, args.warmup, args.repeats)[0] for name, fn in stages.items()
    }
    first, _ = timed(lambda: generate(1), args.warmup, args.repeats)
    full, _ = timed(lambda: generate(args.max_tokens), args.warmup, args.repeats)
    step = (full - first) / (args.max_tokens - 1)

    print(
        f"\nbackend={args.backend}  model={args.model}  dtype={args.dtype}  "
        f"threads={torch.get_num_threads()}  batch={args.batch}  beams={args.beams}  "
        f"enc_len={input_token_ids.shape[1]}  max_tokens={args.max_tokens}\n"
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
