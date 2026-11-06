"""Continuous batching vs padded batches: Florence-2, 8 real images x 4 tasks.

Requests mix long tasks (<OD>, segmentation) with short ones (<OCR>,
<DETAILED_CAPTION>). ``--backend fast`` keeps ``--max-num-seq`` requests in
flight and refills a slot as soon as a request finishes; ``--backend hf`` runs
the same requests as back-to-back padded batches of ``--max-num-seq``. One
backend/config per run -- run twice and compare.

  PYTHONPATH=. python benchmarks/continuous_batching.py --backend fast
  PYTHONPATH=. python benchmarks/continuous_batching.py --backend hf --beamsearch 4
"""

import argparse

import torch

from benchmarks.utils import load_image, timed

IMAGE_URLs = [  # (image URL, referring expression)
    ("http://images.cocodataset.org/val2017/000000039769.jpg", "the cat on the left"),
    ("http://images.cocodataset.org/val2017/000000000139.jpg", "the television"),
    ("http://images.cocodataset.org/val2017/000000000285.jpg", "the bear"),
    ("http://images.cocodataset.org/val2017/000000000632.jpg", "the bed"),
    ("http://images.cocodataset.org/val2017/000000000724.jpg", "the stop sign"),
    ("http://images.cocodataset.org/val2017/000000000776.jpg", "the teddy bear"),
    ("http://images.cocodataset.org/val2017/000000000785.jpg", "the skier"),
    ("http://images.cocodataset.org/val2017/000000000802.jpg", "the oven"),
]

TASKS = ["<OD>", "<OCR>", "<DETAILED_CAPTION>", "<REFERRING_EXPRESSION_SEGMENTATION>"]


def prepare_inputs(processor, num_requests):
    """Cycle through four tasks per image; process each request before timing."""
    images = {}
    labels, input_token_ids, pixels = [], [], []
    for i in range(num_requests):
        url, phrase = IMAGE_URLs[(i // len(TASKS)) % len(IMAGE_URLs)]
        task = TASKS[i % len(TASKS)]
        prompt = (
            task + phrase if task == "<REFERRING_EXPRESSION_SEGMENTATION>" else task
        )
        if url not in images:
            images[url] = load_image(url)
        inputs = processor(text=prompt, images=images[url], return_tensors="pt")
        labels.append((task, url.rsplit("/", 1)[-1]))
        input_token_ids.append(inputs["input_ids"][0].tolist())
        pixels.append(inputs["pixel_values"][0])
    return labels, input_token_ids, pixels


def run_fast(args, input_token_ids, pixels, dtype):
    from fastencdec import LLM, SamplingParams
    from fastencdec.config import default_block_size

    block_size = default_block_size()
    num_sequences = args.max_num_seq * args.beamsearch
    # Include the decoder start token and one extra block per beam for copy-on-write.
    blocks_per_sequence = (args.max_tokens + 1 + block_size - 1) // block_size + 1
    llm = LLM(
        args.model,
        num_blocks=num_sequences * blocks_per_sequence,
        block_size=block_size,
        max_num_seqs=num_sequences,
        dtype=dtype,
    )
    params = SamplingParams(
        max_tokens=args.max_tokens,
        num_beams=args.beamsearch,
        eos_token_id=llm.eos_token_id,
    )
    requests = [
        (ids, [llm.decoder_start_token_id], image_pixels)
        for ids, image_pixels in zip(input_token_ids, pixels)
    ]

    def once():
        return list(llm.engine.generate(requests, params).values())

    seconds, rows = timed(once, args.warmup, args.repeats)
    return seconds, rows, [0] * len(rows)


def run_hf(args, input_token_ids, pixels, dtype, tokenizer):
    from transformers import Florence2ForConditionalGeneration

    model = Florence2ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype
    ).eval()
    batches = []
    for start in range(0, len(input_token_ids), args.max_num_seq):
        end = start + args.max_num_seq
        batch = tokenizer.pad(
            {"input_ids": input_token_ids[start:end]}, padding=True, return_tensors="pt"
        )
        batch["pixel_values"] = torch.stack(pixels[start:end]).to(dtype)
        batches.append(batch)

    @torch.no_grad()
    def once():
        return [
            model.generate(
                **batch,
                max_new_tokens=args.max_tokens,
                num_beams=args.beamsearch,
                # Override checkpoint defaults to match fast-encdec's decoding.
                do_sample=False,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                early_stopping=False,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
            )
            for batch in batches
        ]

    seconds, outputs = timed(once, args.warmup, args.repeats)
    rows, padding = [], []
    for output in outputs:
        for row in output.tolist():
            tokens = row[
                1:
            ]  # HF includes the decoder start token; fast-encdec does not.
            length = len(tokens)
            while tokens and tokens[-1] == tokenizer.pad_token_id:
                tokens.pop()
            rows.append(tokens)
            padding.append(length - len(tokens))
    return seconds, rows, padding


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["fast", "hf"])
    ap.add_argument("--num-req", type=int, default=32)
    ap.add_argument(
        "--max-num-seq",
        type=int,
        default=8,
        help="requests in flight (fast) / padded batch size (hf)",
    )
    ap.add_argument("--beamsearch", type=int, default=1, help="1 = greedy")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    args = ap.parse_args()
    if args.num_req < 1 or args.max_num_seq < 1 or args.beamsearch < 1:
        ap.error("--num-req, --max-num-seq and --beamsearch must be >= 1")

    if args.max_tokens < 1 or args.warmup < 0 or args.repeats < 1:
        ap.error("need --max-tokens >= 1, --warmup >= 0 and --repeats >= 1")

    import transformers
    from transformers import AutoProcessor

    transformers.logging.set_verbosity_error()
    dtype = getattr(torch, args.dtype)

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    labels, input_token_ids, pixels = prepare_inputs(processor, args.num_req)
    if args.backend == "fast":
        seconds, rows, pads = run_fast(args, input_token_ids, pixels, dtype)
    else:
        seconds, rows, pads = run_hf(args, input_token_ids, pixels, dtype, tokenizer)

    print(
        f"\nbackend={args.backend}  model={args.model}  dtype={args.dtype}  "
        f"threads={torch.get_num_threads()}  num_req={args.num_req}  "
        f"max_num_seq={args.max_num_seq}  beams={args.beamsearch}  "
        f"max_tokens={args.max_tokens}\n"
    )
    for i, ((task, cid), toks) in enumerate(zip(labels, rows)):
        text = tokenizer.decode(toks).replace("\n", " ")
        print(f"#{i:<3} {task:<38} {cid}  {len(toks):>4} tok  {text[:60]}")

    print(f"\n{'task':<38}{'reqs':>6}{'gen tok':>9}{'mean':>8}{'pad tok':>9}")
    for task in TASKS:
        idx = [i for i, w in enumerate(labels) if w[0] == task]
        if not idx:
            continue
        gen = sum(len(rows[i]) for i in idx)
        pad = sum(pads[i] for i in idx)
        print(f"{task:<38}{len(idx):>6}{gen:>9}{gen / len(idx):>8.1f}{pad:>9}")

    gen, pad = sum(len(r) for r in rows), sum(pads)
    print(
        f"\ntotal: {seconds * 1e3:.0f} ms  {gen} gen tok  "
        f"{gen / seconds:.1f} tok/s  {len(rows) / seconds:.2f} req/s"
        + (f"  padding {pad / (gen + pad):.1%} of decoded positions" if pad else "")
    )


if __name__ == "__main__":
    main()
