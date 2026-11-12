"""Continuous batching vs padded batches: Florence-2, 32 real images x 4 tasks.

Requests are shuffled once (seed 42) and mix long tasks (<OD>, segmentation)
with short ones (<CAPTION>, <DETAILED_CAPTION>). ``--backend fast`` keeps
``--max-num-req`` requests in flight and refills a slot as soon as a request
finishes; ``--backend hf`` runs the same requests as back-to-back padded
batches of ``--max-num-req``. One backend/config per run -- run twice and compare.

  PYTHONPATH=. python benchmarks/continuous_batching.py --backend fast
  PYTHONPATH=. python benchmarks/continuous_batching.py --backend hf --beamsearch 4
"""

import argparse
import math
import platform
import random

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
    ("http://images.cocodataset.org/val2017/000000000872.jpg", "the baseball glove"),
    ("http://images.cocodataset.org/val2017/000000000885.jpg", "the tennis racket"),
    ("http://images.cocodataset.org/val2017/000000001000.jpg", "the boy in red"),
    ("http://images.cocodataset.org/val2017/000000001296.jpg", "the phone"),
    ("http://images.cocodataset.org/val2017/000000001353.jpg", "the red train car"),
    ("http://images.cocodataset.org/val2017/000000001490.jpg", "the paddle"),
    ("http://images.cocodataset.org/val2017/000000001503.jpg", "the keyboard"),
    ("http://images.cocodataset.org/val2017/000000001532.jpg", "the green road sign"),
    ("http://images.cocodataset.org/val2017/000000001584.jpg", "the red bus"),
    ("http://images.cocodataset.org/val2017/000000001675.jpg", "the cat"),
    ("http://images.cocodataset.org/val2017/000000001761.jpg", "the bridge"),
    ("http://images.cocodataset.org/val2017/000000001818.jpg", "the zebra"),
    ("http://images.cocodataset.org/val2017/000000001993.jpg", "the bed"),
    ("http://images.cocodataset.org/val2017/000000002006.jpg", "the purple bus"),
    ("http://images.cocodataset.org/val2017/000000002149.jpg", "the right apple"),
    ("http://images.cocodataset.org/val2017/000000002153.jpg", "the batter"),
    ("http://images.cocodataset.org/val2017/000000002157.jpg", "the cake"),
    ("http://images.cocodataset.org/val2017/000000002261.jpg", "the surfer"),
    ("http://images.cocodataset.org/val2017/000000002431.jpg", "the wine glass"),
    ("http://images.cocodataset.org/val2017/000000002473.jpg", "the skier in the air"),
    ("http://images.cocodataset.org/val2017/000000002532.jpg", "the skier"),
    ("http://images.cocodataset.org/val2017/000000002587.jpg", "the banana"),
    ("http://images.cocodataset.org/val2017/000000002592.jpg", "the knife"),
    ("http://images.cocodataset.org/val2017/000000002685.jpg", "the woman in black"),
]

TASKS = [
    "<OD>",
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
]


def prepare_inputs(processor, num_requests):
    """Four tasks per image, processed before timing, then shuffled with seed 42."""
    images = {}
    requests = []  # ((task, image file), input token ids, pixel values)
    for i in range(num_requests):
        url, phrase = IMAGE_URLs[(i // len(TASKS)) % len(IMAGE_URLs)]
        task = TASKS[i % len(TASKS)]
        prompt = (
            task + phrase if task == "<REFERRING_EXPRESSION_SEGMENTATION>" else task
        )
        if url not in images:
            images[url] = load_image(url)
        inputs = processor(text=prompt, images=images[url], return_tensors="pt")
        requests.append(
            (
                (task, url.rsplit("/", 1)[-1]),
                inputs["input_ids"][0].tolist(),
                inputs["pixel_values"][0],
            )
        )
    # Both backends get this one list, so they run the same random order.
    random.Random(42).shuffle(requests)
    labels, input_token_ids, pixels = zip(*requests)
    return list(labels), list(input_token_ids), list(pixels)


def run_fast(args, input_token_ids, pixels, dtype):
    from fastencdec import LLM, Request, SamplingParams

    max_num_seqs = args.max_num_req * args.beamsearch
    # The default KV cache covers max_position_embeddings per sequence (4096 on
    # Florence-2-large: ~26 GB at 128 seqs); size it for max_tokens instead.
    block_size = 16 if platform.machine().lower() in ("arm64", "aarch64") else 32
    llm = LLM(
        args.model,
        max_num_seqs=max_num_seqs,
        num_blocks=max_num_seqs * (math.ceil((args.max_tokens + 2) / block_size) + 1),
        dtype=dtype,
        mm_cache_size=args.mm_cache_size,
    )
    params = SamplingParams(
        max_tokens=args.max_tokens,
        num_beams=args.beamsearch,
        eos_token_id=llm.eos_token_id,
    )
    requests = [
        Request(ids, image_pixels) for ids, image_pixels in zip(input_token_ids, pixels)
    ]

    runner = llm.runner

    def once():
        # Cold cache each run: only reuse within the batch (4 tasks per image) counts.
        runner.mm_cache.clear()
        runner.mm_cache_hits = runner.mm_cache_misses = 0
        rows = [o.completion_token_ids for o in llm.generate(requests, params)]
        # generate() logs a rejected request and returns it empty; a real one has >= 1 token.
        assert all(rows), "fast-encdec rejected requests; see the log above"
        return rows

    seconds, rows = timed(once, args.warmup, args.repeats)
    print(
        f"mm cache: size={args.mm_cache_size}  hits={runner.mm_cache_hits}  "
        f"misses={runner.mm_cache_misses}"
    )
    return seconds, rows, [0] * len(rows)


def run_hf(args, input_token_ids, pixels, dtype, tokenizer):
    from transformers import Florence2ForConditionalGeneration

    model = Florence2ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype
    ).eval()
    batches = []
    for start in range(0, len(input_token_ids), args.max_num_req):
        end = start + args.max_num_req
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
                # Everything else is the checkpoint's generation config, as in fast-encdec.
                do_sample=False,
            )
            for batch in batches
        ]

    seconds, outputs = timed(once, args.warmup, args.repeats)
    rows, padding = [], []
    for output in outputs:
        for row in output.tolist():
            # HF includes the decoder start token; fast-encdec does not.
            tokens = row[1:]
            length = len(tokens)
            while tokens and tokens[-1] == tokenizer.pad_token_id:
                tokens.pop()
            rows.append(tokens)
            padding.append(length - len(tokens))
    return seconds, rows, padding


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["fast", "hf"])
    ap.add_argument("--num-req", type=int, default=128)
    ap.add_argument(
        "--max-num-req",
        type=int,
        default=8,
        help="requests in flight (fast) / padded batch size (hf)",
    )
    ap.add_argument("--beamsearch", type=int, default=1, help="1 = greedy")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument(
        "--mm-cache-size", type=int, default=256, help="fast only; 0 disables"
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    args = ap.parse_args()
    if args.num_req < 1 or args.max_num_req < 1 or args.beamsearch < 1:
        ap.error("--num-req, --max-num-req and --beamsearch must be >= 1")

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
        f"max_num_req={args.max_num_req}  beams={args.beamsearch}  "
        f"seqs/step<={args.max_num_req * args.beamsearch}  "
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
