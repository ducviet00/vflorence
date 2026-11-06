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

from benchmarks.utils import (
    FLORENCE,
    HF_DECODE,
    IMAGES,
    blocks_for,
    load_image,
    strip_hf,
    timed,
)

TASKS = ["<OD>", "<OCR>", "<DETAILED_CAPTION>", "<REFERRING_EXPRESSION_SEGMENTATION>"]


def make_workload(num_req: int):
    """Request i: image (i // 4) % 8, task i % 4 -> every 8 requests mix all tasks."""
    images = {cid: load_image(cid) for cid, _ in IMAGES}
    workload = []
    for i in range(num_req):
        cid, phrase = IMAGES[(i // len(TASKS)) % len(IMAGES)]
        task = TASKS[i % len(TASKS)]
        prompt = task + phrase if task.startswith("<REFERRING") else task
        workload.append((task, cid, prompt, images[cid]))
    return workload


def run_fast(args, workload, dtype):
    from transformers import AutoProcessor

    from fastencdec import LLM, SamplingParams
    from fastencdec.config import default_block_size

    block_size = default_block_size()
    processor = AutoProcessor.from_pretrained(args.model)
    prepared = []
    for _, _, prompt, image in workload:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        prepared.append((inputs["input_ids"][0].tolist(), inputs["pixel_values"][0]))
    llm = LLM(
        args.model,
        num_blocks=blocks_for(
            args.max_num_seq * args.beamsearch, args.max_tokens, block_size
        ),
        block_size=block_size,
        # max_num_seqs counts sequences (beams), so scale to keep max_num_seq requests in flight
        max_num_seqs=args.max_num_seq * args.beamsearch,
        dtype=dtype,
    )
    params = SamplingParams(
        max_tokens=args.max_tokens,
        num_beams=args.beamsearch,
        eos_token_id=llm.eos_token_id,
    )
    requests = [(ids, [llm.decoder_start_token_id], px) for ids, px in prepared]

    def once():
        return list(llm.engine.generate(requests, params).values())

    seconds, rows = timed(once, args.warmup, args.repeats)
    return seconds, rows, [0] * len(rows), processor.tokenizer


def run_hf(args, workload, dtype):
    from transformers import AutoProcessor
    from transformers import Florence2ForConditionalGeneration as HFModel

    processor = AutoProcessor.from_pretrained(args.model)
    model = HFModel.from_pretrained(args.model, dtype=dtype).eval()
    chunks = [
        processor(
            text=[w[2] for w in workload[i : i + args.max_num_seq]],
            images=[w[3] for w in workload[i : i + args.max_num_seq]],
            padding=True,
            return_tensors="pt",
        )
        for i in range(0, len(workload), args.max_num_seq)
    ]

    def once():
        rows, pads = [], []
        with torch.no_grad():
            for inputs in chunks:
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    num_beams=args.beamsearch,
                    **HF_DECODE,
                )
                r, p = strip_hf(out, processor.tokenizer.pad_token_id)
                rows += r
                pads += p
        return rows, pads

    seconds, (rows, pads) = timed(once, args.warmup, args.repeats)
    return seconds, rows, pads, processor.tokenizer


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
    ap.add_argument("--model", default=FLORENCE)
    args = ap.parse_args()
    if args.num_req < 1 or args.max_num_seq < 1 or args.beamsearch < 1:
        ap.error("--num-req, --max-num-seq and --beamsearch must be >= 1")

    import transformers

    transformers.logging.set_verbosity_error()
    dtype = getattr(torch, args.dtype)
    workload = make_workload(args.num_req)
    run = run_fast if args.backend == "fast" else run_hf
    seconds, rows, pads, tokenizer = run(args, workload, dtype)

    print(
        f"\nbackend={args.backend}  model={args.model}  dtype={args.dtype}  "
        f"threads={torch.get_num_threads()}  num_req={args.num_req}  "
        f"max_num_seq={args.max_num_seq}  beams={args.beamsearch}  "
        f"max_tokens={args.max_tokens}\n"
    )
    for i, ((task, cid, _, _), toks) in enumerate(zip(workload, rows)):
        text = tokenizer.decode(toks).replace("\n", " ")
        print(f"#{i:<3} {task:<38} {cid}  {len(toks):>4} tok  {text[:60]}")

    print(f"\n{'task':<38}{'reqs':>6}{'gen tok':>9}{'mean':>8}{'pad tok':>9}")
    for task in TASKS:
        idx = [i for i, w in enumerate(workload) if w[0] == task]
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
