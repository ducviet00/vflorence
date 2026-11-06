"""Shared helpers for the benchmark scripts: timer, KV sizing, inputs, HF kwargs."""

import os
import time
import urllib.request

from PIL import Image

FLORENCE = "florence-community/Florence-2-base"
BART = "facebook/bart-large-cnn"

COCO = "http://images.cocodataset.org/val2017/{}.jpg"
IMAGES = [  # (COCO val2017 id, referring expression)
    ("000000039769", "the cat on the left"),
    ("000000000139", "the television"),
    ("000000000285", "the bear"),
    ("000000000632", "the bed"),
    ("000000000724", "the stop sign"),
    ("000000000776", "the teddy bear"),
    ("000000000785", "the skier"),
    ("000000000802", "the oven"),
]

PARAGRAPH = (
    "Scientists at the university announced on Monday that they had discovered a "
    "new species of butterfly in the Amazon rainforest. The insect has bright blue "
    "wings and a wingspan of nearly ten centimetres. The city council approved the "
    "new transit plan after a long debate. The plan adds two tram lines, extends "
    "the bus network to the outer districts and introduces a single ticket for all "
    "modes of transport from next year."
)

# Neutral decoding on the HF side: override the checkpoints' generation_config so
# both engines do the same work (no forced tokens, no n-gram blocking).
HF_DECODE = {
    "do_sample": False,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "early_stopping": False,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}


def timed(fn, warmup: int, repeats: int):
    """Run ``fn`` warmup + repeats times; return (best seconds, last result)."""
    for _ in range(warmup):
        fn()
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def blocks_for(seqs: int, max_tokens: int, block_size: int) -> int:
    """KV blocks for ``seqs`` sequences of up to ``max_tokens`` decoder tokens.

    The paged cache holds only decoder self-attention K/V (cross-attention K/V is
    kept separately), so encoder length does not count.  +1 for the start token,
    +1 block per sequence for copy-on-write slack during beam search.
    """
    return seqs * ((max_tokens + 1 + block_size - 1) // block_size + 1)


def load_image(coco_id: str, cache_dir: str = "/tmp/fastencdec/coco") -> Image.Image:
    path = os.path.join(cache_dir, f"{coco_id}.jpg")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(COCO.format(coco_id), path)
    return Image.open(path).convert("RGB")


def bart_prompt(tokenizer, tokens: int) -> str:
    """A text whose encoder length is ``tokens`` (excluding BOS/EOS)."""
    text = PARAGRAPH
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < tokens:
        text += " " + PARAGRAPH
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:tokens]
    return tokenizer.decode(ids)


def strip_hf(out, pad_token_id):
    """HF generate output -> (token rows without decoder-start/trailing pad, pad counts)."""
    rows, pads = [], []
    for row in out:
        toks = row.tolist()[1:]
        n_pad = 0
        while toks and toks[-1] == pad_token_id:
            toks.pop()
            n_pad += 1
        rows.append(toks)
        pads.append(n_pad)
    return rows, pads
