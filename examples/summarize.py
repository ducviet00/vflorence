"""Summarize an article with BART."""

import logging
import time

from transformers import AutoTokenizer

from vflorence import LLM, Request, SamplingParams

MODEL = "facebook/bart-large-cnn"
ARTICLE = (
    "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and the "
    "fox had been running through the meadow for hours, chasing butterflies and "
    "enjoying the warm weather before returning to its den."
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(MODEL, max_num_seqs=4)  # 1 request x 4 beams

    # bart-large-cnn needs these decoding heuristics; without them raw decoding
    # collapses to short, repetitive output.
    params = SamplingParams(
        max_tokens=64,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        min_length=8,
    )

    request = Request(tokenizer(ARTICLE, truncation=True)["input_ids"])
    start = time.perf_counter()
    outputs = llm.generate(request, params)
    text = tokenizer.decode(outputs[0].completion_token_ids, skip_special_tokens=True)
    print(f"{time.perf_counter() - start:.2f}s  {text}")


if __name__ == "__main__":
    logging.basicConfig(format="%(name)s: %(message)s")
    logging.getLogger("vflorence").setLevel(logging.INFO)  # memory usage
    main()
