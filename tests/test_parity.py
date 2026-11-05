"""Greedy/beam parity against HuggingFace on a tiny random BART.

Run with:  PYTHONPATH=. python tests/test_parity.py
"""

import os
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from fastencdec import LLM, SamplingParams

MODEL = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
TEXT = "The quick brown fox jumps over the lazy dog."
BATCH = [TEXT, "A second, noticeably longer sentence used to exercise batching."]


def main():
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL)
    config.d_model = 64
    config.encoder_attention_heads = 2
    config.decoder_attention_heads = 2
    config.encoder_ffn_dim = 64
    config.decoder_ffn_dim = 64

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForSeq2SeqLM.from_config(config).eval()
    inputs = tokenizer(TEXT, return_tensors="pt")

    with TemporaryDirectory() as tmpdir:
        hf.save_pretrained(tmpdir)
        tokenizer.save_pretrained(tmpdir)
        llm = LLM(tmpdir, num_blocks=64, block_size=16)

    for beams in (1, 2, 4):
        params = SamplingParams(max_tokens=12, num_beams=beams)
        ours = llm.generate(TEXT, sampling_params=params)[0].outputs[0].token_ids
        with torch.no_grad():
            ref = hf.generate(
                **inputs,
                max_new_tokens=12,
                num_beams=beams,
                do_sample=False,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
            )
        ref_tokens = ref[0].tolist()[1:]
        assert ours == ref_tokens, (beams, ours, ref_tokens)
        print(f"[OK] beams={beams}: {ours!r}")

    # A batch must give the same tokens as running each prompt on its own
    # (exercises the batched encoder + cross-attn cache, and for beams > 1 the
    # grouped log_softmax/processor/topk).  EOS is suppressed so the requests
    # stay the same decoder length and batch into one group.
    for beams in (1, 2, 4):
        params = SamplingParams(max_tokens=12, num_beams=beams, min_length=13)
        batched = [
            o.outputs[0].token_ids for o in llm.generate(BATCH, sampling_params=params)
        ]
        single = [
            llm.generate(text, sampling_params=params)[0].outputs[0].token_ids
            for text in BATCH
        ]
        for i, (b, s) in enumerate(zip(batched, single)):
            assert b == s, (beams, i, b, s)
            print(f"[OK] batch-vs-single beams={beams} seq={i}: {b!r}")


if __name__ == "__main__":
    main()
