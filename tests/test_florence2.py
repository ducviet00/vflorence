"""Greedy/beam parity against HuggingFace on a tiny random Florence-2.

The tiny checkpoint is internally inconsistent (its processor inserts
``<image>`` id 51289 while the model replaces id 4, and 768px images overflow
its 2D position embeddings), so inputs are built by hand from a small synthetic
image instead of going through the processor.

Run with:  PYTHONPATH=. python tests/test_florence2.py
"""

import json
import os
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import Florence2ForConditionalGeneration as HFFlorence2

from vflorence import LLM, Request, SamplingParams

MODEL = "hf-tiny-v2/tiny-random-Florence2ForConditionalGeneration"


def build_encoder_token_ids(config, tokenizer, num_image_tokens):
    text = config.text_config
    prompt_ids = tokenizer("<CAPTION>", add_special_tokens=False)["input_ids"]
    return (
        [config.image_token_id] * num_image_tokens
        + [text.bos_token_id]
        + prompt_ids
        + [text.eos_token_id]
    )


def main():
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL)
    config.text_config.d_model = 32
    config.text_config.encoder_attention_heads = 1
    config.text_config.decoder_attention_heads = 1
    config.vision_config.projection_dim = 32
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf = HFFlorence2(config).eval()

    with TemporaryDirectory() as tmpdir:
        hf.save_pretrained(tmpdir)
        # compile=False is a full opt-out: the vision tower must stay eager too.
        eager_runner = LLM(tmpdir, max_num_seqs=8, compile=False).runner
        ours = eager_runner.model
        tower_eager = (
            not eager_runner.compiled
            and eager_runner.image_fn == ours.get_image_features
        )
        assert tower_eager, "compile=False compiled the vision tower or decoder"
        print(
            f"[{'OK' if tower_eager else 'MISMATCH'}] compile=False keeps the tower eager"
        )

        # Image features and encoder hidden states must match exactly.
        pixel_values = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            feat_ref = hf.get_image_features(pixel_values).pooler_output
            feat_our = ours.get_image_features(pixel_values)
            num_image_tokens = feat_ref.shape[1]
            encoder_token_ids = build_encoder_token_ids(
                config, tokenizer, num_image_tokens
            )
            encoder = torch.tensor([encoder_token_ids])
            hidden_ref = hf.model(
                input_ids=encoder, pixel_values=pixel_values
            ).encoder_last_hidden_state
            hidden_our = ours.encode(encoder, pixel_values)

        feat_diff = (feat_ref - feat_our).abs().max().item()
        hidden_diff = (hidden_ref - hidden_our).abs().max().item()
        tol = 1e-5
        assert feat_diff <= tol, feat_diff
        assert hidden_diff <= tol, hidden_diff
        print(
            f"[{'OK' if feat_diff <= tol else 'MISMATCH'}] image features: maxdiff={feat_diff}"
        )
        print(
            f"[{'OK' if hidden_diff <= tol else 'MISMATCH'}] encoder hidden: maxdiff={hidden_diff}"
        )

        # With a preprocessor config the warmup compiles the image graphs:
        # image requests must then compile nothing new.
        preprocessor = os.path.join(tmpdir, "preprocessor_config.json")
        with open(preprocessor, "w") as f:
            json.dump({"crop_size": {"height": 32, "width": 32}}, f)
        LLM(tmpdir, max_num_seqs=8)  # no image_seq_length: text-only warmup, no crash
        with open(preprocessor, "w") as f:
            size = {"height": 32, "width": 32}
            json.dump({"crop_size": size, "image_seq_length": num_image_tokens}, f)
        graphs = lambda: torch._dynamo.utils.counters["stats"]["unique_graphs"]
        engine = LLM(tmpdir, max_num_seqs=8)
        warm = graphs()
        image_request = Request(encoder_token_ids, pixel_values[0])
        for requests in ([image_request], [image_request] * 3):
            engine.generate(requests, SamplingParams(max_tokens=3), use_tqdm=False)
        assert graphs() == warm, "an image request compiled a graph the warmup missed"
        os.remove(preprocessor)  # the rest of the test builds its inputs by hand
        print("[OK] image warmup compiled every graph")

        # Greedy/beam generation parity through the paged engine.
        for beams in (1, 2, 4):
            params = SamplingParams(max_tokens=6, num_beams=beams)
            engine = LLM(tmpdir, max_num_seqs=8)
            (out,) = engine.generate(
                [Request(encoder_token_ids, pixel_values[0])],
                params,
            )
            our_tokens = [
                config.text_config.decoder_start_token_id
            ] + out.completion_token_ids

            with torch.no_grad():
                ref = hf.generate(
                    input_ids=encoder,
                    pixel_values=pixel_values,
                    max_new_tokens=6,
                    num_beams=beams,
                    do_sample=False,
                )
            ref_tokens = ref[0].tolist()
            assert our_tokens == ref_tokens, (beams, our_tokens, ref_tokens)
            print(f"[OK] beams={beams}: {our_tokens}")

        # Different images/lengths must batch to the same tokens as single runs.
        batch = [(torch.randn(3, 32, 32), None), (torch.randn(3, 40, 40), None)]
        singles = []
        for image, _ in batch:
            with torch.no_grad():
                n_img = ours.get_image_features(image.unsqueeze(0)).shape[1]
            ids = build_encoder_token_ids(config, tokenizer, n_img)
            params = SamplingParams(max_tokens=6, num_beams=1)
            engine = LLM(tmpdir, max_num_seqs=8)
            (out,) = engine.generate([Request(ids, image)], params)
            singles.append(
                [config.text_config.decoder_start_token_id] + out.completion_token_ids
            )

        params = SamplingParams(max_tokens=6, num_beams=1)
        engine = LLM(tmpdir, max_num_seqs=8)
        engine_requests = []
        for image, _ in batch:
            with torch.no_grad():
                n_img = ours.get_image_features(image.unsqueeze(0)).shape[1]
            ids = build_encoder_token_ids(config, tokenizer, n_img)
            engine_requests.append(Request(ids, image))
        completions = engine.generate(engine_requests, params)
        for i, out in enumerate(completions):
            batched = [
                config.text_config.decoder_start_token_id
            ] + out.completion_token_ids
            assert batched == singles[i], (i, batched, singles[i])
            print(f"[OK] batch-vs-single {i}: {batched}")

        # Multimodal cache: a repeated image runs the vision encoder once, across calls.
        image_a, image_b = batch[0][0], torch.randn(3, 32, 32)
        ids_a = engine_requests[0].input_ids
        engine = LLM(tmpdir, max_num_seqs=8, mm_cache_size=1)
        runner, calls = engine.runner, []
        image_fn = runner.image_fn
        runner.image_fn = lambda px: calls.append(len(px)) or image_fn(px)
        first = engine.generate(
            [Request(ids_a, image_a), Request(ids_a, image_a)], params
        )
        again = engine.generate([Request(ids_a, image_a.clone())], params)
        assert calls == [1], calls  # dedup within a step + hit across calls
        assert [o.completion_token_ids for o in first] == [singles[0][1:]] * 2, first
        assert [o.completion_token_ids for o in again] == [singles[0][1:]], again
        engine.generate([Request(ids_a, image_b)], params)  # evicts image_a
        assert len(runner.mm_cache) == 1 and calls == [1, 1], (runner.mm_cache, calls)
        engine.generate([Request(ids_a, image_a)], params)
        assert calls == [1, 1, 1], calls
        assert (runner.mm_cache_hits, runner.mm_cache_misses) == (2, 3)
        # Mixed hit + miss in one step: the later request (cached image_a) stays.
        hash_a = next(iter(runner.mm_cache))
        engine.generate([Request(ids_a, image_b), Request(ids_a, image_a)], params)
        assert list(runner.mm_cache) == [hash_a] and calls == [1, 1, 1, 1], calls
        print("[OK] mm cache: dedup, cross-call hit, LRU eviction, hit stats")


if __name__ == "__main__":
    main()
