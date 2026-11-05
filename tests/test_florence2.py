"""Greedy/beam parity against HuggingFace on a tiny random Florence-2.

The tiny checkpoint is internally inconsistent (its processor inserts
``<image>`` id 51289 while the model replaces id 4, and 768px images overflow
its 2D position embeddings), so inputs are built by hand from a small synthetic
image instead of going through the processor.

Run with:  PYTHONPATH=. python tests/test_florence2.py
"""

import os
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import Florence2ForConditionalGeneration as HFFlorence2

from fastencdec import SamplingParams
from fastencdec.config import Config
from fastencdec.engine import LLMEngine
from fastencdec.models.florence2 import Florence2ForConditionalGeneration
from fastencdec.utils.loader import load_hf_weights

MODEL = "hf-tiny-v2/tiny-random-Florence2ForConditionalGeneration"


def build_encoder_ids(config, tokenizer, num_image_tokens):
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

    ours = Florence2ForConditionalGeneration(config)
    with TemporaryDirectory() as tmpdir:
        hf.save_pretrained(tmpdir)
        load_hf_weights(ours, tmpdir, hf_model_cls=HFFlorence2)
    ours = ours.eval()

    # compile=False is a full opt-out: the vision tower must stay eager too.
    eager_runner = LLMEngine(
        ours, Config(num_blocks=64, block_size=16, max_num_seqs=8, compile=False)
    ).runner
    tower_eager = eager_runner.mm_step is None and eager_runner.decode_step is None
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
        encoder_ids = build_encoder_ids(config, tokenizer, num_image_tokens)
        encoder = torch.tensor([encoder_ids])
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

    # Greedy/beam generation parity through the paged engine.
    for beams in (1, 2, 4):
        params = SamplingParams(max_tokens=6, num_beams=beams)
        engine = LLMEngine(ours, Config(num_blocks=64, block_size=16, max_num_seqs=8))
        (request,) = engine.generate(
            [
                (
                    encoder_ids,
                    [config.text_config.decoder_start_token_id],
                    pixel_values[0],
                )
            ],
            params,
        )
        our_tokens = [config.text_config.decoder_start_token_id] + engine.results[
            request
        ]

        with torch.no_grad():
            ref = hf.generate(
                input_ids=encoder,
                pixel_values=pixel_values,
                max_new_tokens=6,
                num_beams=beams,
                do_sample=False,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
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
        ids = build_encoder_ids(config, tokenizer, n_img)
        params = SamplingParams(max_tokens=6, num_beams=1)
        engine = LLMEngine(ours, Config(num_blocks=64, block_size=16, max_num_seqs=8))
        (request,) = engine.generate(
            [(ids, [config.text_config.decoder_start_token_id], image)], params
        )
        singles.append(
            [config.text_config.decoder_start_token_id] + engine.results[request]
        )

    params = SamplingParams(max_tokens=6, num_beams=1)
    engine = LLMEngine(ours, Config(num_blocks=64, block_size=16, max_num_seqs=8))
    engine_requests = []
    for image, _ in batch:
        with torch.no_grad():
            n_img = ours.get_image_features(image.unsqueeze(0)).shape[1]
        ids = build_encoder_ids(config, tokenizer, n_img)
        engine_requests.append(
            (ids, [config.text_config.decoder_start_token_id], image)
        )
    requests = engine.generate(engine_requests, params)
    for i, request in enumerate(requests):
        batched = [config.text_config.decoder_start_token_id] + engine.results[request]
        assert batched == singles[i], (i, batched, singles[i])
        print(f"[OK] batch-vs-single {i}: {batched}")


if __name__ == "__main__":
    main()
