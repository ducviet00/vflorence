"""Engine lifetime, cache budgets, input validation, and model-loading regressions."""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch
from transformers import BartConfig, GenerationConfig
from transformers import BartForConditionalGeneration as HFBart
from transformers.generation.logits_process import ForcedEOSTokenLogitsProcessor

from fastencdec import LLM, SamplingParams
from fastencdec.config import Config
from fastencdec.engine.llm_engine import LLMEngine
from fastencdec.engine.sequence import Sequence
from fastencdec.layers.logits_processor import LogitsProcessor
from fastencdec.models.bart import BartForConditionalGeneration
from fastencdec.utils.loader import load_hf_weights


def model(activation="gelu"):
    torch.manual_seed(0)
    return BartForConditionalGeneration(
        BartConfig(
            vocab_size=32,
            d_model=64,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            max_position_embeddings=128,
            decoder_start_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            activation_function=activation,
        )
    ).eval()


def engine(num_blocks=8, block_size=16):
    return LLMEngine(
        model(),
        Config(
            num_blocks=num_blocks,
            block_size=block_size,
            compile=False,
        ),
        eos_token_id=2,
    )


REQUEST = ([3, 4, 5], [1], None)


class EngineTests(unittest.TestCase):
    def assert_clean(self, llm):
        self.assertTrue(llm.is_finished())
        self.assertEqual(
            llm.block_manager.num_free_blocks, llm.block_manager.num_blocks
        )
        self.assertTrue(not any(llm.block_manager.ref_counts))
        self.assertFalse(llm._finished_beams)
        for layer in llm.runner.decoder_layers:
            self.assertFalse(layer.encoder_attn.encoder_kv_cache)
        self.assertIsNone(llm.runner._cross_staged)

    def test_cache_failure_then_valid_batch(self):
        llm = engine(num_blocks=1)
        params = SamplingParams(max_tokens=3, min_length=4)
        with self.assertRaisesRegex(RuntimeError, "KV cache is full"):
            llm.generate([REQUEST, REQUEST], params)
        self.assert_clean(llm)
        self.assertEqual(len(llm.generate([REQUEST], params)), 1)
        self.assert_clean(llm)

    def test_forward_failure_after_beam_fork(self):
        llm = engine(num_blocks=4)
        params = SamplingParams(max_tokens=4, num_beams=2, min_length=5)
        forward = llm.runner.run
        calls = 0

        def fail(seqs):
            nonlocal calls
            calls += 1
            logits = forward(seqs)
            if calls == 2:
                raise RuntimeError("failed after writing cache")
            return logits

        with (
            patch.object(llm.runner, "run", side_effect=fail),
            self.assertRaisesRegex(RuntimeError, "failed after writing cache"),
        ):
            llm.generate([REQUEST], params)
        self.assert_clean(llm)
        self.assertEqual(len(llm.generate([REQUEST], params)), 1)
        self.assert_clean(llm)

    def test_partial_enqueue_failure(self):
        llm = engine()
        with self.assertRaises(TypeError):
            llm.generate([REQUEST, (None, [1], None)], SamplingParams())
        self.assert_clean(llm)

    def test_beam_finalization_failure_cleans_detached_request(self):
        llm = engine()
        params = SamplingParams(
            max_tokens=3, num_beams=2, min_length=4, length_penalty=1024.0
        )
        with self.assertRaises(OverflowError):
            llm.generate([REQUEST], params)
        self.assert_clean(llm)
        llm.generate([REQUEST], SamplingParams(max_tokens=1))
        self.assert_clean(llm)

    def test_forced_eos_list_matches_reference(self):
        try:
            params = SamplingParams.from_generation_config(
                GenerationConfig(forced_eos_token_id=[2, 3], max_new_tokens=2)
            )
        except ValueError as exc:
            self.fail(f"valid multi-EOS configuration rejected: {exc}")
        histories = torch.tensor([[1, 4]])
        scores = torch.zeros(1, 8)
        processor = LogitsProcessor(
            params,
            vocab_size=8,
            max_length=3,
            begin_index=1,
        )
        expected = ForcedEOSTokenLogitsProcessor(3, [2, 3])(histories, scores.clone())
        self.assertTrue(torch.equal(processor(histories, scores.clone()), expected))
        llm = engine()
        (result,) = llm.generate([REQUEST], params).values()
        self.assertEqual(len(result), 2)
        self.assertIn(result[-1], [2, 3])

    def test_generate_returns_each_batch_and_keeps_nothing(self):
        llm = engine()
        params = SamplingParams(max_tokens=2)
        for _ in range(4):
            completions = llm.generate([REQUEST, REQUEST], params)
            self.assertEqual([len(t) for t in completions.values()], [2, 2])
            self.assertFalse(llm.results)
        self.assertEqual(llm.generate([], params), {})

    def test_too_long_inputs_are_rejected_before_running(self):
        llm = engine(num_blocks=64)
        cases = [
            (([3] * 129, [1], None), SamplingParams(max_tokens=2), "encoder prompt"),
            (REQUEST, SamplingParams(max_tokens=129), "max_position_embeddings"),
            (REQUEST, SamplingParams(max_tokens=128, num_beams=8), "KV blocks"),
        ]
        for request, params, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                llm.generate([request], params)
            self.assert_clean(llm)
        # The exact limits still run: 128 encoder tokens and 128 new tokens.
        (tokens,) = llm.generate(
            [([3] * 128, [1], None)], SamplingParams(max_tokens=128, min_length=200)
        ).values()
        self.assertEqual(len(tokens), 128)

    def test_shared_beams_fit_two_blocks(self):
        llm = engine(num_blocks=2)
        params = SamplingParams(max_tokens=3, num_beams=2, min_length=4)
        (tokens,) = llm.generate([REQUEST], params).values()
        self.assertEqual(len(tokens), 3)
        self.assert_clean(llm)

    def test_copy_budget_preserves_nonwriting_owner(self):
        llm = engine(num_blocks=3)
        params = SamplingParams()
        parent = Sequence(0, [1], [3], params)
        blocks = llm.block_manager
        blocks.ensure_capacity(parent)
        parent.num_cached_tokens = 1
        children = [llm._fork(parent, 4, 0), llm._fork(parent, 5, 0)]
        self.assertEqual(blocks.required_new_blocks(children), 2)
        blocks.deallocate(parent)
        self.assertEqual(blocks.required_new_blocks(children), 1)

    def test_copy_on_write_copies_each_pair(self):
        llm = engine(num_blocks=4)
        blocks = llm.block_manager
        parent = Sequence(0, [1], [3], SamplingParams())
        self.assertIsNone(blocks.ensure_capacity(parent))
        parent.num_cached_tokens = 1
        child = llm._fork(parent, 4, 0)
        src = parent.block_table[0]
        dst = blocks.ensure_capacity(child)[1]
        self.assertEqual(child.block_table, [dst])
        self.assertEqual((blocks.ref_counts[src], blocks.ref_counts[dst]), (1, 1))

        caches = [
            cache
            for layer in llm.runner.decoder_layers
            for cache in (layer.self_attn.k_cache, layer.self_attn.v_cache)
        ]
        for cache in caches:
            for block in range(4):
                cache[block] = block
        llm.runner._copy_blocks([(0, 2), (1, 3)])
        for cache in caches:
            self.assertEqual([cache[b].unique().item() for b in range(4)], [0, 1, 0, 1])

    def test_failed_copy_aborts_cleanly(self):
        llm = engine()
        params = SamplingParams(max_tokens=3, num_beams=2, min_length=4)
        failing = patch.object(
            llm.runner, "_copy_blocks", side_effect=RuntimeError("copy failed")
        )
        with failing, self.assertRaisesRegex(RuntimeError, "copy failed"):
            llm.generate([REQUEST], params)
        self.assert_clean(llm)
        (tokens,) = llm.generate([REQUEST], params).values()
        self.assertEqual(len(tokens), 3)

    def test_invalid_sampling_params(self):
        invalid = [
            {"max_tokens": 0},
            {"max_tokens": 1.5},
            {"max_tokens": True},
            {"num_beams": 0},
            {"no_repeat_ngram_size": -1},
            {"min_length": -1},
            {"repetition_penalty": 0},
            {"length_penalty": float("inf")},
            {"eos_token_id": -1},
            {"suppress_tokens": [-1]},
            {"suppress_tokens": 3},
            {"forced_bos_token_id": [1]},
            {"forced_eos_token_id": [2, -1]},
            {"early_stopping": "never"},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SamplingParams(**kwargs)
        llm = engine()
        params = SamplingParams()
        params.num_beams = 0
        with self.assertRaises(ValueError):
            llm.generate([REQUEST], params)
        self.assert_clean(llm)

    def test_florence_token_prompt_is_rejected(self):
        llm = LLM.__new__(LLM)
        llm.multimodal = True
        llm.eos_token_id = 2
        llm.decoder_start_token_id = 1
        llm.processor = Mock()
        with self.assertRaisesRegex(ValueError, "text prompt"):
            llm.generate({"prompt_token_ids": [3]}, SamplingParams())
        llm.processor.assert_not_called()

    def test_missing_and_extra_weights_are_rejected(self):
        ref = HFBart(model().config)
        state = ref.state_dict()
        for missing in (True, False):
            bad = dict(state)
            if missing:
                del bad["model.encoder.layers.0.fc1.weight"]
            else:
                bad["extra.weight"] = torch.zeros(1)
            with self.subTest(missing=missing), TemporaryDirectory() as path:
                ref.save_pretrained(path, state_dict=bad)
                with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                    load_hf_weights(model(), path)

    def test_relu_matches_reference(self):
        ours = model("relu")
        ref = HFBart(ours.config).eval()
        with TemporaryDirectory() as path:
            ref.save_pretrained(path)
            load_hf_weights(ours, path)
        inputs = torch.tensor([[3, 4, 5]])
        with torch.no_grad():
            want = ref.get_encoder()(input_ids=inputs).last_hidden_state
            got = ours.encode(inputs)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
        with self.assertRaises(KeyError):
            model("unknown_activation")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
