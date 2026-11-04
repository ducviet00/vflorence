"""Engine lifetime, cache budgets, input validation, and model-loading regressions."""

import unittest
from contextlib import contextmanager
from copy import deepcopy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import BartConfig, GenerationConfig
from transformers import BartForConditionalGeneration as HFBart
from transformers.generation.logits_process import ForcedEOSTokenLogitsProcessor

from fastencdec import LLM, SamplingParams
from fastencdec.config import Config
from fastencdec.engine.cudagraph import DecodeGraphRunner
from fastencdec.engine.llm_engine import LLMEngine
from fastencdec.engine.sequence import Sequence
from fastencdec.layers import attention
from fastencdec.layers.logits_processor import LogitsProcessor
from fastencdec.models.bart import BartForConditionalGeneration
from fastencdec.utils.loader import load_hf_weights


def model(activation="gelu"):
    torch.manual_seed(0)
    return BartForConditionalGeneration(
        BartConfig(
            vocab_size=32,
            d_model=8,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=16,
            decoder_ffn_dim=16,
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
            attn_backend="sdpa",
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
        self.assertTrue(all(b.ref_count == 0 for b in llm.block_manager.blocks))
        self.assertFalse(llm._finished_beams)
        for layer in llm.runner.decoder_layers:
            self.assertFalse(layer.encoder_attn.encoder_kv_cache)
            self.assertIsNone(layer.encoder_attn.cross_staged)

    def test_backend_isolation(self):
        llm = engine()
        params = SamplingParams(max_tokens=3)
        first = llm.results[llm.generate([REQUEST], params)[0]]
        with attention.use_backend("cpu"):
            other = engine()
            other.generate([REQUEST], params)
            second = llm.results[llm.generate([REQUEST], params)[0]]
            self.assertEqual(attention._PREFER, "cpu")
        self.assertEqual(first, second)

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
            device="cpu",
            max_length=3,
            begin_index=1,
            is_beam=False,
        )
        expected = ForcedEOSTokenLogitsProcessor(3, [2, 3])(histories, scores.clone())
        self.assertTrue(torch.equal(processor(histories, scores.clone()), expected))
        llm = engine()
        result = llm.results[llm.generate([REQUEST], params)[0]]
        self.assertEqual(len(result), 2)
        self.assertIn(result[-1], [2, 3])

    def test_results_only_keep_latest_batch(self):
        llm = engine()
        params = SamplingParams(max_tokens=2)
        first = llm.results[llm.generate([REQUEST], params)[0]]
        for _ in range(4):
            ids = llm.generate([REQUEST, REQUEST], params)
            self.assertEqual(set(llm.results), set(ids))
        self.assertEqual(len(first), 2)
        llm.generate([], params)
        self.assertFalse(llm.results)

    def test_shared_beams_fit_two_blocks(self):
        llm = engine(num_blocks=2)
        params = SamplingParams(max_tokens=3, num_beams=2, min_length=4)
        ids = llm.generate([REQUEST], params)
        self.assertEqual(len(llm.results[ids[0]]), 3)
        self.assert_clean(llm)

    def test_copy_budget_preserves_nonwriting_owner(self):
        llm = engine(num_blocks=3)
        params = SamplingParams()
        parent = Sequence(0, [1], [3], params)
        blocks = llm.block_manager
        blocks.ensure_capacity(parent, 1, llm.runner._copy_block)
        parent.num_cached_tokens = 1
        children = [llm._fork(parent, 4, 0), llm._fork(parent, 5, 0)]
        self.assertEqual(blocks.required_new_blocks(children), 2)
        blocks.deallocate(parent)
        self.assertEqual(blocks.required_new_blocks(children), 1)

    def test_failed_copy_returns_reserved_block(self):
        llm = engine(num_blocks=2)
        parent = Sequence(0, [1], [3], SamplingParams())
        blocks = llm.block_manager
        blocks.ensure_capacity(parent, 1, llm.runner._copy_block)
        parent.num_cached_tokens = 1
        child = llm._fork(parent, 4, 0)
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            blocks.ensure_capacity(
                child, 1, Mock(side_effect=RuntimeError("copy failed"))
            )
        self.assertEqual(blocks.num_free_blocks, 1)
        self.assertEqual(blocks.blocks[parent.block_table[0]].ref_count, 2)

    def test_invalid_sampling_params(self):
        invalid = [
            {"max_tokens": 0},
            {"max_tokens": 1.5},
            {"max_tokens": True},
            {"num_beams": 0},
            {"top_k": -1},
            {"top_k": 0.5},
            {"no_repeat_ngram_size": -1},
            {"min_length": -1},
            {"top_p": 0},
            {"top_p": 1.1},
            {"repetition_penalty": 0},
            {"temperature": float("nan")},
            {"temperature": -1.0},
            {"length_penalty": float("inf")},
            {"eos_token_id": -1},
            {"suppress_tokens": [-1]},
            {"suppress_tokens": 3},
            {"forced_bos_token_id": [1]},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SamplingParams(**kwargs)
        llm = engine()
        params = SamplingParams()
        params.num_beams = 0
        with self.assertRaises(ValueError):
            llm.generate([REQUEST], params)
        with self.assertRaisesRegex(ValueError, "beam sampling"):
            llm.generate([REQUEST], SamplingParams(num_beams=2, temperature=1))
        self.assert_clean(llm)

    def test_generation_config_ignores_warpers_without_sampling(self):
        config = SimpleNamespace(
            **{name: None for name in SamplingParams.__slots__},
            do_sample=False,
            max_new_tokens=2,
            max_length=None,
        )
        config.top_p, config.top_k = 0.0, -1
        params = SamplingParams.from_generation_config(config)
        self.assertEqual((params.top_p, params.top_k), (1.0, 0))

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

    def test_cross_cache_reuses_finished_request_space(self):
        runner = engine(block_size=256).runner
        graph = DecodeGraphRunner(runner, max_num_seqs=2)
        runner.graph = graph
        for request_id in range(17):
            for layer in runner.decoder_layers:
                attn = layer.encoder_attn
                key = torch.full(
                    (attn.num_heads, 1024, attn.head_dim), float(request_id)
                )
                attn.encoder_kv_cache[request_id] = (key, key + 1)
        for request_id in range(1, 17):
            ids = (request_id, 0) if request_id == 1 else (0, request_id)
            graph.stage([SimpleNamespace(request_id=i) for i in ids])
            self.assertEqual(graph.cross_capacity, 9)
            self.assertEqual(graph.next_block, 8)
            anchor = graph.req_blocks[0][0]
            first = graph.req_blocks[request_id][0]
            self.assertEqual({anchor, first}, {0, 4})
            for layer in runner.decoder_layers:
                key, value = layer.encoder_attn.cross_cache
                self.assertTrue((key[anchor : anchor + 4] == 0).all())
                self.assertTrue((key[first : first + 4] == request_id).all())
                self.assertTrue((value[first : first + 4] == request_id + 1).all())
            if request_id == 1:
                graph.graphs[1] = object()
            else:
                self.assertIn(1, graph.graphs)
            runner.clear_request(request_id)

    def test_interrupted_graph_capture_can_retry(self):
        runner = engine(block_size=256).runner
        graph = DecodeGraphRunner(runner, max_num_seqs=2)
        for layer in runner.decoder_layers:
            attn = layer.encoder_attn
            key = torch.zeros(attn.num_heads, 17, attn.head_dim)
            attn.encoder_kv_cache[0] = (key, key)
        graph.stage([SimpleNamespace(request_id=0)])
        attempts = 0

        @contextmanager
        def capture(*args):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise RuntimeError("capture OOM")
            yield

        with (
            patch.object(
                runner, "decoder", side_effect=lambda ids, pos: torch.zeros(len(ids), 8)
            ),
            patch(
                "torch.cuda.CUDAGraph",
                side_effect=lambda: SimpleNamespace(pool=lambda: object()),
            ),
            patch("torch.cuda.graph", capture),
            patch("torch.cuda.synchronize"),
        ):
            with self.assertRaisesRegex(RuntimeError, "capture OOM"):
                graph.capture()
            self.assertFalse(graph.graphs)
            self.assertIsNone(graph.pool)
            graph.capture()
            self.assertEqual(set(graph.graphs), {1, 2})
            self.assertEqual(attempts, 4)

    def test_cross_cache_allocation_failure_can_retry(self):
        llm = engine(block_size=256)
        runner = llm.runner
        # Two cache layers let allocation fail after replacing only the first one.
        runner.decoder_layers.append(deepcopy(runner.decoder_layers[0]))
        graph = DecodeGraphRunner(runner, max_num_seqs=2)
        runner.graph = graph

        def add(request_id, length, value):
            seq = Sequence(request_id, [1], [3] * length, SamplingParams())
            llm.scheduler.add(seq)
            for layer in runner.decoder_layers:
                attn = layer.encoder_attn
                key = torch.full((attn.num_heads, length, attn.head_dim), float(value))
                attn.encoder_kv_cache[request_id] = (key, key)
            return seq

        first = add(0, 17, 0)
        graph.stage([first])
        graph.graphs[1] = object()
        graph.pool = object()
        second = add(1, 1024, 1)
        zeros = torch.zeros
        calls = 0

        def fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise torch.OutOfMemoryError("cross-cache allocation failed")
            return zeros(*args, **kwargs)

        with (
            patch("torch.zeros", side_effect=fail),
            self.assertRaisesRegex(
                torch.OutOfMemoryError, "cross-cache allocation failed"
            ),
        ):
            graph.stage([first, second])
        self.assertFalse(graph.graphs)
        self.assertIsNone(graph.pool)
        self.assertIsNone(graph.bufs)
        self.assertEqual(graph.cross_capacity, 0)
        self.assertTrue(
            all(
                layer.encoder_attn.cross_cache is None
                for layer in runner.decoder_layers
            )
        )
        # generate() must release even requests not present in scheduler.running.
        llm._abort([0, 1])
        retry = add(2, 17, 9)
        graph.stage([retry])
        self.assertEqual(graph.cross_capacity, 2)
        for layer in runner.decoder_layers:
            key, value = layer.encoder_attn.cross_cache
            self.assertTrue((key[0, :17] == 9).all())
            self.assertTrue((value[0, :17] == 9).all())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
