"""Engine lifetime, cache budgets, input validation, and model-loading regressions."""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from transformers import BartConfig, GenerationConfig
from transformers import BartForConditionalGeneration as HFBart
from transformers.generation.logits_process import ForcedEOSTokenLogitsProcessor

from vflorence import LLM, Request, RequestOut, SamplingParams
from vflorence.config import Config
from vflorence.engine.sequence import Sequence
from vflorence.layers.logits_processor import LogitsProcessor
from vflorence.models.bart import BartForConditionalGeneration
from vflorence.utils.context import get_context, set_context
from vflorence.utils.loader import load_hf_weights


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


REQUEST = Request([3, 4, 5])
# What the engine fills unset fields with when a checkpoint ships no generation config.
DEFAULTS = SamplingParams.from_generation_config(GenerationConfig())


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = TemporaryDirectory()
        torch.manual_seed(0)
        HFBart(model().config).eval().save_pretrained(cls._tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def engine(self, max_num_seqs=2, **kwargs):
        return LLM(
            self._tmpdir.name,
            max_num_seqs=max_num_seqs,
            compile=False,
            **{"max_decoder_len": 128, **kwargs},
        )

    def assert_clean(self, llm):
        blocks = llm.scheduler.block_manager
        self.assertTrue(llm.is_finished())
        self.assertEqual(blocks.num_free_blocks, blocks.num_blocks)
        self.assertTrue(not any(blocks.ref_counts))
        self.assertFalse(llm.scheduler._finished_beams)
        runner = llm.runner
        self.assertFalse(runner._cross_pages)
        pool = runner.decoder_layers[0].encoder_attn.k_cache.shape[0]
        self.assertEqual(sorted(runner._cross_free), list(range(pool)))
        self.assertIsNone(llm.runner._cross_staged)

    def test_cache_full_raises(self):
        llm = self.engine(max_num_seqs=1)
        params = SamplingParams(max_tokens=3, min_length=4).with_defaults(DEFAULTS)
        llm.scheduler.running = [
            Sequence(i, [1], [3], params)
            for i in range(llm.scheduler.block_manager.num_blocks + 1)
        ]
        with self.assertRaisesRegex(RuntimeError, "KV cache is full"):
            llm.scheduler.schedule()

    def test_mixed_sampling_params_rejected(self):
        llm = self.engine(max_num_seqs=4)
        llm.add_request(REQUEST, SamplingParams(max_tokens=2))
        with self.assertRaisesRegex(ValueError, "share the same SamplingParams"):
            llm.add_request(REQUEST, SamplingParams(max_tokens=2, num_beams=2))
        llm.add_request(REQUEST, SamplingParams(max_tokens=2))  # equal values are fine
        while not llm.is_finished():
            llm.step()
        self.assert_clean(llm)

    def test_mutated_sampling_params_rebuild_processor(self):
        llm = self.engine()
        params = SamplingParams(max_tokens=2, min_length=10, forced_eos_token_id=2)
        (out,) = llm.generate([REQUEST], params)
        self.assertEqual(len(out.completion_token_ids), 2)
        params.max_tokens = 4
        (out,) = llm.generate([REQUEST], params)
        self.assertEqual(len(out.completion_token_ids), 4)
        self.assert_clean(llm)

    def test_pixel_values_validation(self):
        llm = self.engine()
        with self.assertRaisesRegex(ValueError, "pixel_values"):
            llm.add_request(Request([3], torch.zeros(1)), SamplingParams())
        self.assert_clean(llm)

    def test_rejected_request_returns_empty_and_others_run(self):
        llm = self.engine()
        bad1 = Request([3] * 129)  # longer than max_position_embeddings
        bad2 = Request([4] * 130)
        with self.assertLogs("vflorence.engine.llm_engine", "ERROR") as logs:
            outs = llm.generate([bad1, REQUEST, bad2], SamplingParams(max_tokens=2))
        self.assertEqual(len(logs.output), 2)
        self.assertIn("encoder prompt", logs.output[0])
        self.assertEqual([len(o.completion_token_ids) for o in outs], [0, 2, 0])
        self.assertEqual(
            [o.input_ids for o in outs],
            [bad1.input_ids, REQUEST.input_ids, bad2.input_ids],
        )
        self.assert_clean(llm)

    def test_loads_by_model_type(self):
        ref = HFBart(model().config)
        ref.config.architectures = ["BartModel"]
        with TemporaryDirectory() as path:
            ref.save_pretrained(path)
            ref.config.architectures = None
            ref.config.save_pretrained(path)
            self.assertIsInstance(
                load_hf_weights(Config(path)), BartForConditionalGeneration
            )

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
        llm = self.engine()
        (result,) = llm.generate([REQUEST], params)
        self.assertEqual(len(result.completion_token_ids), 2)
        self.assertIn(result.completion_token_ids[-1], [2, 3])

    def test_generate_returns_each_batch_and_keeps_nothing(self):
        llm = self.engine()
        params = SamplingParams(max_tokens=2)
        for _ in range(4):
            completions = llm.generate([REQUEST, REQUEST], params)
            self.assertEqual([len(o.completion_token_ids) for o in completions], [2, 2])
            self.assertEqual(
                [o.input_ids for o in completions], [REQUEST.input_ids] * 2
            )
            self.assert_clean(llm)
        self.assertEqual(llm.generate([], params), [])

    def test_generation_releases_global_cross_cache(self):
        llm = self.engine()
        self.addCleanup(set_context)
        llm.generate([REQUEST], SamplingParams(max_tokens=2))
        self.assert_clean(llm)
        self.assertIsNone(get_context().cross_staged)

    def test_exact_block_budget_fits(self):
        llm = self.engine(max_num_seqs=1)
        bs = llm.config.block_size
        (out,) = llm.generate(
            [REQUEST], SamplingParams(max_tokens=bs, min_length=bs + 1)
        )
        self.assertEqual(len(out.completion_token_ids), bs)
        self.assert_clean(llm)

    def test_runner_failure_releases_global_cross_cache(self):
        llm = self.engine()
        self.addCleanup(set_context)
        llm.scheduler.add(Sequence(0, [1], [3, 4, 5], DEFAULTS))
        seqs, copies = llm.scheduler.schedule()
        with (
            patch.object(
                llm.runner, "decode_fn", side_effect=RuntimeError("decode failed")
            ),
            self.assertRaisesRegex(RuntimeError, "decode failed"),
        ):
            llm.runner.run(seqs, copies)
        self.assertIsNone(get_context().cross_staged)

    def test_none_fields_take_checkpoint_defaults(self):
        llm = self.engine()
        llm._sampling_params = SamplingParams(
            max_tokens=3, min_length=10, eos_token_id=2
        ).with_defaults(DEFAULTS)
        (out,) = llm.generate([REQUEST])  # no params: checkpoint defaults as-is
        self.assertEqual(len(out.completion_token_ids), 3)
        filled = SamplingParams(max_tokens=5).with_defaults(llm._sampling_params)
        self.assertEqual((filled.max_tokens, filled.eos_token_id), (5, 2))
        self.assertEqual(filled.min_length, 10)  # unset, not a hard-coded 0
        # A checkpoint's no_repeat_ngram_size reaches a request that leaves it unset
        # (Florence-2-large loops on <s> without its default of 3).
        llm._sampling_params = SamplingParams.from_generation_config(
            GenerationConfig(no_repeat_ngram_size=3, num_beams=2)
        )
        llm.add_request(REQUEST, SamplingParams(max_tokens=2, num_beams=1))
        params = llm.scheduler.waiting[0].sampling_params
        self.assertEqual((params.no_repeat_ngram_size, params.num_beams), (3, 1))
        while not llm.is_finished():
            llm.step()

    def test_beam_search_mixed_beam_counts_match_per_request(self):
        llm = self.engine()
        params = SamplingParams(num_beams=4, eos_token_id=2)
        # Request 0 has 4 live beams, request 1 is on its first step with 1 beam.
        seqs = [Sequence(0, [1, 5 + b], [3], params) for b in range(4)]
        seqs.append(Sequence(1, [1], [3], params))
        for b, seq in enumerate(seqs):
            seq.cum_logprob = -0.1 * b
        scores = torch.log_softmax(torch.randn(len(seqs), 32), dim=-1)

        def picks(candidates):
            return {
                request_id: [(seqs.index(p), t, round(c, 5)) for p, t, c in cands]
                for request_id, cands in candidates.items()
            }

        mixed = picks(llm.runner._beam_search(seqs, scores.clone(), params))
        alone = {
            **picks(llm.runner._beam_search(seqs[:4], scores[:4].clone(), params)),
            **picks(llm.runner._beam_search(seqs[4:], scores[4:].clone(), params)),
        }
        self.assertEqual(mixed, alone)
        self.assertEqual(len(mixed[0]), 8)  # max(2, 1 + 1 eos) * 4 beams

    def test_beam_search_matches_flattened_topk(self):
        llm = self.engine()
        torch.manual_seed(0)
        for eos, vocab in (([2], 32), ([2, 3, 4], 32), ([2], 5)):
            params = SamplingParams(num_beams=4, eos_token_id=eos)
            keep = max(2, 1 + len(eos)) * 4
            seqs = [Sequence(r, [1], [3], params) for r in (0, 0, 0, 0, 1, 2, 2)]
            for seq in seqs:
                seq.cum_logprob = -torch.rand(()).item()
            scores = torch.log_softmax(torch.randn(len(seqs), vocab), dim=-1)
            scores[-1, :2] = float("-inf")  # masked; too few to tie in the top K
            got = llm.runner._beam_search(seqs, scores.clone(), params)

            # Reference: the old full-vocab add + flattened topk per request.
            cum = torch.tensor([seq.cum_logprob for seq in seqs]).unsqueeze(-1)
            total = scores + cum
            for request_id in (0, 1, 2):
                rows = [i for i, seq in enumerate(seqs) if seq.request_id == request_id]
                flat = total[rows].flatten()
                values, indices = torch.topk(flat, min(keep, flat.numel()))
                expected = [
                    (rows[i // vocab], i % vocab, v)
                    for i, v in zip(indices.tolist(), values.tolist())
                ]
                row_of = {id(seq): i for i, seq in enumerate(seqs)}
                actual = [(row_of[id(p)], t, c) for p, t, c in got[request_id]]
                with self.subTest(eos=eos, vocab=vocab, request_id=request_id):
                    self.assertEqual(actual, expected)

    def test_eos_override_and_single_request_out(self):
        llm = self.engine()
        # None takes the checkpoint's EOS (2); [] never stops early.
        for requested_eos, expected_len in ((None, 1), ([], 2), ([2, 3], 1)):
            with self.subTest(requested_eos=requested_eos):
                params = SamplingParams(
                    max_tokens=2,
                    eos_token_id=requested_eos,
                    forced_bos_token_id=2,
                    min_length=1,
                )
                (result,) = llm.generate(REQUEST, params)
                self.assertIsInstance(result, RequestOut)
                self.assertEqual(result.input_ids, REQUEST.input_ids)
                self.assertIsNone(result.pixel_values)
                self.assertEqual(len(result.completion_token_ids), expected_len)
                self.assertEqual(result.completion_token_ids[0], 2)

    def test_failed_generate_leaves_engine_usable(self):
        llm = self.engine()
        self.addCleanup(set_context)
        decode, steps = llm.runner.decode_fn, []

        def flaky(*args):
            # Fail on the second step: blocks, pages and a queued request in flight.
            steps.append(args)
            if len(steps) == 2:
                raise RuntimeError("decode failed")
            return decode(*args)

        for num_beams in (1, 2):
            params = SamplingParams(max_tokens=4, min_length=5, num_beams=num_beams)
            steps.clear()
            with (
                self.subTest(num_beams=num_beams),
                patch.object(llm.runner, "decode_fn", side_effect=flaky),
                self.assertRaisesRegex(RuntimeError, "decode failed"),
            ):
                llm.generate([REQUEST] * 3, params, use_tqdm=False)
            self.assert_clean(llm)
            (out,) = llm.generate([REQUEST], params, use_tqdm=False)
            self.assertEqual(len(out.completion_token_ids), 4)
            self.assert_clean(llm)

    def test_too_long_inputs_are_rejected_before_running(self):
        llm = self.engine(max_num_seqs=2)
        cases = [
            (Request([3] * 129), SamplingParams(max_tokens=2), "encoder prompt"),
            (REQUEST, SamplingParams(max_tokens=129), "max_decoder_len"),
            (REQUEST, SamplingParams(max_tokens=128, num_beams=9), "max_num_seqs"),
        ]
        for request, params, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                llm.add_request(request, params)
            self.assert_clean(llm)
        # The exact limits still run: 128 encoder tokens and 128 new tokens.
        (out,) = llm.generate(
            [Request([3] * 128)], SamplingParams(max_tokens=128, min_length=200)
        )
        self.assertEqual(len(out.completion_token_ids), 128)
        llm = self.engine(num_blocks=1)
        with self.assertRaisesRegex(ValueError, "KV blocks"):
            llm.add_request(REQUEST, SamplingParams(max_tokens=128))
        self.assert_clean(llm)

    def test_kv_pools_are_fixed_by_decoder_and_encoder_len(self):
        llm = self.engine(max_num_seqs=2, max_decoder_len=40, max_encoder_len=10)
        bs = llm.config.block_size
        self.assertEqual(llm.config.num_blocks, 2 * -(-40 // bs))
        pool = llm.runner.decoder_layers[0].encoder_attn.k_cache
        self.assertEqual(pool.shape[0], 2 * -(-10 // bs))
        llm.generate([Request([3] * 10)] * 3, SamplingParams(max_tokens=40))
        self.assertIs(llm.runner.decoder_layers[0].encoder_attn.k_cache, pool)
        self.assert_clean(llm)
        with self.assertRaisesRegex(ValueError, "max_encoder_len"):
            llm.add_request(Request([3] * 11), SamplingParams(max_tokens=2))
        with self.assertRaisesRegex(ValueError, "max_decoder_len must be"):
            self.engine(max_decoder_len=129)

    def test_shared_beams_fit_two_blocks(self):
        llm = self.engine(max_num_seqs=2)
        params = SamplingParams(max_tokens=3, num_beams=2, min_length=4)
        (out,) = llm.generate([REQUEST], params)
        self.assertEqual(len(out.completion_token_ids), 3)
        self.assert_clean(llm)

    def test_copy_budget_preserves_nonwriting_owner(self):
        llm = self.engine()
        params = SamplingParams()
        parent = Sequence(0, [1], [3], params)
        blocks = llm.scheduler.block_manager
        blocks.ensure_capacity(parent)
        parent.num_cached_tokens = 1
        children = [parent.fork(4, 0), parent.fork(5, 0)]
        for child in children:
            blocks.fork(parent, child)
        self.assertEqual(blocks.required_new_blocks(children), 2)
        blocks.deallocate(parent)
        self.assertEqual(blocks.required_new_blocks(children), 1)

    def test_copy_on_write_copies_each_pair(self):
        llm = self.engine()
        blocks = llm.scheduler.block_manager
        parent = Sequence(0, [1], [3], SamplingParams())
        self.assertIsNone(blocks.ensure_capacity(parent))
        parent.num_cached_tokens = 1
        child = parent.fork(4, 0)
        blocks.fork(parent, child)
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
                    load_hf_weights(Config(path))

    def test_relu_matches_reference(self):
        ref = HFBart(model("relu").config).eval()
        with TemporaryDirectory() as path:
            ref.save_pretrained(path)
            ours = load_hf_weights(Config(path))
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
