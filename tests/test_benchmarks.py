"""Benchmark helper checks without downloading a model."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import verify_parity

from benchmarks import continuous_batching, stage_latency, utils


class BenchmarkTests(unittest.TestCase):
    def test_timed_returns_best_and_last(self):
        calls = iter(range(10))
        seconds, last = utils.timed(lambda: next(calls), warmup=2, repeats=3)
        self.assertEqual(last, 4)
        self.assertGreaterEqual(seconds, 0)

    def test_prepare_inputs_shuffles_tasks_and_processes_requests(self):
        processor = Mock(
            return_value={
                "input_ids": torch.tensor([[3, 4]]),
                "pixel_values": torch.zeros(1, 3, 2, 2),
            }
        )
        with patch.object(continuous_batching, "load_image") as load:
            labels, ids, pixels = continuous_batching.prepare_inputs(processor, 128)
        self.assertEqual(load.call_count, len(continuous_batching.IMAGE_URLs))
        self.assertEqual(ids, [[3, 4]] * 128)
        self.assertEqual(len(pixels), 128)
        self.assertEqual(pixels[0].shape, (3, 2, 2))
        # every image x task pair exactly once, not in image order
        self.assertEqual(len(set(labels)), 128)
        self.assertNotEqual(labels, sorted(labels, key=lambda label: label[1]))
        self.assertEqual(
            processor.call_args_list[3].kwargs["text"],
            continuous_batching.TASKS[3] + continuous_batching.IMAGE_URLs[0][1],
        )
        with patch.object(continuous_batching, "load_image"):
            again, _, _ = continuous_batching.prepare_inputs(processor, 128)
        self.assertEqual(labels, again)  # same order for every backend and run

    def test_hf_batches_preprocessed_inputs_and_counts_padding(self):
        args = SimpleNamespace(
            model="unused",
            max_num_req=2,
            max_tokens=4,
            beamsearch=1,
            warmup=0,
            repeats=1,
        )
        tokenizer = Mock(pad_token_id=1)
        tokenizer.pad.side_effect = [
            {
                "input_ids": torch.tensor([[3, 4], [3, 1]]),
                "attention_mask": torch.tensor([[1, 1], [1, 0]]),
            },
            {"input_ids": torch.tensor([[5]])},
        ]
        model = Mock()
        model.generate.side_effect = [
            torch.tensor([[2, 5, 6, 2, 1, 1], [2, 7, 2, 1, 1, 1]]),
            torch.tensor([[2, 8, 2]]),
        ]
        with patch(
            "transformers.Florence2ForConditionalGeneration.from_pretrained",
            return_value=Mock(eval=Mock(return_value=model)),
        ):
            _, rows, pads = continuous_batching.run_hf(
                args,
                [[3, 4], [3], [5]],
                [torch.zeros(3, 2, 2)] * 3,
                torch.bfloat16,
                tokenizer,
            )
        self.assertEqual(rows, [[5, 6, 2], [7, 2], [8, 2]])
        self.assertEqual(pads, [2, 3, 0])
        self.assertEqual(
            tokenizer.pad.call_args_list[0].args[0], {"input_ids": [[3, 4], [3]]}
        )
        self.assertEqual(
            model.generate.call_args_list[0].kwargs["pixel_values"].dtype,
            torch.bfloat16,
        )

    def test_stage_bart_inputs_have_exact_length(self):
        tokenizer = Mock(
            return_value={"input_ids": [4, 5, 6]}, bos_token_id=0, eos_token_id=2
        )
        args = SimpleNamespace(model="unused", enc_len=10, batch=2)
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ):
            ids, pixels = stage_latency.prepare_inputs(args, False, torch.float32)
        self.assertEqual(ids.tolist(), [[0, 4, 5, 6, 4, 5, 6, 4, 5, 2]] * 2)
        self.assertIsNone(pixels)

    def test_fast_stages_use_preprocessed_inputs(self):
        args = SimpleNamespace(model="unused", batch=2, beams=4, max_tokens=32)
        ids = torch.tensor([[0, 4, 2], [0, 5, 2]])
        for pixels in (None, torch.zeros(2, 3, 2, 2)):
            with self.subTest(vision=pixels is not None):
                llm = Mock(decoder_start_token_id=2)
                llm.runner._inductor_ctx.side_effect = nullcontext
                with patch("fastencdec.LLM", return_value=llm):
                    stages, generate = stage_latency.fast_stages(
                        args, ids, pixels, torch.float32
                    )
                self.assertEqual(
                    set(stages),
                    {"vision", "encode"} if pixels is not None else {"encode"},
                )
                stages["encode"]()
                self.assertIs(llm.runner.encode_fn.call_args.args[0], ids)
                generate(7)
                requests, params = llm.generate.call_args.args
                self.assertEqual(requests[0].input_ids, [0, 4, 2])
                self.assertEqual(params.max_tokens, 7)
                self.assertEqual(params.min_length, 8)
                self.assertEqual(params.num_beams, 4)
                if pixels is None:
                    self.assertIsNone(requests[0].pixel_values)
                else:
                    torch.testing.assert_close(requests[0].pixel_values, pixels[0])

    def test_hf_stage_generation_uses_explicit_inputs_and_token_limit(self):
        args = SimpleNamespace(model="unused", beams=4)
        ids = torch.tensor([[0, 4, 2]])
        for pixels in (None, torch.zeros(1, 3, 2, 2)):
            with self.subTest(vision=pixels is not None):
                model = Mock()
                model_class = (
                    "Florence2ForConditionalGeneration"
                    if pixels is not None
                    else "AutoModelForSeq2SeqLM"
                )
                with patch(
                    f"transformers.{model_class}.from_pretrained",
                    return_value=Mock(eval=Mock(return_value=model)),
                ):
                    _, generate = stage_latency.hf_stages(
                        args, ids, pixels, torch.float32
                    )
                generate(7)
                kwargs = model.generate.call_args.kwargs
                self.assertIs(kwargs["input_ids"], ids)
                self.assertEqual(kwargs["max_new_tokens"], 7)
                self.assertEqual(kwargs["min_new_tokens"], 7)
                self.assertEqual(kwargs["num_beams"], 4)
                if pixels is not None:
                    self.assertIs(kwargs["pixel_values"], pixels)
                else:
                    self.assertNotIn("pixel_values", kwargs)

    def test_parity_failures_raise(self):
        with self.assertRaises(AssertionError):
            verify_parity._check_tokens([1, 2], [1, 3], torch.float32)
        with self.assertRaises(AssertionError):
            verify_parity._check_tokens([1, 2], [1], torch.bfloat16)
        self.assertIn(
            "BF16 token differences",
            verify_parity._check_tokens([1, 2], [1, 3], torch.bfloat16),
        )


if __name__ == "__main__":
    unittest.main()
