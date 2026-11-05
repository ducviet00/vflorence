"""Benchmark helper checks without downloading a model."""

import contextlib
import io
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from benchmarks import (
    benchmark,
    benchmark_vllm,
    throughput,
    verify_parity,
)


class BenchmarkTests(unittest.TestCase):
    def test_cpu_throughput_profile(self):
        tokenizer = Mock(return_value={"input_ids": [3, 4, 5]})
        tokenizer.decode.return_value = "prompt"
        profiler = MagicMock()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "throughput.py",
                    "--requests",
                    "1",
                    "--max-tokens",
                    "2",
                    "--warmup",
                    "0",
                    "--repeats",
                    "1",
                    "--profile",
                ],
            ),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch.object(throughput, "LLM"),
            patch("torch.profiler.profile", return_value=profiler) as profile,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            throughput.main()
        from torch.profiler import ProfilerActivity

        self.assertEqual(profile.call_args.kwargs["activities"], [ProfilerActivity.CPU])

    def test_vllm_uses_same_workload_definitions(self):
        self.assertIs(benchmark_vllm.make_prompt, benchmark.make_prompt)
        self.assertIs(benchmark_vllm.SCENARIOS, benchmark.SCENARIOS)
        self.assertIs(benchmark_vllm.CONFIGS, benchmark.CONFIGS)

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
