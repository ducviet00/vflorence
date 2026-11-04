"""Device checks must work without downloading a model or calling CUDA on CPU."""

import contextlib
import io
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from benchmarks import (
    attn_backend,
    benchmark,
    benchmark_vllm,
    throughput,
    verify_parity,
)


class BenchmarkTests(unittest.TestCase):
    def test_cuda_attention_comparison_rejected_before_loading(self):
        stderr = io.StringIO()
        with (
            patch.object(sys, "argv", ["attn_backend.py", "--device", "cuda"]),
            patch("transformers.AutoTokenizer.from_pretrained") as load,
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as error,
        ):
            attn_backend.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("requires --device cpu", stderr.getvalue())
        load.assert_not_called()

    def test_cpu_throughput_never_synchronizes_cuda(self):
        tokenizer = Mock(return_value={"input_ids": [3, 4, 5]})
        tokenizer.decode.return_value = "prompt"
        profiler = MagicMock()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "throughput.py",
                    "--device",
                    "cpu",
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
            patch(
                "torch.cuda.synchronize", side_effect=AssertionError("CUDA on CPU")
            ) as sync,
            patch("torch.profiler.profile", return_value=profiler) as profile,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            throughput.main()
        sync.assert_not_called()
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
