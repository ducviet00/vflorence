"""Benchmark helper checks without downloading a model."""

import unittest
from unittest.mock import Mock

import torch
import verify_parity

from benchmarks import continuous_batching, utils


class BenchmarkTests(unittest.TestCase):
    def test_blocks_for_covers_decoder_tokens(self):
        # 8 seqs x (ceil(513 / 32) = 17 blocks + 1 CoW slack)
        self.assertEqual(utils.blocks_for(8, 512, 32), 8 * 18)
        self.assertEqual(utils.blocks_for(1, 15, 16), 2)

    def test_timed_returns_best_and_last(self):
        calls = iter(range(10))
        seconds, last = utils.timed(lambda: next(calls), warmup=2, repeats=3)
        self.assertEqual(last, 4)
        self.assertGreaterEqual(seconds, 0)

    def test_strip_hf_counts_padding(self):
        out = torch.tensor([[2, 5, 6, 2, 1, 1], [2, 7, 2, 1, 1, 1]])
        rows, pads = utils.strip_hf(out, pad_token_id=1)
        self.assertEqual(rows, [[5, 6, 2], [7, 2]])
        self.assertEqual(pads, [2, 3])

    def test_workload_mixes_all_tasks_per_window(self):
        utils_load = continuous_batching.load_image
        continuous_batching.load_image = Mock(return_value=None)
        try:
            workload = continuous_batching.make_workload(32)
        finally:
            continuous_batching.load_image = utils_load
        for start in range(0, 32, 8):
            tasks = {w[0] for w in workload[start : start + 8]}
            self.assertEqual(tasks, set(continuous_batching.TASKS))
        self.assertEqual(len({w[1] for w in workload}), len(utils.IMAGES))
        self.assertTrue(workload[3][2].endswith(utils.IMAGES[0][1]))

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
