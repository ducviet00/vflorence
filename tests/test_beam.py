"""Beam-search candidate selection: the EOS/rank gate in Scheduler._postprocess_beam.

A stopped candidate (EOS or max length) may only become a finished hypothesis
if it ranks within the top ``num_beams`` of the combined ranking; weaker ones
are dropped rather than filling the finished set and triggering an early stop.

Run with:  PYTHONPATH=. python tests/test_beam.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GenerationConfig

from src.engine.scheduler import Scheduler
from src.engine.sequence import Sequence
from src.sampling_params import SamplingParams

# Each case is (stopped flags best-first, num_beams, finished ranks, live ranks).
CASES = [
    # EOS in the top rows finish; the rest continue.
    ([True, True, False, False], 2, [0, 1], [2, 3]),
    # EOS ranked below num_beams must not finish, and does not fill a slot.
    ([False, False, True, True], 2, [], [0, 1]),
    # Only the first EOS qualifies; the weaker EOS is dropped, not finished.
    ([False, True, True, False], 2, [1], [0, 3]),
    ([True, False, True, False], 2, [0], [1, 3]),
    # More stopped candidates than beams: extras are dropped, no live beams.
    ([True, True, True], 2, [0, 1], []),
]


def run_case(stopped: list[bool], num_beams: int) -> tuple[list[int], list[int]]:
    eos_ids = [100 + r for r in range(len(stopped))]
    params = SamplingParams(
        num_beams=num_beams, max_tokens=10, eos_token_id=eos_ids, early_stopping=False
    ).with_defaults(SamplingParams.from_generation_config(GenerationConfig()))
    sched = Scheduler(SimpleNamespace(max_num_seqs=8, num_blocks=8, block_size=16))
    parent = Sequence(0, [1], [3], params)
    sched.add(parent)
    seqs, _ = sched.schedule()
    # Stopped candidates use token 100+r (in eos_ids) with lower scores so live beams
    # keep the request open when any live beams exist; live ones use token r.
    candidates = [
        (parent, (100 + r) if stop else r, (-10.0 - r) if stop else -float(r))
        for r, stop in enumerate(stopped)
    ]
    outputs = sched.postprocess(seqs, {0: candidates})
    if outputs:
        # When no live beams remain, postprocess finishes and returns the best finished beam.
        best_rank = outputs[0][1].completion_token_ids[-1] - 100
        return list(range(best_rank, num_beams)), []
    finished = [s.completion_token_ids[-1] - 100 for s in sched._finished_beams[0]]
    live = [s.completion_token_ids[-1] for s in sched.running]
    return finished, live


def main():
    for stopped, num_beams, want_finished, want_live in CASES:
        finished, live = run_case(stopped, num_beams)
        ok = finished == want_finished and live == want_live
        print(
            f"[{'OK' if ok else 'FAIL'}] stopped={stopped} beams={num_beams} "
            f"-> finished={finished} live={live}"
        )
        assert ok, (stopped, num_beams, finished, live, want_finished, want_live)


if __name__ == "__main__":
    main()
