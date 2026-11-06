from collections import defaultdict
from dataclasses import replace

import torch

from ..config import Config
from ..layers.logits_processor import LogitsProcessor
from ..utils.context import set_context
from .block_manager import BlockManager
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence


def _length_penalized(seq: Sequence, length_penalty: float) -> float:
    """Compute the length-normalized cumulative log probability of a sequence."""
    return seq.cum_logprob / (max(seq.num_generated, 1) ** length_penalty)


def select_beam_candidates(
    stopped: list[bool], num_beams: int
) -> tuple[list[int], list[int]]:
    """Split ranked candidates into finished (if in top num_beams) and live indices."""
    finished = [r for r, stop in enumerate(stopped) if stop and r < num_beams]
    live = [r for r, stop in enumerate(stopped) if not stop][:num_beams]
    return finished, live


class LLMEngine:
    """Step-level execution engine driving scheduling, forward passes, greedy, and beam search."""

    def __init__(self, model, config: Config, eos_token_id: int | None = None):
        self.block_manager = BlockManager(config.num_blocks, config.block_size)
        self.runner = ModelRunner(
            model,
            self.block_manager,
            config.dtype,
            compile=config.compile,
            compile_mm_encoder=config.compile_mm_encoder,
        )
        self.scheduler = Scheduler(config)
        self.eos_token_id = eos_token_id
        self.results: dict[int, list[int]] = {}
        self._finished_beams: dict[int, list[Sequence]] = defaultdict(list)
        self._logits_processor = None
        self._next_request_id = 0

    def generate(self, requests, params) -> list[int]:
        """Enqueue a batch of requests, run until all complete, and return their request IDs."""
        if not self.is_finished():
            raise RuntimeError("finish pending requests before calling generate")
        params.validate()
        requests = list(requests)
        self.results.clear()
        if not requests:
            return []
        if params.eos_token_id is None and self.eos_token_id is not None:
            params = replace(params, eos_token_id=self.eos_token_id)

        prompt_len = len(requests[0][1])
        if any(len(decoder_ids) != prompt_len for _, decoder_ids, _ in requests):
            raise ValueError("decoder prompt length must be uniform across the batch")
        if prompt_len != 1:
            raise ValueError("the decoder prompt must be a single token")
        self._logits_processor = LogitsProcessor(
            params,
            vocab_size=self.runner.model.lm_head.out_features,
            max_length=prompt_len + params.max_tokens,
            begin_index=prompt_len + int(params.forced_bos_token_id is not None),
        )

        request_ids = []
        try:
            for encoder_token_ids, decoder_token_ids, pixel_values in requests:
                request_id = self._next_request_id
                self._next_request_id += 1
                self.scheduler.add(
                    Sequence(
                        request_id,
                        decoder_token_ids,
                        encoder_token_ids,
                        params,
                        pixel_values,
                    )
                )
                request_ids.append(request_id)
            self.run()
        except BaseException:
            self._abort(request_ids)
            raise
        return request_ids

    def _abort(self, request_ids) -> None:
        """Discard a failed batch, including partially written decoder state."""
        for request_id in request_ids:
            self.runner.clear_request(request_id)
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self._finished_beams.clear()
        self.block_manager.reset()
        self.results.clear()
        self._logits_processor = None
        set_context()

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def step(self) -> None:
        seqs = self.scheduler.schedule()
        logits = self.runner.run(seqs)
        self._postprocess(seqs, logits)

    def run(self) -> None:
        while not self.is_finished():
            self.step()

    def _postprocess(self, seqs, logits) -> None:
        if not seqs:
            return
        if seqs[0].sampling.num_beams > 1:
            self._postprocess_beam(seqs, logits)
        else:
            self._postprocess_greedy(seqs, logits)

    def _postprocess_greedy(self, seqs, logits) -> None:
        params = seqs[0].sampling
        lens = [len(s.token_ids) for s in seqs]
        w = max(lens)
        histories = [[-1] * (w - len(s.token_ids)) + s.token_ids for s in seqs]
        lengths = None if min(lens) == w else torch.tensor(lens, dtype=torch.long)
        scores = self._logits_processor(
            torch.tensor(histories, dtype=torch.long), logits.float(), lengths
        )
        for seq, token in zip(seqs, scores.argmax(dim=-1).tolist()):
            seq.token_ids.append(token)
            if token in params.eos_ids() or seq.num_generated >= params.max_tokens:
                self.scheduler.remove(seq)
                self.block_manager.deallocate(seq)
                self.runner.clear_request(seq.request_id)
                self.results[seq.request_id] = seq.completion_token_ids

    def _postprocess_beam(self, seqs, logits) -> None:
        """Group beam requests by shape and run one batched topk per group."""
        num_beams = seqs[0].sampling.num_beams
        vocab = logits.shape[-1]
        by_request: dict[int, list[int]] = defaultdict(list)
        for index, seq in enumerate(seqs):
            by_request[seq.request_id].append(index)
        groups: dict[tuple[int, int], list] = defaultdict(list)
        for request_id, indices in by_request.items():
            groups[(len(indices), len(seqs[indices[0]].token_ids))].append(
                (request_id, indices)
            )
        for (beam_count, _), group in groups.items():
            rows = [i for _, seq_indices in group for i in seq_indices]
            input_ids = torch.tensor(
                [seqs[i].token_ids for i in rows], dtype=torch.long
            )
            logprobs = self._logits_processor(
                input_ids, torch.log_softmax(logits[rows].float(), dim=-1)
            )
            cumulative = torch.tensor(
                [
                    [seqs[i].cum_logprob for i in seq_indices]
                    for _, seq_indices in group
                ],
                dtype=torch.float32,
            ).unsqueeze(-1)
            scores = (
                logprobs.reshape(len(group), beam_count, vocab) + cumulative
            ).reshape(len(group), -1)
            params = seqs[group[0][1][0]].sampling
            keep = min(max(2, 1 + len(params.eos_ids())) * num_beams, scores.shape[-1])
            top_values, top_indices = torch.topk(scores, keep, dim=-1)
            for (request_id, seq_indices), vals, idxs in zip(
                group, top_values.tolist(), top_indices.tolist()
            ):
                self._beam_advance(
                    request_id,
                    [seqs[i] for i in seq_indices],
                    params,
                    [k // vocab for k in idxs],
                    [k % vocab for k in idxs],
                    vals,
                )

    def _fork(self, parent: Sequence, token: int, score: float) -> Sequence:
        """Create a child beam hypothesis sharing parent's KV blocks with token appended."""
        child = Sequence(
            parent.request_id,
            parent.decoder_prompt_ids,
            parent.encoder_token_ids,
            parent.sampling,
            parent.pixel_values,
        )
        child.token_ids = [*parent.token_ids, token]
        child.num_cached_tokens = parent.num_cached_tokens
        child.cum_logprob = score
        self.block_manager.fork(parent, child)
        return child

    def _beam_advance(self, request_id, beams, params, parents, tokens, top_values):
        """Fork winning finished/live beam continuations, free parents, and check stopping."""
        num_beams = params.num_beams
        eos_ids = params.eos_ids()
        stopped = [
            token in eos_ids or beams[parent].num_generated + 1 >= params.max_tokens
            for parent, token in zip(parents, tokens)
        ]
        finished_ranks, live_ranks = select_beam_candidates(stopped, num_beams)

        # Fork children before freeing parents so shared blocks stay alive.
        new_finished = [
            self._fork(beams[parents[r]], tokens[r], top_values[r])
            for r in finished_ranks
        ]
        live = [
            self._fork(beams[parents[r]], tokens[r], top_values[r]) for r in live_ranks
        ]
        for beam in beams:
            self.scheduler.remove(beam)
            self.block_manager.deallocate(beam)

        finished = self._finished_beams[request_id]
        if new_finished:
            finished.extend(new_finished)
            finished.sort(
                key=lambda s: _length_penalized(s, params.length_penalty), reverse=True
            )
            for s in finished[num_beams:]:
                self.block_manager.deallocate(s)
            del finished[num_beams:]

        if not live or (
            len(finished) >= num_beams
            and _length_penalized(finished[-1], params.length_penalty)
            >= max(s.cum_logprob for s in live)
            / (max(live[0].num_generated, 1) ** params.length_penalty)
        ):
            for s in self._finished_beams.pop(request_id, []) + live:
                self.block_manager.deallocate(s)
            self.runner.clear_request(request_id)
            self.results[request_id] = (
                finished[0].completion_token_ids if finished else []
            )
        else:
            self.scheduler.running.extend(live)
