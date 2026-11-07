from collections import defaultdict, deque

from ..config import Config
from .block_manager import BlockManager
from .sequence import RequestOut, Sequence


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


class Scheduler:
    """Continuous-batching scheduler that admits requests up to max_num_seqs."""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.block_manager = BlockManager(config.num_blocks, config.block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._finished_beams: dict[int, list[Sequence]] = defaultdict(list)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        if seq.sampling.num_beams > self.max_num_seqs:
            raise ValueError(
                f"num_beams ({seq.sampling.num_beams}) exceeds "
                f"max_num_seqs ({self.max_num_seqs})"
            )
        self.waiting.append(seq)

    def remove(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.block_manager.deallocate(seq)

    def schedule(self) -> tuple[list[Sequence], list[tuple[int, int]]]:
        """Admit waiting sequences up to max_num_seqs, allocate KV blocks, and return (running, copies)."""
        used = sum({s.request_id: s.sampling.num_beams for s in self.running}.values())
        while (
            self.waiting
            and used + self.waiting[0].sampling.num_beams <= self.max_num_seqs
        ):
            seq = self.waiting.popleft()
            self.running.append(seq)
            used += seq.sampling.num_beams

        seqs = list(self.running)
        if (
            self.block_manager.required_new_blocks(seqs)
            > self.block_manager.num_free_blocks
        ):
            raise RuntimeError("KV cache is full; increase num_blocks")
        copies = []
        for seq in seqs:
            if (copy := self.block_manager.ensure_capacity(seq)) is not None:
                copies.append(copy)
        return seqs, copies

    def postprocess(
        self, seqs: list[Sequence], sample_out
    ) -> list[tuple[int, RequestOut]]:
        if not seqs:
            return []
        for seq in seqs:
            seq.num_cached_tokens += 1
        if seqs[0].sampling.num_beams > 1:
            return self._postprocess_beam(seqs, sample_out)
        return self._postprocess_greedy(seqs, sample_out)

    def _postprocess_greedy(
        self, seqs: list[Sequence], token_ids: list[int]
    ) -> list[tuple[int, RequestOut]]:
        params = seqs[0].sampling
        eos_ids = params.eos_ids()
        outputs = []
        for seq, token in zip(seqs, token_ids):
            seq.append_token(token)
            if token in eos_ids or seq.num_generated >= params.max_tokens:
                self.remove(seq)
                outputs.append(
                    (
                        seq.request_id,
                        RequestOut(
                            input_ids=seq.encoder_token_ids,
                            completion_token_ids=seq.completion_token_ids,
                            pixel_values=seq.pixel_values,
                        ),
                    )
                )
        return outputs

    def _postprocess_beam(
        self, seqs: list[Sequence], beam_candidates: dict[int, tuple]
    ) -> list[tuple[int, RequestOut]]:
        by_request: dict[int, list[Sequence]] = defaultdict(list)
        for seq in seqs:
            by_request[seq.request_id].append(seq)
        outputs = []
        for request_id, beams in by_request.items():
            parents, tokens, vals = beam_candidates[request_id]
            out = self._beam_advance(
                request_id, beams, beams[0].sampling, parents, tokens, vals
            )
            if out is not None:
                outputs.append((request_id, out))
        return outputs

    def _fork(
        self, parent: Sequence, token: int, score: float, share_blocks: bool = True
    ) -> Sequence:
        """Create a child hypothesis with token appended, sharing parent's KV blocks."""
        child = Sequence(
            parent.request_id,
            parent.decoder_token_ids,
            parent.encoder_token_ids,
            parent.sampling,
            parent.pixel_values,
            parent.mm_hash,
        )
        child.token_ids = [*parent.token_ids, token]
        child.num_cached_tokens = parent.num_cached_tokens
        child.cum_logprob = score
        if share_blocks:
            self.block_manager.fork(parent, child)
        return child

    def _beam_advance(
        self, request_id, beams, params, parents, tokens, top_values
    ) -> RequestOut | None:
        """Fork winning finished/live beam continuations, free parents, and check stopping."""
        num_beams = params.num_beams
        eos_ids = params.eos_ids()
        stopped = [
            token in eos_ids or beams[parent].num_generated + 1 >= params.max_tokens
            for parent, token in zip(parents, tokens)
        ]
        finished_ranks, live_ranks = select_beam_candidates(stopped, num_beams)

        # Finished hypotheses never decode again, so they need no KV blocks.
        new_finished = [
            self._fork(beams[parents[r]], tokens[r], top_values[r], share_blocks=False)
            for r in finished_ranks
        ]
        # Fork live children before freeing parents so shared blocks stay alive.
        live = [
            self._fork(beams[parents[r]], tokens[r], top_values[r]) for r in live_ranks
        ]
        for beam in beams:
            self.remove(beam)

        lp = params.length_penalty
        finished = self._finished_beams[request_id]
        if new_finished:
            finished.extend(new_finished)
            finished.sort(key=lambda s: _length_penalized(s, lp), reverse=True)
            del finished[num_beams:]

        # With num_beams finished, stop at once (early_stopping) or once no live
        # beam can still beat the worst finished one.
        done = not live or (
            len(finished) >= num_beams
            and (
                params.early_stopping
                or _length_penalized(finished[-1], lp)
                >= max(_length_penalized(s, lp) for s in live)
            )
        )
        if done:
            self._finished_beams.pop(request_id, None)
            for s in live:
                self.block_manager.deallocate(s)
            return RequestOut(
                input_ids=beams[0].encoder_token_ids,
                completion_token_ids=(
                    finished[0].completion_token_ids if finished else []
                ),
                pixel_values=beams[0].pixel_values,
            )
        self.running.extend(live)
        return None
