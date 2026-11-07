from collections import defaultdict, deque

from ..config import Config
from .block_manager import BlockManager
from .sequence import RequestOut, Sequence


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
        self.waiting.append(seq)

    def remove(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.block_manager.deallocate(seq)

    def schedule(self) -> tuple[list[Sequence], list[tuple[int, int]]]:
        """Admit waiting sequences up to max_num_seqs, allocate KV blocks, and return (running, copies)."""
        used = sum(
            {
                seq.request_id: seq.sampling_params.num_beams for seq in self.running
            }.values()
        )
        while (
            self.waiting
            and used + self.waiting[0].sampling_params.num_beams <= self.max_num_seqs
        ):
            seq = self.waiting.popleft()
            self.running.append(seq)
            used += seq.sampling_params.num_beams

        seqs = list(self.running)
        if (
            self.block_manager.required_new_blocks(seqs)
            > self.block_manager.num_free_blocks
        ):
            raise RuntimeError("KV cache is full; increase num_blocks")
        copies = [
            copy
            for seq in seqs
            if (copy := self.block_manager.ensure_capacity(seq)) is not None
        ]
        return seqs, copies

    def postprocess(
        self, seqs: list[Sequence], sample_out
    ) -> list[tuple[int, RequestOut]]:
        for seq in seqs:
            seq.num_cached_tokens += 1
        if seqs and seqs[0].sampling_params.num_beams > 1:
            return self._postprocess_beam(seqs, sample_out)
        return self._postprocess_greedy(seqs, sample_out)

    def _postprocess_greedy(
        self, seqs: list[Sequence], token_ids: list[int]
    ) -> list[tuple[int, RequestOut]]:
        outputs = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if seq.is_finished:
                self.remove(seq)
                outputs.append((seq.request_id, seq.to_output()))
        return outputs

    def _postprocess_beam(
        self, seqs: list[Sequence], topk_candidates: dict[int, list]
    ) -> list[tuple[int, RequestOut]]:
        outputs = []
        for request_id, candidates in topk_candidates.items():
            sampling_params = candidates[0][0].sampling_params
            num_beams = sampling_params.num_beams
            early_stopping = sampling_params.early_stopping
            finished_beams = self._finished_beams[request_id]
            running_beams = []

            # Split top-K continuations into finished (only if rank < num_beams) and running beams.
            for rank, (parent, token_id, cum_logprob) in enumerate(candidates):
                child = parent.fork(token_id, cum_logprob)
                if child.is_finished:
                    if rank < num_beams:
                        finished_beams.append(child)
                elif len(running_beams) < num_beams:
                    running_beams.append((parent, child))

            # Keep top num_beams finished beams ranked by length-penalized score.
            finished_beams.sort(key=lambda seq: seq.score, reverse=True)
            del finished_beams[num_beams:]

            # Stop when no running beams remain or the early-stop heuristic is satisfied.
            best_running_score = max(
                (child.score for _, child in running_beams), default=float("-inf")
            )
            is_early_stop_satisfied = len(finished_beams) >= num_beams and (
                early_stopping or finished_beams[-1].score >= best_running_score
            )
            if not running_beams or is_early_stop_satisfied:
                self._finished_beams.pop(request_id, None)
                outputs.append((request_id, finished_beams[0].to_output()))
            else:
                # Share parent KV blocks for the next iteration's running beams.
                for parent, child in running_beams:
                    self.block_manager.fork(parent, child)
                    self.running.append(child)

        for seq in seqs:
            self.remove(seq)
        return outputs
