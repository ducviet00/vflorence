from collections import defaultdict
from dataclasses import dataclass

import torch

from ..config import Config
from ..layers.logits_processor import LogitsProcessor
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence


@dataclass
class Request:
    input_ids: list[int]
    pixel_values: torch.Tensor | None = None


@dataclass
class RequestOut:
    input_ids: list[int]
    completion_token_ids: list[int]
    pixel_values: torch.Tensor | None = None


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

    def __init__(self, model: str, **kwargs):
        config = Config(model, **kwargs)
        self.config = config
        self.runner = ModelRunner(config)
        self.scheduler = Scheduler(config)
        self.max_positions = config.text_config.max_position_embeddings
        self.decoder_start_token_id = config.text_config.decoder_start_token_id
        self.eos_token_id = config.text_config.eos_token_id
        self.default_sampling_params = SamplingParams.from_generation_config(
            config.generation_config, self.eos_token_id
        )
        self.results: dict[int, RequestOut] = {}
        self._finished_beams: dict[int, list[Sequence]] = defaultdict(list)
        self._logits_processor = None
        self._next_request_id = 0

    @torch.inference_mode()
    def generate(
        self,
        requests: Request | list[Request],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOut]:
        """Run batched generation on requests and return one RequestOut per request."""
        if isinstance(requests, Request):
            requests = [requests]
        else:
            requests = list(requests)
        if not requests:
            return []
        params = sampling_params or self.default_sampling_params
        self._validate(requests, params)

        # Lengths count the single decoder start token, as transformers does.
        self._logits_processor = LogitsProcessor(
            params,
            vocab_size=self.runner.model.lm_head.out_features,
            max_length=1 + params.max_tokens,
            begin_index=1 + int(params.forced_bos_token_id is not None),
        )

        decoder_token_ids = [self.decoder_start_token_id]
        request_ids = []
        for req in requests:
            request_id = self._next_request_id
            self._next_request_id += 1
            self.scheduler.add(
                Sequence(
                    request_id,
                    decoder_token_ids,
                    req.input_ids,
                    params,
                    req.pixel_values,
                )
            )
            request_ids.append(request_id)
        self.run()
        return [self.results.pop(r) for r in request_ids]

    def _validate(self, requests, params) -> None:
        """Reject requests that would crash or run out of KV cache mid-batch."""
        if params.max_tokens > self.max_positions:
            raise ValueError(
                f"max_tokens ({params.max_tokens}) exceeds the model's "
                f"max_position_embeddings ({self.max_positions})"
            )
        for req in requests:
            if len(req.input_ids) > self.max_positions:
                raise ValueError(
                    f"encoder prompt has {len(req.input_ids)} tokens; the model "
                    f"supports at most {self.max_positions}"
                )
        # shortcut: checks one request alone; a full batch can still run out.
        # Upgrade to preemption if that shows up in practice.
        bs = self.config.block_size
        max_num_blocks = params.num_beams * (params.max_tokens // bs)
        if max_num_blocks > self.config.num_blocks:
            raise ValueError(
                f"one request may need {max_num_blocks} KV blocks but num_blocks is "
                f"{self.config.num_blocks}; increase num_blocks"
            )

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def step(self) -> None:
        seqs, copies = self.scheduler.schedule()
        logits = self.runner.run(seqs, copies)
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
        eos_ids = params.eos_ids()
        lens = [len(s.token_ids) for s in seqs]
        w = max(lens)
        # Histories differ in length: left-pad with -1, which every logits rule ignores.
        histories = [[-1] * (w - len(s.token_ids)) + s.token_ids for s in seqs]
        scores = self._logits_processor(
            torch.tensor(histories, dtype=torch.long),
            logits.float(),
            torch.tensor(lens, dtype=torch.long),
        )
        for seq, token in zip(seqs, scores.argmax(dim=-1).tolist()):
            seq.token_ids.append(token)
            if token in eos_ids or seq.num_generated >= params.max_tokens:
                self.scheduler.remove(seq)
                self.runner.clear_request(seq.request_id)
                self.results[seq.request_id] = RequestOut(
                    input_ids=seq.encoder_token_ids,
                    completion_token_ids=seq.completion_token_ids,
                    pixel_values=seq.pixel_values,
                )

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
            # Keep enough candidates that num_beams live ones remain after EOS
            # candidates are set aside (same rule as transformers).
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
            self.scheduler.block_manager.fork(parent, child)
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
            self.scheduler.remove(beam)

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
                self.scheduler.block_manager.deallocate(s)
            self.runner.clear_request(request_id)
            self.results[request_id] = RequestOut(
                input_ids=beams[0].encoder_token_ids,
                completion_token_ids=(
                    finished[0].completion_token_ids if finished else []
                ),
                pixel_values=beams[0].pixel_values,
            )
        else:
            self.scheduler.running.extend(live)
