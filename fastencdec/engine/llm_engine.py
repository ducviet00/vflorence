import logging
import math

import torch

from ..config import Config
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Request, RequestOut, Sequence

__all__ = ["LLMEngine", "Request", "RequestOut"]

logger = logging.getLogger(__name__)


class LLMEngine:
    """Step-level execution engine driving scheduling, forward passes, greedy, and beam search."""

    def __init__(self, model: str, **kwargs):
        config = Config(model, **kwargs)
        self.config = config
        self.runner = ModelRunner(config)
        self.scheduler = Scheduler(config)
        self.decoder_start_token_id = config.text_config.decoder_start_token_id
        self.eos_token_id = config.text_config.eos_token_id
        self._sampling_params = SamplingParams.from_generation_config(
            config.generation_config, self.eos_token_id
        )
        self._next_request_id = 0

    def add_request(
        self,
        request: Request,
        sampling_params: SamplingParams | None = None,
    ) -> int:
        if sampling_params is None:
            sampling_params = self._sampling_params
        else:
            sampling_params = sampling_params.with_defaults(self._sampling_params)
        self._validate(request, sampling_params)
        request_id = self._next_request_id
        self._next_request_id += 1
        self.scheduler.add(
            Sequence(
                request_id,
                [self.decoder_start_token_id],
                request.input_ids,
                sampling_params,
                request.pixel_values,
            )
        )
        return request_id

    def step(self) -> list[tuple[int, RequestOut]]:
        seqs, copies = self.scheduler.schedule()
        sample_out = self.runner.run(seqs, copies)
        outputs = self.scheduler.postprocess(seqs, sample_out)
        for request_id, _ in outputs:
            self.runner.clear_request(request_id)
        return outputs

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

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

        slots: list[int | RequestOut] = []
        for request in requests:
            try:
                slots.append(self.add_request(request, sampling_params))
            except ValueError as e:
                logger.error("request rejected: %s", e)
                slots.append(RequestOut(request.input_ids, [], request.pixel_values))
        outputs: dict[int, RequestOut] = {}
        while not self.is_finished():
            for request_id, out in self.step():
                outputs[request_id] = out
        self._log_memory()
        return [outputs[s] if isinstance(s, int) else s for s in slots]

    def _log_memory(self) -> None:
        mm_cache = self.runner.mm_cache
        mm_bytes = sum(emb.nbytes for emb in mm_cache.values())
        mb = 1024**2
        logger.info(
            "mm cache %.1f MB (%d/%d images)",
            mm_bytes / mb,
            len(mm_cache),
            self.config.mm_cache_size,
        )

    def _validate(self, request: Request, sampling_params: SamplingParams) -> None:
        """Reject a request that would crash or run out of KV cache mid-step."""
        if request.pixel_values is not None and self.runner.image_fn is None:
            raise ValueError(
                "pixel_values are only supported for Florence-2 checkpoints"
            )

        active = self.scheduler.waiting or self.scheduler.running
        if active and active[0].sampling_params != sampling_params:
            raise ValueError("all active requests must share the same SamplingParams")
        if sampling_params.max_tokens > self.config.max_decoder_len:
            raise ValueError(
                f"max_tokens ({sampling_params.max_tokens}) exceeds the engine's "
                f"max_decoder_len ({self.config.max_decoder_len})"
            )
        if len(request.input_ids) > self.config.max_encoder_len:
            raise ValueError(
                f"encoder prompt has {len(request.input_ids)} tokens; the engine "
                f"supports at most max_encoder_len={self.config.max_encoder_len}"
            )
        if sampling_params.num_beams > self.config.max_num_seqs:
            raise ValueError(
                f"num_beams ({sampling_params.num_beams}) exceeds "
                f"max_num_seqs ({self.config.max_num_seqs})"
            )

        max_num_blocks = sampling_params.num_beams * math.ceil(
            sampling_params.max_tokens / self.config.block_size
        )
        if max_num_blocks > self.config.num_blocks:
            raise ValueError(
                f"one request may need {max_num_blocks} KV blocks but num_blocks is "
                f"{self.config.num_blocks}; increase num_blocks"
            )
