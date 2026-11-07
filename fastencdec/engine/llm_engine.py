import torch

from ..config import Config
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Request, RequestOut, Sequence

__all__ = ["LLMEngine", "Request", "RequestOut"]


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
        self._next_request_id = 0

    def add_request(self, request: Request, sampling_params: SamplingParams) -> int:
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
        params = sampling_params or self.default_sampling_params
        self._validate(requests, params)

        request_ids = [self.add_request(req, params) for req in requests]
        outputs: dict[int, RequestOut] = {}
        while not self.is_finished():
            for request_id, out in self.step():
                outputs[request_id] = out
        return [outputs[r] for r in request_ids]

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
