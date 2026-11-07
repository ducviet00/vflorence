from dataclasses import dataclass

import torch
from transformers import AutoConfig, GenerationConfig

from .config import Config
from .engine.llm_engine import LLMEngine
from .models.bart import BartForConditionalGeneration
from .models.florence2 import Florence2ForConditionalGeneration
from .sampling_params import SamplingParams
from .utils.loader import load_hf_weights


@dataclass
class Request:
    input_ids: list[int]
    pixel_values: torch.Tensor | None = None


@dataclass
class RequestOut:
    completion_token_ids: list[int]


def _load_generation_config(model_path: str) -> GenerationConfig:
    """Load the checkpoint's GenerationConfig, falling back to an empty default."""
    try:
        return GenerationConfig.from_pretrained(model_path)
    except OSError:
        return GenerationConfig()


class LLM:
    """Offline CPU inference entrypoint for BART and Florence-2 checkpoints."""

    def __init__(
        self,
        model_path: str,
        *,
        num_blocks: int = 512,
        block_size: int | None = None,
        max_num_seqs: int = 32,
        dtype: torch.dtype = torch.float32,
        compile: bool = True,
        compile_mm_encoder: bool = True,
        mm_cache_size: int = 256,
    ):
        config = Config(
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            compile=compile,
            compile_mm_encoder=compile_mm_encoder,
            mm_cache_size=mm_cache_size,
        )
        hf_config = AutoConfig.from_pretrained(model_path)
        self.multimodal = hf_config.model_type == "florence2"

        if self.multimodal:
            from transformers import Florence2ForConditionalGeneration as HFFlorence2

            self.model = load_hf_weights(
                Florence2ForConditionalGeneration(hf_config),
                model_path,
                hf_model_cls=HFFlorence2,
            ).to(dtype)
            text_config = hf_config.text_config
        else:
            self.model = load_hf_weights(
                BartForConditionalGeneration(hf_config), model_path
            ).to(dtype)
            text_config = hf_config

        self.decoder_start_token_id = text_config.decoder_start_token_id
        self.eos_token_id = text_config.eos_token_id
        self.generation_config = _load_generation_config(model_path)
        self.default_sampling_params = SamplingParams.from_generation_config(
            self.generation_config, self.eos_token_id
        )
        self.engine = LLMEngine(self.model, config, eos_token_id=self.eos_token_id)

    @torch.inference_mode()
    def generate(
        self,
        requests: Request | list[Request],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOut]:
        """Run batched generation on requests and return one RequestOut per request."""
        if isinstance(requests, Request):
            requests = [requests]
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        if not self.multimodal and any(r.pixel_values is not None for r in requests):
            raise ValueError(
                "pixel_values are only supported for Florence-2 checkpoints"
            )
        completions = self.engine.generate(requests, sampling_params)
        return [RequestOut(tokens) for tokens in completions.values()]
