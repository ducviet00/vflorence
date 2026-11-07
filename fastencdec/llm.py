from dataclasses import replace

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from .config import Config
from .engine.llm_engine import LLMEngine
from .models.bart import BartForConditionalGeneration
from .models.florence2 import Florence2ForConditionalGeneration
from .outputs import CompletionOutput, RequestOutput
from .sampling_params import SamplingParams
from .utils.loader import load_hf_weights


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
            from transformers import AutoProcessor
            from transformers import Florence2ForConditionalGeneration as HFFlorence2

            self.processor = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor.tokenizer
            self.model = load_hf_weights(
                Florence2ForConditionalGeneration(hf_config),
                model_path,
                hf_model_cls=HFFlorence2,
            ).to(dtype)
            text_config = hf_config.text_config
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=not self.multimodal)

    def _parse_prompt(self, item):
        if isinstance(item, str):
            return item, None, None
        if not isinstance(item, dict):
            raise TypeError(f"prompt must be a str or dict, got {type(item).__name__}")
        text, token_ids = item.get("prompt"), item.get("prompt_token_ids")
        if text is None and token_ids is None:
            raise ValueError("prompt dict needs 'prompt' or 'prompt_token_ids'")
        return text, token_ids, (item.get("multi_modal_data") or {}).get("image")

    @torch.no_grad()
    def generate(
        self,
        prompts,
        sampling_params: SamplingParams | None = None,
        *,
        mm_processor_kwargs: dict | None = None,
    ) -> list[RequestOutput]:
        """Run batched generation on prompts and return one RequestOutput per prompt."""
        if isinstance(prompts, (str, dict)):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        if sampling_params.eos_token_id is None and self.eos_token_id is not None:
            sampling_params = replace(sampling_params, eos_token_id=self.eos_token_id)
        eos_ids = sampling_params.eos_ids()
        decoder_token_ids = [self.decoder_start_token_id]

        records, engine_requests = [], []
        for item in prompts:
            encoder_prompt, encoder_token_ids, image = self._parse_prompt(item)
            if self.multimodal:
                if encoder_token_ids is not None:
                    raise ValueError(
                        "Florence-2 requires a text prompt, not prompt_token_ids"
                    )
                if image is None:
                    raise ValueError(
                        "Florence-2 needs an image, e.g. "
                        '{"prompt": "<CAPTION>", "multi_modal_data": {"image": img}}'
                    )
                inputs = self.processor(
                    text=encoder_prompt,
                    images=image,
                    return_tensors="pt",
                    **(mm_processor_kwargs or {}),
                )
                encoder_token_ids = inputs["input_ids"][0].tolist()
                pixel_values = inputs["pixel_values"][0]
            else:
                if image is not None:
                    raise ValueError(
                        "images are only supported for Florence-2 checkpoints"
                    )
                if encoder_token_ids is None:
                    # Cut long text to the model's max length, like transformers pipelines.
                    encoder_token_ids = self.tokenizer(
                        encoder_prompt,
                        add_special_tokens=True,
                        padding=False,
                        truncation=True,
                    )["input_ids"]
                pixel_values = None

            records.append((encoder_prompt, encoder_token_ids))
            engine_requests.append((encoder_token_ids, decoder_token_ids, pixel_values))

        completions = self.engine.generate(engine_requests, sampling_params)
        prompt_text = self._decode(decoder_token_ids)
        results = []
        for (request_id, token_ids), (encoder_prompt, encoder_token_ids) in zip(
            completions.items(), records
        ):
            finished = bool(token_ids) and token_ids[-1] in eos_ids
            results.append(
                RequestOutput(
                    request_id=request_id,
                    prompt=prompt_text,
                    prompt_token_ids=decoder_token_ids,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=self._decode(token_ids),
                            token_ids=token_ids,
                            finish_reason="stop" if finished else "length",
                            stop_reason=token_ids[-1] if finished else None,
                        )
                    ],
                    encoder_prompt=encoder_prompt,
                    encoder_prompt_token_ids=encoder_token_ids,
                )
            )
        return results
