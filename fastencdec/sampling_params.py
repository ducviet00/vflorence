from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(slots=True)
class SamplingParams:
    """Decoding parameters; temperature 0 uses greedy, num_beams > 1 uses beam search."""

    max_tokens: int = 64
    num_beams: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    eos_token_id: int | list[int] | None = None
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    forced_bos_token_id: int | None = None
    forced_eos_token_id: int | list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "max_tokens",
            "num_beams",
            "top_k",
            "no_repeat_ngram_size",
            "min_length",
        ):
            value = getattr(self, name)
            minimum = 1 if name in ("max_tokens", "num_beams") else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("temperature", "top_p", "repetition_penalty", "length_penalty"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        for name in (
            "eos_token_id",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "suppress_tokens",
            "begin_suppress_tokens",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if name in ("suppress_tokens", "begin_suppress_tokens") and not isinstance(
                value, list
            ):
                raise ValueError(f"{name} must be a list of token IDs")
            ids = value if isinstance(value, list) else [value]
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError(f"{name} must contain non-negative integer token IDs")
            if name == "forced_bos_token_id" and isinstance(value, list):
                raise ValueError(f"{name} must be a single token ID")

    def eos_ids(self) -> set[int]:
        """Return EOS token IDs normalized as a set."""
        if self.eos_token_id is None:
            return set()
        return (
            {self.eos_token_id}
            if isinstance(self.eos_token_id, int)
            else set(self.eos_token_id)
        )

    @classmethod
    def from_generation_config(
        cls,
        config,
        eos_token_id: int | None = None,
        decoder_prompt_len: int = 1,
    ) -> "SamplingParams":
        """Build SamplingParams from a HuggingFace GenerationConfig."""
        val = lambda x, default: default if x is None else x
        temperature = 0.0 if not config.do_sample else val(config.temperature, 1.0)
        if config.max_new_tokens is not None:
            max_tokens = config.max_new_tokens
        elif config.max_length is not None:
            max_tokens = config.max_length - decoder_prompt_len
        else:
            max_tokens = 64

        return cls(
            max_tokens=max_tokens,
            num_beams=val(config.num_beams, 1),
            temperature=temperature,
            # HF ignores the sampling warpers when do_sample is off.
            top_p=val(config.top_p, 1.0) if config.do_sample else 1.0,
            top_k=val(config.top_k, 0) if config.do_sample else 0,
            repetition_penalty=val(config.repetition_penalty, 1.0),
            length_penalty=val(config.length_penalty, 1.0),
            no_repeat_ngram_size=val(config.no_repeat_ngram_size, 0),
            min_length=val(config.min_length, 0),
            eos_token_id=val(config.eos_token_id, eos_token_id),
            forced_bos_token_id=config.forced_bos_token_id,
            forced_eos_token_id=config.forced_eos_token_id,
            suppress_tokens=list(config.suppress_tokens)
            if config.suppress_tokens is not None
            else None,
            begin_suppress_tokens=list(config.begin_suppress_tokens)
            if config.begin_suppress_tokens is not None
            else None,
        )
