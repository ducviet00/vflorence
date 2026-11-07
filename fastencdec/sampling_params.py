from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(slots=True)
class SamplingParams:
    """Decoding parameters; num_beams == 1 uses greedy, num_beams > 1 uses beam search."""

    max_tokens: int = 64
    num_beams: int = 1
    repetition_penalty: float = 1.0
    eos_token_id: int | list[int] | None = 2
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    forced_bos_token_id: int | None = None
    forced_eos_token_id: int | list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    early_stopping: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, minimum in (
            ("max_tokens", 1),
            ("num_beams", 1),
            ("no_repeat_ngram_size", 0),
            ("min_length", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not _is_number(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be a positive finite number")
        if not _is_number(self.length_penalty):
            raise ValueError("length_penalty must be a finite number")
        if type(self.early_stopping) is not bool:
            raise ValueError("early_stopping must be True or False")
        # Token ids: single id, id or list, or list only.
        if self.forced_bos_token_id is not None and not _is_id(
            self.forced_bos_token_id
        ):
            raise ValueError("forced_bos_token_id must be a single token ID")
        for name in ("eos_token_id", "forced_eos_token_id"):
            value = getattr(self, name)
            if value is not None and not (_is_id(value) or _is_id_list(value)):
                raise ValueError(f"{name} must be a token ID or a list of token IDs")
        for name in ("suppress_tokens", "begin_suppress_tokens"):
            value = getattr(self, name)
            if value is not None and not _is_id_list(value):
                raise ValueError(f"{name} must be a list of token IDs")

    def eos_ids(self) -> set[int]:
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
        eos_token_id: int | list[int] | None = 2,
        decoder_prompt_len: int = 1,
    ) -> "SamplingParams":
        val = lambda x, default: default if x is None else x
        max_tokens = val(
            config.max_new_tokens,
            config.max_length - decoder_prompt_len
            if config.max_length is not None
            else 64,
        )

        return cls(
            max_tokens=max_tokens,
            num_beams=val(config.num_beams, 1),
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
            # shortcut: transformers' "never" is treated as False (the length
            # heuristic); add it if a checkpoint needs it.
            early_stopping=config.early_stopping is True,
        )


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)


def _is_id(value) -> bool:
    return type(value) is int and value >= 0


def _is_id_list(value) -> bool:
    return isinstance(value, list) and all(_is_id(token) for token in value)
