from dataclasses import dataclass, field


@dataclass
class CompletionOutput:
    """Output data for one generated completion hypothesis."""

    index: int
    text: str
    token_ids: list[int]
    finish_reason: str | None = None
    stop_reason: int | None = None


@dataclass
class RequestOutput:
    """Output data for a completed generation request."""

    request_id: int
    prompt: str | None
    prompt_token_ids: list[int] | None
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = True
    encoder_prompt: str | None = None
    encoder_prompt_token_ids: list[int] | None = None
