from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    """Per-step metadata shared across decoder attention layers."""

    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    self_metadata: torch.Tensor | None = None
    isa: str | None = None
    cross_staged: list | None = None


_CONTEXT: Context | None = None


def get_context() -> Context:
    """Return the active step's Context."""
    return _CONTEXT


def set_context(**fields) -> None:
    """Set the global Context for the current model forward pass."""
    global _CONTEXT
    _CONTEXT = Context(**fields)
