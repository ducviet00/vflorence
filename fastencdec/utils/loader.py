import torch


def fuse_projections(model, state_dict: dict) -> None:
    """Concatenate separate HF projections (e.g. q/k/v_proj -> qkv_proj) along dim 0 in state_dict."""
    for prefix, module in model.named_modules():
        for target, sources in getattr(module, "fused_projections", {}).items():
            for suffix in ("weight", "bias"):
                parts = [f"{prefix}.{source}.{suffix}" for source in sources]
                if all(part in state_dict for part in parts):
                    state_dict[f"{prefix}.{target}.{suffix}"] = torch.cat(
                        [state_dict.pop(part) for part in parts], dim=0
                    )


def load_hf_weights(model, model_path: str, hf_model_cls=None):
    """Load HuggingFace checkpoint weights (with projection fusion) into model."""
    if hf_model_cls is None:
        from transformers import AutoModelForSeq2SeqLM

        hf_model_cls = AutoModelForSeq2SeqLM

    # HF fills missing weights before returning; strict state_dict loading alone misses them.
    hf_model, info = hf_model_cls.from_pretrained(
        model_path, dtype=torch.float32, output_loading_info=True
    )
    errors = {
        key: info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if errors:
        raise RuntimeError(f"incompatible checkpoint: {errors}")
    state_dict = hf_model.state_dict()
    fuse_projections(model, state_dict)

    model.load_state_dict(state_dict, strict=True)
    return model
