"""Autoregressive generation with incremental key/value caching."""
import torch
from typing import Optional, Callable
from .kv_cache import KVCache


@torch.no_grad()
def autoregressive_generate(model: torch.nn.Module, input_ids: torch.Tensor,
                            max_new_tokens: int = 20, eos_token_id: Optional[int] = None,
                            sampler_fn: Optional[Callable] = None, use_cache: bool = False) -> torch.Tensor:
    if input_ids.dim() != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must contain a nonempty batch of token sequences")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if hasattr(model, 'eval'):
        model.eval()
    generated = input_ids.clone()
    finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)
    config = getattr(model, "config", None)
    num_layers = getattr(model, "num_layers", getattr(config, "n_layers", 12))
    cache = KVCache(num_layers) if use_cache else None
    for step in range(max_new_tokens):
        model_inputs = generated[:, -1:] if cache is not None and step else generated
        logits = model(model_inputs, kv_cache=cache) if cache is not None else model(model_inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_logits = logits[:, -1, :]
        next_token = sampler_fn(next_logits) if sampler_fn is not None else next_logits.argmax(-1)
        next_token = next_token.reshape(generated.size(0), 1).to(device=generated.device, dtype=generated.dtype)
        if eos_token_id is not None:
            next_token = torch.where(finished[:, None], eos_token_id, next_token)
            finished |= next_token[:, 0] == eos_token_id
        generated = torch.cat((generated, next_token), dim=-1)
        if eos_token_id is not None and finished.all():
            break
    return generated
