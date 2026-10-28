"""Sign-aware repetition penalty for each sequence's token history."""
import torch
from typing import List, Union


def apply_repetition_penalty(logits: torch.Tensor,
                             generated_tokens: Union[List[int], torch.Tensor],
                             penalty: float = 1.2) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    if penalty == 1:
        return logits
    result = logits.clone()
    history = torch.as_tensor(generated_tokens, dtype=torch.long, device=logits.device)
    if history.numel() == 0:
        return result
    if history.dim() <= 1:
        ids = history.unique()
        ids = ids[(ids >= 0) & (ids < logits.size(-1))]
        values = result[..., ids]
        result[..., ids] = torch.where(values < 0, values * penalty, values / penalty)
    else:
        if logits.dim() != 2 or history.size(0) != logits.size(0):
            raise ValueError("Batched histories must match the logits batch")
        for row in range(logits.size(0)):
            ids = history[row].unique()
            ids = ids[(ids >= 0) & (ids < logits.size(-1))]
            values = result[row, ids]
            result[row, ids] = torch.where(values < 0, values * penalty, values / penalty)
    return result
