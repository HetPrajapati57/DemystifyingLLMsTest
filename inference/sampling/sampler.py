"""Sampling from temperature-scaled, filtered vocabulary logits."""
import torch
from typing import Optional, List, Union
from .temperature import scale_logits
from .top_k import apply_top_k
from .top_p import apply_top_p
from .repetition_penalty import apply_repetition_penalty


class Sampler:
    def __init__(self, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
        if temperature < 0 or top_k < 0 or not 0 <= top_p <= 1 or repetition_penalty <= 0:
            raise ValueError("Invalid sampling parameters")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def sample(self, logits: torch.Tensor,
               generated_tokens: Optional[Union[List[int], torch.Tensor]] = None, generator=None) -> torch.Tensor:
        logits = logits.clone()
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if generated_tokens is not None:
            logits = apply_repetition_penalty(logits, generated_tokens, self.repetition_penalty)
        if self.temperature == 0:
            return logits.argmax(dim=-1)
        logits = scale_logits(logits, self.temperature)
        logits = apply_top_k(logits, self.top_k)
        logits = apply_top_p(logits, self.top_p)
        return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(-1)
