"""Module implementation."""
import torch
import torch.nn.functional as F


def scale_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive; use greedy sampling for zero temperature")
    return logits / temperature


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return a normalized sampling distribution at the requested temperature."""
    return F.softmax(scale_logits(logits, temperature), dim=-1)