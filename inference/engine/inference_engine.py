"""High-level inference engine coordinating model execution and sampling."""
from typing import Optional, List, Union, Any
import inspect
import torch
import torch.nn as nn
from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.constants import EOS_TOKEN_ID
from inference.generation.kv_cache import KVCache
from inference.generation.beam_search import BeamSearch
from inference.sampling import Sampler


class FullTransformerModel(nn.Module):
    """Modular decoder with shared configuration and optional weight tying."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead, tie_weights
        self.config = config
        self.num_layers = config.n_layers
        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        if config.tie_weights:
            tie_weights(self.lm_head, self.embeddings.token_embedding)

    def forward(self, input_ids: torch.Tensor, kv_cache=None, **kwargs) -> torch.Tensor:
        offset = kv_cache.get_seq_length() if kv_cache is not None else 0
        h = self.embeddings(input_ids, offset=offset)
        return self.lm_head(self.blocks(h, kv_cache=kv_cache, **kwargs))


class InferenceEngine:
    def __init__(self, model_config: Optional[ModelConfig] = None,
                 inference_config: Optional[InferenceConfig] = None,
                 model: Optional[Any] = None,
                 device: Optional[Union[str, torch.device]] = None):
        self.model_config = model_config or getattr(model, "config", None) or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = model if model is not None else FullTransformerModel(self.model_config)
        self._prepare_model()

    def _prepare_model(self) -> None:
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def generate_tokens(self, prompt_tokens: List[int], max_new_tokens: Optional[int] = None,
                        temperature: Optional[float] = None, top_k: Optional[int] = None,
                        top_p: Optional[float] = None) -> List[int]:
        self._prepare_model()
        cfg = self.inference_config
        max_tokens = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        if max_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        tokens = list(prompt_tokens)
        if max_tokens == 0:
            return tokens
        if not tokens:
            raise ValueError("Generation requires at least one prompt token")
        if len(tokens) + max_tokens > self.model_config.max_seq_len:
            raise ValueError("Prompt and generation exceed the model context window")
        sampler = Sampler(temperature=cfg.temperature if temperature is None else temperature,
                          top_k=cfg.top_k if top_k is None else top_k,
                          top_p=cfg.top_p if top_p is None else top_p,
                          repetition_penalty=cfg.repetition_penalty)
        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        stop_ids = set(cfg.stop_token_ids) | {EOS_TOKEN_ID}
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        if cfg.beam_width > 1:
            search = BeamSearch(cfg.beam_width, max_tokens, EOS_TOKEN_ID if cfg.early_stopping else None)
            with torch.no_grad():
                return search.search(self.model, input_ids)[0].tolist()
        forward = self.model.forward if hasattr(self.model, 'forward') else self.model
        parameters = inspect.signature(forward).parameters
        supports_cache = 'kv_cache' in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        cache = KVCache(getattr(self.model, 'num_layers', self.model_config.n_layers)) if cfg.use_kv_cache and supports_cache else None
        with torch.no_grad():
            for step in range(max_tokens):
                model_inputs = input_ids[:, -1:] if cache is not None and step else input_ids
                logits = self.model(model_inputs, kv_cache=cache) if cache is not None else self.model(model_inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token = int(sampler.sample(logits[:, -1, :], generated_tokens=tokens, generator=generator).item())
                tokens.append(next_token)
                input_ids = torch.cat((input_ids, input_ids.new_tensor([[next_token]])), dim=-1)
                if cfg.early_stopping and next_token in stop_ids:
                    break
        return tokens

    def generate(self, prompt: str, tokenizer: Optional[Any] = None,
                 max_new_tokens: Optional[int] = None, **kwargs) -> str:
        if tokenizer is None:
            from tokenizer import BPETokenizer
            tokenizer = BPETokenizer()
        # BOS is optional for text generation. Do not inject a special token
        # belonging to a different vocabulary into a smaller supplied model.
        bos_id = getattr(getattr(tokenizer, "vocab", None), "bos_token_id", None)
        encode_parameters = inspect.signature(tokenizer.encode).parameters
        if bos_id is not None and bos_id >= self.model_config.vocab_size and "add_bos" in encode_parameters:
            prompt_ids = tokenizer.encode(prompt, add_bos=False)
        else:
            prompt_ids = tokenizer.encode(prompt)
        output_ids = self.generate_tokens(prompt_ids, max_new_tokens=max_new_tokens, **kwargs)
        return tokenizer.decode(output_ids)
