"""Text generation pipeline using the shared inference engine."""
from typing import Optional, Any, Union
import torch
from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from inference.engine.inference_engine import InferenceEngine


class ServingPipeline:
    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None,
                 config: Optional[ModelConfig] = None,
                 inference_config: Optional[InferenceConfig] = None,
                 device: Optional[Union[str, torch.device]] = None):
        self.config = config or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if tokenizer is None:
            from tokenizer import BPETokenizer
            tokenizer = BPETokenizer()
        self.tokenizer = tokenizer
        self.engine = InferenceEngine(self.config, self.inference_config, model, self.device)
        self.model = self.engine.model

    def process(self, text: str, max_new_tokens: Optional[int] = None) -> str:
        return self.engine.generate(text, tokenizer=self.tokenizer, max_new_tokens=max_new_tokens)
