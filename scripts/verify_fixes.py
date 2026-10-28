"""Local regression checks for transformer math and end-to-end generation.

Run: python scripts/verify_fixes.py
The challenge's tests/ and .github/ directories are left untouched.
"""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from config import ModelConfig, InferenceConfig
from model.attention import scaled_dot_product_attention, create_causal_mask
from model.layers import LayerNorm, ResidualConnection
from model.embeddings import SinusoidalPositionEmbedding
from model.heads import is_tied
from inference.engine.inference_engine import FullTransformerModel, InferenceEngine
from inference.generation import autoregressive_generate, KVCache, StopCriteria
from inference.sampling import apply_top_k, apply_top_p, apply_temperature, apply_repetition_penalty, Sampler
from tokenizer import BPETokenizer
from tokenizer.vocab import Vocabulary
from serving.pipeline import ServingPipeline
from serving.api.request_handler import RequestHandler
from serving.model_loader import ModelLoader
from utils.checkpoint import save_checkpoint
from utils.tensor_utils import to_device


class TransformerRegressionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.config = ModelConfig(d_model=32, n_heads=4, n_layers=2,
                                  vocab_size=50257, max_seq_len=64, dropout=0.0)

    def test_attention_reference(self):
        q, k, v = [torch.randn(2, 4, 5, 8) for _ in range(3)]
        mask = create_causal_mask(5)
        result, weights = scaled_dot_product_attention(q, k, v, mask)
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(result, reference)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 5))
        self.assertEqual(torch.count_nonzero(torch.triu(weights, diagonal=1)).item(), 0)

    def test_layernorm_and_residual_gradients(self):
        x = torch.randn(2, 5, 32, requires_grad=True)
        norm = LayerNorm(config=self.config)
        torch.testing.assert_close(norm(x), F.layer_norm(x, (32,), norm.weight, norm.bias, norm.eps))
        branch = x.square()
        out = ResidualConnection(dropout=0)(x, branch)
        torch.testing.assert_close(out, x + branch)
        out.sum().backward()
        torch.testing.assert_close(x.grad, 1 + 2 * x.detach())

    def test_positions_and_device(self):
        pe = SinusoidalPositionEmbedding(7, 16)(4)
        torch.testing.assert_close(pe[0, 0, 0::2], torch.zeros(4))
        torch.testing.assert_close(pe[0, 0, 1::2], torch.ones(3))
        self.assertEqual(to_device(torch.ones(2), "meta").device.type, "meta")

    def test_sampling_filters(self):
        logits = torch.tensor([[1., 5., 2., 4.], [8., 1., 3., 2.]])
        self.assertEqual(torch.isfinite(apply_top_k(logits, 2)).tolist(),
                         [[False, True, False, True], [True, False, True, False]])
        nucleus = apply_top_p(logits, 0.7)
        self.assertEqual(torch.isfinite(nucleus).tolist(),
                         [[False, True, False, True], [True, False, False, False]])
        torch.testing.assert_close(apply_temperature(logits, 2), logits / 2)
        self.assertEqual(Sampler(temperature=0).sample(logits).tolist(), [1, 0])

    def test_repetition_is_sign_aware_and_per_sequence(self):
        logits = torch.tensor([[4., -2., 6.], [4., -2., 6.]])
        history = torch.tensor([[0, 1], [2, 2]])
        expected = torch.tensor([[2., -4., 6.], [4., -2., 3.]])
        torch.testing.assert_close(apply_repetition_penalty(logits, history, 2), expected)
        torch.testing.assert_close(logits, torch.tensor([[4., -2., 6.], [4., -2., 6.]]))

    def test_tokenizer_and_ranked_merges(self):
        tokenizer = BPETokenizer()
        ids = tokenizer.encode("hello world")
        self.assertEqual(ids[0], tokenizer.vocab.bos_token_id)
        self.assertEqual(tokenizer.decode(ids), "hello world")
        self.assertEqual(tokenizer.decode(ids[1:2]), "hello")
        self.assertEqual(tokenizer.decode([]), "")
        bpe = BPETokenizer(merges=[("b", "c"), ("a", "b"), ("a", "bc")])
        self.assertEqual(bpe.bpe("abc"), ["abc"])
        self.assertEqual(bpe._merge_word(list("abab"), ("a", "b")), ["ab", "ab"])
        first, second = Vocabulary(), Vocabulary()
        first.add_token("private_word")
        self.assertNotIn("private_word", second)

    def test_cache_matches_full_model_and_causality(self):
        for rotary in (False, True):
            config = ModelConfig.from_dict({**self.config.to_dict(), "use_rotary": rotary})
            model = FullTransformerModel(config).eval()
            ids = torch.tensor([[2, 3, 4, 5, 6]])
            with torch.no_grad():
                full = model(ids)
                cache = KVCache(config.n_layers)
                chunks = [model(ids[:, :3], kv_cache=cache), model(ids[:, 3:], kv_cache=cache)]
                torch.testing.assert_close(torch.cat(chunks, 1), full, atol=2e-5, rtol=2e-5)
                changed = ids.clone()
                changed[:, 3:] = 9
                torch.testing.assert_close(model(changed)[:, :3], full[:, :3])
            self.assertEqual(cache.get_seq_length(), 5)
            self.assertTrue(is_tied(model.lm_head, model.embeddings.token_embedding))
            self.assertIsNot(model.blocks[0].attn.q_proj.weight, model.blocks[0].attn.k_proj.weight)
            cache.reset()
            self.assertEqual(cache.get_seq_length(), 0)

    def test_generation_cache_and_prompt_preservation(self):
        model = FullTransformerModel(self.config)
        ids = torch.tensor([[2, 3, 4]])
        cached = autoregressive_generate(model, ids, 4, use_cache=True)
        uncached = autoregressive_generate(model, ids, 4, use_cache=False)
        torch.testing.assert_close(cached, uncached)
        torch.testing.assert_close(cached[:, :3], ids)
        self.assertEqual(cached.shape, (1, 7))
        self.assertFalse(cached.requires_grad)
        self.assertFalse(model.training)
        self.assertTrue(StopCriteria(7).should_stop(7))
        torch.testing.assert_close(autoregressive_generate(model, ids, 0), ids)

    def test_engine_pipeline_and_completion(self):
        model = FullTransformerModel(self.config)
        tokenizer = BPETokenizer()
        cfg = InferenceConfig(max_new_tokens=3, temperature=0, repetition_penalty=1)
        engine = InferenceEngine(self.config, cfg, model)
        prompt = "hello world"
        ids = tokenizer.encode(prompt)
        result = engine.generate_tokens(ids)
        self.assertEqual(result[:len(ids)], ids)
        self.assertEqual(len(result), len(ids) + 3)
        self.assertEqual(engine.generate(prompt, tokenizer), tokenizer.decode(result))
        pipeline = ServingPipeline(model, tokenizer, self.config, cfg)
        self.assertEqual(pipeline.process(prompt), tokenizer.decode(result))
        response = RequestHandler(engine, tokenizer).handle_completion(
            {"prompt": prompt, "max_tokens": 3, "temperature": 0})
        self.assertEqual(response["text"], tokenizer.decode(result[len(ids):]))
        self.assertEqual(response["usage"]["completion_tokens"], 3)
        cfg.temperature = 0.8
        self.assertEqual(engine.generate_tokens(ids), engine.generate_tokens(ids))

    def test_errors_propagate(self):
        engine = InferenceEngine(self.config, InferenceConfig(temperature=0), FullTransformerModel(self.config))
        with self.assertRaises((IndexError, RuntimeError)):
            engine.generate_tokens([self.config.vocab_size], max_new_tokens=1)
        with self.assertRaises(ValueError):
            engine.generate_tokens([], max_new_tokens=1)
        with self.assertRaises(RuntimeError):
            RequestHandler().handle_completion({"prompt": "hello"})

    def test_checkpoint_round_trip(self):
        model = FullTransformerModel(self.config).eval()
        ids = torch.tensor([[2, 3]])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model_weights.pt")
            save_checkpoint(model.state_dict(), path, use_pickle=False)
            from scripts.download_weights import initialize_seeded_weights
            initialized_path = str(Path(directory) / 'initialized.pt')
            initialize_seeded_weights(initialized_path, config=self.config)
            ModelLoader(initialized_path).load_weights(FullTransformerModel(self.config), strict=True)
            loaded = ModelLoader(path).load_weights(FullTransformerModel(self.config), strict=True).eval()
            with torch.no_grad():
                torch.testing.assert_close(loaded(ids), model(ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
