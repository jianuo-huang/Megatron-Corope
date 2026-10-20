"""Logging must preserve gradients and handle padding/count-weighted summaries."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pretraining.metrics import evaluation_summary, flatten_metrics, summarize_sums
from pretraining.model import LocalLanguageModel, LocalSpec


class TestLogging(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.model = LocalLanguageModel(
            LocalSpec(32, 16, 8, 1, 2, 16, 4, 8, 1e6, 1e-6), torch.randn(32, 16)
        )
        self.model.chunk_size = 5
        self.ids = torch.randint(32, (5, 7))
        self.labels = torch.randint(32, (5, 7))
        self.labels[0, 2:] = -100
        self.labels[-1, 4:] = -100
        self.sources = torch.tensor([0, 1, 1, 2, 0])

    def test_logging_preserves_loss_and_gradients(self):
        plain = copy.deepcopy(self.model)
        observed = copy.deepcopy(self.model)
        a = plain(self.ids, self.labels)
        b, sums = observed(self.ids, self.labels, sources=self.sources, return_metrics=True)
        a.backward()
        b.backward()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for p, q in zip(plain.parameters(), observed.parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        expected = (
            F.cross_entropy(plain(self.ids).flatten(0, 1), self.labels.flatten(), reduction="none")
            .view(5, 7)
            .detach()
        )
        result = summarize_sums(sums.numpy())
        self.assertEqual(result["valid_tokens"], int((self.labels != -100).sum()))
        np.testing.assert_allclose(
            result["loss_by_position"],
            expected.sum(0).numpy() / (self.labels != -100).sum(0).numpy(),
            rtol=1e-6,
        )
        self.assertAlmostEqual(result["loss"], a.item() / result["valid_tokens"], places=6)
        self.assertFalse(sums.requires_grad)

    def test_uneven_rank_sums_match_global_and_empty_cells_are_missing(self):
        _, total = self.model(self.ids, self.labels, sources=self.sources, return_metrics=True)
        parts = []
        for sl in [slice(0, 1), slice(1, 5)]:
            _, part = self.model(
                self.ids[sl], self.labels[sl], sources=self.sources[sl], return_metrics=True
            )
            parts.append(part)
        torch.testing.assert_close(parts[0] + parts[1], total, rtol=1e-6, atol=1e-6)
        summary = summarize_sums(parts[0].numpy())
        flat = flatten_metrics("train", summary)
        self.assertIsNone(summary["loss_by_position"][3])
        self.assertNotIn("train/loss_by_position/d3", flat)
        self.assertEqual(flat["train/valid_tokens_by_position/d3"], 0)
        self.assertNotIn("train/stackedu/loss", flat)

    def test_eval_uses_fixed_mixture_not_equal_or_count_weighted_domains(self):
        raw = {
            s: dict(
                nll=loss,
                accuracy=0.5,
                tokens=tokens,
                nll_by_depth=[loss] * 7,
                accuracy_by_depth=[0.5] * 7,
                tokens_by_depth=[tokens // 7] * 7,
            )
            for s, loss, tokens in [
                ("fineweb", 1.0, 700),
                ("stackedu", 2.0, 70),
                ("finemath", 4.0, 140),
            ]
        }
        result = evaluation_summary(raw)
        self.assertAlmostEqual(result["loss"], 1.8)
        np.testing.assert_allclose(result["loss_by_position"], [1.8] * 7)
        self.assertEqual(flatten_metrics("eval", result)["eval/loss_by_position/d0"], 1.8)
        raw["fineweb"]["nll_by_depth"][6] = None
        raw["fineweb"]["accuracy_by_depth"][6] = None
        missing = evaluation_summary(raw)
        self.assertIsNone(missing["loss_by_position"][6])
        self.assertNotIn("eval/loss_by_position/d6", flatten_metrics("eval", missing))

    def test_resume_accepts_only_logging_changes(self):
        spec = importlib.util.spec_from_file_location("logging_cli", ROOT / "reference_train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        old = {"world_size": 8, "training": {"lr": 0.0006, "seed": 42}}
        new = copy.deepcopy(old)
        new.update(logging_schema_version=2)
        new["training"]["log_position_metrics"] = True
        self.assertEqual(mod.resume_identity(old), mod.resume_identity(new))
        new["training"]["lr"] = 0.001
        self.assertNotEqual(mod.resume_identity(old), mod.resume_identity(new))


if __name__ == "__main__":
    unittest.main()
