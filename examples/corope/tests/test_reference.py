"""CPU contracts for the standalone public-corpus pretraining pipeline."""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Qwen3Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pretraining.data import SOURCES, WindowData, build_dataset, document_split, windows
from pretraining.model import (
    LocalLanguageModel,
    LocalSpec,
    initialize_draft,
    read_expert,
    tensor_sha,
    transferable,
)


def draft_config():
    c = json.loads((ROOT / "configs/reference.json").read_text())
    c.update(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_target_layers=2,
    )
    c["dflash_config"].update(
        target_layer_ids=[0],
        emb_dim=4,
        ensemble_context_hidden_dim=8,
        ensemble_context_num_attention_heads=2,
        ensemble_context_intermediate_dim=16,
    )
    return Qwen3Config(**c)


def tiny_model():
    spec = LocalSpec.from_draft(draft_config())
    embedding = torch.randn(32, 16)
    model = LocalLanguageModel(spec, embedding)
    identity = {"embedding_sha256": tensor_sha(embedding), "tokenizer_files": {"fake": "test"}}
    return model, identity


class Tokenizer:
    eos_token_id = 31

    def encode(self, text, **kwargs):
        return [ord(c) % 30 for c in text]


class TestPretraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_shared_causality_and_no_condition_gradient(self):
        m, _ = tiny_model()
        ids = torch.randint(30, (3, 7))
        emb = m.token_embedding.weight.clone()
        out = m(ids)
        changed = ids.clone()
        changed[:, 1:] = (changed[:, 1:] + 1) % 30
        torch.testing.assert_close(out[:, 0], m(changed)[:, 0], atol=0, rtol=0)
        out[:, 0].square().sum().backward()
        self.assertIsNone(m.token_embedding.weight.grad)
        self.assertIsNone(m.context_transformer.anchor_init_proj.weight.grad)
        self.assertGreater(m.context_transformer.input_proj.weight.grad.norm().item(), 0)
        torch.testing.assert_close(emb, m.token_embedding.weight, atol=0, rtol=0)
        self.assertTrue(
            all(not p.requires_grad for p in m.context_transformer.anchor_init_proj.parameters())
        )

    def test_shifted_rope_matches_independent_zero_offset_reference(self):
        m, _ = tiny_model()
        ids = torch.randint(30, (2, 7))
        x = m.context_transformer.input_proj(m.token_embedding(ids))
        c = m.context_transformer
        pos = torch.arange(7)[None]
        rot = c.rotary_emb(x, pos)
        for layer in c.layers:
            x = layer(x, attention_mask=None, position_embeddings=rot, use_cache=False)
        expected = c.final_norm(x)
        torch.testing.assert_close(
            c.forward_tokens(m.token_embedding(ids)), expected, atol=1e-6, rtol=1e-5
        )

    def test_chunked_ce_and_microbatch_normalization(self):
        m, _ = tiny_model()
        m.chunk_size = 3
        a = copy.deepcopy(m)
        b = copy.deepcopy(m)
        ids = torch.randint(30, (5, 7))
        labels = torch.randint(30, (5, 7))
        labels[0, 2:] = -100
        labels[-1, 1:] = -100
        n = (labels != -100).sum()
        loss = m(ids, labels) / n
        loss.backward()
        expected = F.cross_entropy(
            a(ids).float().flatten(0, 1), labels.flatten(), ignore_index=-100
        )
        expected.backward()
        torch.testing.assert_close(loss, expected)
        for x, y in zip(ids.split(2), labels.split(2)):
            (b(x, y) / n).backward()
        for p, q, r in zip(m.parameters(), a.parameters(), b.parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(p.grad, r.grad, atol=1e-6, rtol=1e-5)

    def test_windows_label_alignment_tail_and_no_cross_document(self):
        self.assertEqual(
            list(windows(list(range(11)), 8)), [list(range(8)), [7, 8, 9, 10, -1, -1, -1, -1]]
        )
        self.assertEqual(list(windows([4])), [])
        wins = list(windows(list(range(17))))
        labels = [v for row in wins for v in row[1:] if v >= 0]
        self.assertEqual(labels, list(range(1, 17)))

    def test_document_split_mixture_sampler_and_resume_identity(self):
        docs = [
            dict(
                source=s,
                text=f"{s} document {i}: abcdefghijklmnopqrstuvwxyz",
                group_id=f"group-{s}-{i//2}",
                doc_id=f"{s}-{i}",
            )
            for s in SOURCES
            for i in range(300)
        ]
        docs.append(dict(docs[0], source="stackedu", group_id="different"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            meta = build_dataset(docs, root, Tokenizer(), {"fake": "identity"})
            self.assertEqual(meta["duplicates_removed"], 1)
            records = [json.loads(x) for x in (root / "documents.jsonl").read_text().splitlines()]
            sets = {
                s: {x["group_id"] for x in records if x["split"] == s}
                for s in ("train", "validation")
            }
            self.assertFalse(sets["train"] & sets["validation"])
            d = WindowData(root)
            ids, labels, sources = d.sample(20000, seed=42, step=9)
            ids2, labels2, sources2 = WindowData(root).sample(20000, seed=42, step=9)
            np.testing.assert_array_equal(ids, ids2)
            np.testing.assert_array_equal(labels, labels2)
            np.testing.assert_array_equal(sources, sources2)
            counts = np.array([(labels[sources == i] != -100).sum() for i in range(3)])
            np.testing.assert_allclose(counts / counts.sum(), [0.6, 0.2, 0.2], atol=0.012)
            self.assertTrue((ids >= 0).all())
            self.assertTrue((labels == -100).any())
            with (root / "fineweb.train.bin").open("ab") as f:
                f.write(b"1234")
            with self.assertRaisesRegex(ValueError, "checksum"):
                WindowData(root)

    @unittest.skipUnless(
        os.getenv("COROPE_JOINT_SOURCE"), "Set COROPE_JOINT_SOURCE for external joint integration"
    )
    def test_exact_export_transfer_and_untouched_backbone(self):
        m, identity = tiny_model()
        spec = importlib.util.spec_from_file_location(
            "pretrain_test_dflash",
            Path(os.environ["COROPE_JOINT_SOURCE"]) / "specforge/modeling/draft/dflash.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        draft = mod.DFlashDraftModel(draft_config())
        before = {k: v.clone() for k, v in draft.state_dict().items() if not transferable(k)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "expert"
            m.export(root, identity)
            info = initialize_draft(draft, root, identity)
            self.assertTrue(info["loaded_keys"])
            for k, v in before.items():
                torch.testing.assert_close(v, draft.state_dict()[k], atol=0, rtol=0)
            ids = torch.randint(30, (3, 7))
            local = m(ids)
            migrated = draft.embed_proj(
                draft.context_transformer.forward_tokens(m.token_embedding(ids))
            )
            torch.testing.assert_close(local, migrated, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "identity"):
                read_expert(root, m.spec, {"wrong": "embedding"})
            with (root / "expert.safetensors").open("ab") as f:
                f.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_expert(root, m.spec, identity)

    def test_optimizer_resume_preserves_fp32_updates(self):
        modspec = importlib.util.spec_from_file_location(
            "pretrain_cli", ROOT / "reference_train.py"
        )
        mod = importlib.util.module_from_spec(modspec)
        modspec.loader.exec_module(mod)
        m, identity = tiny_model()
        optim = mod.optimizer_for(m, 6e-4, 0.1)
        ids = torch.randint(30, (3, 7))
        labels = torch.randint(30, (3, 7))

        def step(model, opt):
            opt.zero_grad()
            (model(ids, labels) / labels.numel()).backward()
            opt.step()

        step(m, optim)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "expert"
            m.export(root, identity)
            torch.save(optim.state_dict(), Path(tmp) / "optim.pt")
            fresh, _ = tiny_model()
            fresh.token_embedding.weight.data.copy_(m.token_embedding.weight)
            fresh.load_expert(root, identity)
            o2 = mod.optimizer_for(fresh, 6e-4, 0.1)
            o2.load_state_dict(torch.load(Path(tmp) / "optim.pt", weights_only=True))
            step(m, optim)
            step(fresh, o2)
            for k, v in m.state_dict().items():
                if transferable(k):
                    torch.testing.assert_close(v, fresh.state_dict()[k], atol=0, rtol=0)
            for state in o2.state.values():
                self.assertEqual(state["exp_avg"].dtype, torch.float32)
                self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
