"""Scale-up contracts: data equivalence, recoverability and token-budget resume."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer as FastTokenizer
from tokenizers import models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from pretraining.data import build_dataset, sha_file, windows
from pretraining.model import tokenizer_identity


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_domino_pretrain_scale")


class Tokenizer:
    eos_token_id = 31

    def encode(self, text, **kwargs):
        return [ord(x) % 30 for x in text]


class TestScale(unittest.TestCase):
    def test_vectorized_packing_equals_reference(self):
        for length in range(150):
            tokens = list(range(length))
            expected = np.asarray(list(windows(tokens)), dtype=np.int32).reshape(-1, 8)
            np.testing.assert_array_equal(prepare.pack(np.asarray(tokens)), expected)
            self.assertEqual(
                int((prepare.pack(np.asarray(tokens))[:, 1:] >= 0).sum()), max(0, length - 1)
            )

    def test_real_target_tokenizer_matches_auto(self):
        # Same implementation in production; exercise Unicode and special tokens.
        from transformers import AutoTokenizer

        target = os.getenv("COROPE_TARGET")
        if not target:
            self.skipTest("Set COROPE_TARGET for the real tokenizer check")
        path = Path(target)
        if not path.exists():
            self.skipTest("Target tokenizer is not installed")
        reference = AutoTokenizer.from_pretrained(path, local_files_only=True)
        direct = FastTokenizer.from_file(str(path / "tokenizer.json"))
        for text in [
            "你好，world!\nprint(x)",
            "∫ x² dx = x³/3",
            "<|im_start|>assistant\nHello",
            "def f(x):\n    return x + 2",
        ]:
            self.assertEqual(
                direct.encode(text, add_special_tokens=False).ids,
                reference.encode(text, add_special_tokens=False),
            )

    def test_builder_resume_and_document_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tok = root / "tokenizer"
            tok.mkdir()
            tokenizer = FastTokenizer(
                models.WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2, "[EOS]": 3}, unk_token="[UNK]")
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(tok / "tokenizer.json"))
            (tok / "config.json").write_text(json.dumps({"eos_token_id": 3}))
            items = []
            for kind, language in [
                ("fineweb", ""),
                ("finemath", ""),
                ("stackedu", "Python"),
                ("stackedu", "JavaScript"),
            ]:
                path = root / f"{kind}{language}.parquet"
                if kind == "stackedu":
                    rows = [
                        dict(
                            blob_id=f"{language}-{i}",
                            length_bytes=512,
                            repo_name=f"{language}-repo-{i//2}",
                            language=language,
                            src_encoding="utf8",
                        )
                        for i in range(500)
                    ]
                else:
                    rows = [
                        dict(
                            text=f"alpha beta {kind} {i} " * (2 + i % 7),
                            url=f"https://example.com/{kind}/{i//2}",
                            id=str(i),
                        )
                        for i in range(500)
                    ]
                pq.write_table(pa.Table.from_pylist(rows), path)
                items.append(
                    dict(
                        kind=kind,
                        repo="test",
                        revision="pinned",
                        path=f"{language}/file.parquet",
                        local=str(path),
                        sha256=sha_file(path),
                    )
                )
            raw = root / "raw.json"
            raw.write_text(json.dumps(items))

            def fake_code(args):
                row = dict(args[0])
                content = row["doc_id"]
                row.update(
                    ids=np.arange(25, dtype=np.int32) % 3,
                    content_sha256=content,
                    content_verified=True,
                )
                return row

            def run(output, resume=False, base=None, budget="6000", raw_path=None):
                argv = [
                    "prepare",
                    "--raw-manifest",
                    str(raw_path or raw),
                    "--tokenizer",
                    str(tok),
                    "--output",
                    str(output),
                    "--web-tokens",
                    budget,
                    "--math-tokens",
                    budget,
                    "--code-tokens-per-language",
                    budget,
                    "--code-fraction",
                    "1",
                    "--workers",
                    "1",
                    "--download-workers",
                    "2",
                ]
                if resume:
                    argv.append("--resume")
                if base:
                    argv.extend(["--base-dataset", str(base)])
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(prepare, "fetch_code", fake_code),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    prepare.main()

            uninterrupted = root / "uninterrupted"
            recovered = root / "recovered"
            run(uninterrupted)
            original_save = prepare.save_json

            def interrupt_at_boundary(path, value):
                original_save(path, value)
                if Path(path).name == "prepare_status.json" and value["stage"] == 1:
                    raise RuntimeError("injected crash after committed stage")

            with patch.object(prepare, "save_json", interrupt_at_boundary):
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    run(recovered)
            # Simulate uncommitted bytes after the last durable checkpoint.
            with (recovered / "fineweb.train.bin").open("ab") as f:
                f.write(b"garbage")
            with (recovered / "documents.jsonl").open("ab") as f:
                f.write(b"uncommitted")
            run(recovered, resume=True)
            for file in uninterrupted.glob("*.bin"):
                self.assertEqual(file.read_bytes(), (recovered / file.name).read_bytes())
            self.assertEqual(
                (uninterrupted / "documents.jsonl").read_bytes(),
                (recovered / "documents.jsonl").read_bytes(),
            )
            records = [
                json.loads(x) for x in (recovered / "documents.jsonl").read_text().splitlines()
            ]
            train = {x["group_id"] for x in records if x["split"] == "train"}
            valid = {x["group_id"] for x in records if x["split"] == "validation"}
            self.assertFalse(train & valid)

            # A held-out math document reappearing at a *training* web URL must
            # not enter the expanded train split, even with a different row ID.
            heldout = next(
                x for x in records if x["source"] == "finemath" and x["split"] == "validation"
            )
            math_rows = pq.read_table(root / "finemath.parquet").to_pylist()
            text = next(x["text"] for x in math_rows if x["id"] == heldout["doc_id"])
            url = next(
                f"https://duplicate/{i}"
                for i in range(1000)
                if prepare.document_split(f"url:https://duplicate/{i}") == "train"
            )
            duplicate = root / "duplicate.parquet"
            pq.write_table(
                pa.Table.from_pylist([dict(text=text, url=url, id="new-duplicate-id")]), duplicate
            )
            extra = dict(
                kind="fineweb",
                repo="test",
                revision="pinned",
                path="-duplicate.parquet",
                local=str(duplicate),
                sha256=sha_file(duplicate),
            )
            extended_raw = root / "extended_raw.json"
            extended_raw.write_text(json.dumps(items + [extra]))
            expanded = root / "expanded"
            run(expanded, base=uninterrupted, budget="8000", raw_path=extended_raw)
            expanded_resume = root / "expanded_resume"
            with patch.object(prepare, "save_json", interrupt_at_boundary):
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    run(expanded_resume, base=uninterrupted, budget="8000", raw_path=extended_raw)
            run(
                expanded_resume,
                resume=True,
                base=uninterrupted,
                budget="8000",
                raw_path=extended_raw,
            )
            for file in expanded.glob("*.bin"):
                self.assertEqual(file.read_bytes(), (expanded_resume / file.name).read_bytes())
                if ".validation." in file.name:
                    self.assertEqual(file.read_bytes(), (uninterrupted / file.name).read_bytes())
                else:
                    self.assertTrue(
                        file.read_bytes().startswith((uninterrupted / file.name).read_bytes())
                    )
            manifest = json.loads((expanded / "manifest.json").read_text())
            self.assertGreater(manifest["fixed_validation"]["heldout_content_skipped"], 0)
            self.assertGreater(manifest["fixed_validation"]["known_documents_skipped"], 0)
            expanded_records = [
                json.loads(x) for x in (expanded / "documents.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                len(expanded_records), len({x["content_sha256"] for x in expanded_records})
            )
            self.assertNotIn(
                heldout["content_sha256"],
                {x["content_sha256"] for x in expanded_records if x["split"] == "train"},
            )

    def test_token_budget_real_process_resume_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.mkdir()
            (target / "tokenizer_config.json").write_text("{}")
            torch.manual_seed(17)
            save_file(
                {"model.embed_tokens.weight": torch.randn(32, 16)}, target / "model.safetensors"
            )
            config = json.loads((ROOT / "configs/reference.json").read_text())
            config.update(vocab_size=32, hidden_size=16)
            config["dflash_config"].update(
                ensemble_context_hidden_dim=8,
                ensemble_context_num_attention_heads=2,
                ensemble_context_intermediate_dim=16,
                emb_dim=4,
            )
            c = root / "config.json"
            c.write_text(json.dumps(config))
            docs = [
                dict(source=s, text=f"{s} sample {i} abcdefghijklmnop", group_id=f"{s}-{i}")
                for s in ("fineweb", "stackedu", "finemath")
                for i in range(100)
            ]
            data = root / "data"
            build_dataset(docs, data, Tokenizer(), tokenizer_identity(target))
            common = [
                sys.executable,
                str(ROOT / "reference_train.py"),
                "--draft-config",
                str(c),
                "--target-tokenizer-embedding",
                str(target),
                "--data",
                str(data),
                "--device",
                "cpu",
                "--loss-backend",
                "chunked",
                "--chunk-size",
                "5",
                "--micro-batch",
                "2",
                "--steps",
                "1",
                "--max-tokens",
                "90",
                "--save-interval",
                "2",
                "--eval-interval",
                "3",
                "--validation-windows",
                "2",
                "--keep-optimizer-states",
                "2",
                "--log-interval",
                "1",
            ]

            def run(out, extra=()):
                r = subprocess.run(
                    common + ["--output", str(out)] + list(extra),
                    env=dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES="",
                        OMP_NUM_THREADS="2",
                        MKL_NUM_THREADS="2",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                if r.returncode:
                    self.fail(r.stdout + r.stderr)

            full = root / "full"
            resumed = root / "resumed"
            run(full)
            run(resumed, ["--stop-after", "2"])
            self.assertFalse(json.loads((resumed / "completion.json").read_text())["finished"])
            run(resumed, ["--resume", str(resumed / "step_000002")])
            a = json.loads((full / "completion.json").read_text())
            b = json.loads((resumed / "completion.json").read_text())
            self.assertTrue(a["finished"] and b["finished"])
            self.assertEqual(a["tokens"], b["tokens"])
            self.assertGreaterEqual(a["tokens"], 90)
            self.assertLess(a["tokens"], 104)
            self.assertGreater(a["completed_steps"], 1)
            for k, v in load_file(str(Path(a["checkpoint"]) / "expert.safetensors")).items():
                torch.testing.assert_close(
                    v,
                    load_file(str(Path(b["checkpoint"]) / "expert.safetensors"))[k],
                    atol=0,
                    rtol=0,
                )
            self.assertEqual(len(list(resumed.glob("step_*/trainer.pt"))), 2)
            self.assertTrue((resumed / "step_000002" / "expert.safetensors").exists())
            final = json.loads((resumed / "status.json").read_text())
            self.assertAlmostEqual(final["lr"], 6e-5)
            self.assertEqual(len(final["loss_by_position"]), 7)
            self.assertEqual(final["metric_scope"], "current_pre_update_optimizer_batch")
            self.assertEqual(sum(final["valid_tokens_by_position"]), final["valid_tokens"])
            self.assertTrue((resumed / "eval_metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
