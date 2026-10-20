# Functional commit sequence

Base: NVIDIA/Megatron-LM core_v0.19.2 (`4b4acac9a1d28ea6829c8d4f566d75698a21249d`).

Commits are ordered by dependency; later performance changes can be reviewed individually.

| # | Commit | Change | Original source |
|---|---|---|---|
| 1 | `02c47901ac3e` | add example layout and source licensing | `` |
| 2 | `597d56f7a488` | add document-isolated windows and resumable corpus preparation | `04f69f7` |
| 3 | `a1b488290408` | add the Qwen3 reference model and causal-training contracts | `04f69f7` |
| 4 | `cec0cd25ec3f` | add shared and loss-free routed experts | `04f69f7` |
| 5 | `c461cfb949f4` | add normalized frozen-head cross entropy and feature gradients | `04f69f7` |
| 6 | `09898b06521d` | compose the context model from Megatron Core and TE layers | `04f69f7` |
| 7 | `1d68ef284ebc` | add distributed training and strict optimizer-state resume | `04f69f7` |
| 8 | `46e51bcf9365` | validate target-head identity and checkpoint transfer | `696f5b7` |
| 9 | `62eaa97decad` | limit shared-filesystem status writes to reporting steps | `ceb76c7` |
| 10 | `fa3c2d16b465` | add configurable W&B pretraining and validation metrics | `8f6f217` |
| 11 | `b2298a76d164` | add resumable global shuffled epochs | `5cedb31` |
| 12 | `464ccd75ba4d` | reduce deterministic allocation and diagnostic overhead | `b56e240` |
| 13 | `c4364c60b085` | add opt-in BF16 grouped expert GEMMs | `6e30151` |
| 14 | `b3576036190f` | add shuffle preparation and self-contained GPU verification | `` |

The final documentation commit adds runtime pins, provenance, benchmark evidence, and packaging validation.

Local history starts at an exact shallow upstream commit. A native GitHub fork supplies the full upstream ancestry.
The patch series and incremental bundle require this upstream base when applied to another checkout.
