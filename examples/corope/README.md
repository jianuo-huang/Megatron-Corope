# CoRoPE context pretraining on Megatron Core

This example trains a token-only context expert using Megatron Core DDP,
distributed AdamW and Transformer Engine. The target embedding and vocabulary
head are frozen. Each input window has isolated causal attention.

The upstream base is NVIDIA/Megatron-LM `core_v0.19.2`, commit
`4b4acac9a1d28ea6829c8d4f566d75698a21249d`. Implementation, tests and performance
changes are kept in separate commits. No upstream core kernels are modified.

Imported context/reference code is derived from jianuo-huang/Dmax-DFlash and
sgl-project/SpecForge under the MIT license; see `LICENSES/SpecForge-MIT.txt`.
Megatron-LM retains its original license and history.
