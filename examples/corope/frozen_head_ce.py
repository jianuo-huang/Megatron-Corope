"""Opt-in frozen-head CE adapter for migration experiments (not deployed).

Return a SUM over valid labels. The training loop owns global-label
normalization and DDP scaling; this adapter never changes those semantics.
Requires liger-kernel 0.8.2; weights must be frozen, but features stay attached.
"""

import math

import torch
from torch.autograd.function import once_differentiable


def frozen_head_ce(features, head, labels):
    if head.requires_grad:
        raise ValueError("This adapter requires a frozen vocabulary head")
    if features.ndim != 2 or head.ndim != 2 or labels.shape != features.shape[:1]:
        raise ValueError("Expected features [T,H], head [V,H], labels [T]")
    if features.shape[1] != head.shape[1] or features.device != head.device:
        raise ValueError("Feature/head shape or device mismatch")
    if not features.is_cuda or labels.device != features.device:
        raise ValueError("The fused backend requires colocated CUDA tensors")
    if labels.dtype != torch.long:
        raise ValueError("Labels must be int64")
    # Empty local batches still provide an attached zero for distributed training.
    if features.shape[0] == 0:
        return features.float().sum() * 0
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

    return LigerFusedLinearCrossEntropyFunction.apply(
        features.to(head.dtype).contiguous(),
        head,
        labels.contiguous(),
        None,
        None,
        -100,
        0.0,
        0.0,
        "sum",
        None,
        False,
        torch.float32,
        False,
        False,
        False,
    )[0]


class _NormalizedFrozenHeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, head, labels, normalizer, chunk_size, compute_gradients):
        import triton
        from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
        from liger_kernel.ops.utils import is_hip

        rows, width = features.shape
        vocab = head.shape[0]
        loss = torch.zeros(rows, device=features.device, dtype=torch.float32)
        grad = torch.empty_like(features) if compute_gradients else None
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            # BF16 logits, FP32 softmax/reduction, normalized gradients rounded
            # to BF16 BEFORE the dgrad GEMM, matching the reference ordering.
            logits = features[start:end] @ head.t()
            target = labels[start:end]
            local_loss = loss[start:end]
            liger_cross_entropy_kernel[(end - start,)](
                X_ptr=logits,
                X_stride=logits.stride(0),
                Y_ptr=target,
                Y_stride=target.stride(0),
                weight_ptr=None,
                loss_ptr=local_loss,
                z_loss_ptr=None,
                loss_stride=1,
                token_accuracy_ptr=None,
                token_accuracy_stride=0,
                predicted_tokens_ptr=None,
                predicted_tokens_stride=0,
                n_cols=vocab,
                n_non_ignore=normalizer,
                sum_non_ignore_weight=normalizer,
                weight_sum=0.0,
                ignore_index=-100,
                lse_square_scale=0.0,
                label_smoothing=0.0,
                reduction='mean',
                softcap=None,
                RETURN_Z_LOSS=False,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                HAS_WEIGHT=False,
                HAS_SOFTCAPPING=False,
                HAS_GRADIENTS=compute_gradients,
                BLOCK_SIZE=min(32768, triton.next_power_of_2(vocab)),
                num_warps=16 if is_hip() else 32,
            )
            if compute_gradients:
                grad[start:end] = logits @ head
        ctx.save_for_backward(grad)
        return loss.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad * grad_output, None, None, None, None, None


def normalized_frozen_head_ce(features, head, labels, *, normalizer, chunk_size=512):
    """SUM(CE) / normalizer; apply normalization exactly once.

    For ordinary averaging DDP, pass total valid labels across ALL ranks AND
    accumulation microbatches divided by world size. The outer loop must NOT
    divide this loss again. Empty/ignored local batches are supported.
    This uses a version-specific Liger 0.8.2 kernel, not its generic loss wrapper.
    Only first-order gradients into features are supported; the head is frozen.
    """
    if head.requires_grad:
        raise ValueError('The vocabulary head must remain frozen')
    if features.ndim != 2 or head.ndim != 2 or labels.shape != features.shape[:1]:
        raise ValueError('Expected features [T,H], head [V,H], labels [T]')
    if features.shape[1] != head.shape[1] or labels.dtype != torch.long:
        raise ValueError('Incompatible feature width or label dtype')
    if not features.is_cuda or not (features.device == head.device == labels.device):
        raise ValueError('Inputs must reside on the same CUDA device')
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('normalizer must be finite and positive')
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    x = features.to(head.dtype).contiguous()
    return _NormalizedFrozenHeadCE.apply(
        x,
        head,
        labels.contiguous(),
        float(normalizer),
        chunk_size,
        torch.is_grad_enabled() and x.requires_grad,
    )
