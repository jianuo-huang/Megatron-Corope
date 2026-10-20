"""Compare the fused frozen-head loss with independent dense autograd."""

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frozen_head_ce import normalized_frozen_head_ce


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA frozen-head kernel check')
@pytest.mark.parametrize(
    'rows,chunk_size,all_ignored', [(7, 3, False), (7, 7, False), (7, 3, True), (0, 3, False)]
)
def test_normalized_ce_matches_dense_autograd(
    rows: int, chunk_size: int, all_ignored: bool
) -> None:
    """Include tail chunks, ignored labels, and empty local batches with attached gradients."""
    torch.manual_seed(17)
    features = torch.randn(rows, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    reference = features.detach().clone().requires_grad_()
    head = torch.randn(31, 16, device='cuda', dtype=torch.bfloat16) * 0.1
    labels = torch.randint(31, (rows,), device='cuda')
    labels[::3] = -100
    if all_ignored:
        labels.fill_(-100)
    # A global normalizer may exceed the valid-label count on this rank.
    actual = normalized_frozen_head_ce(
        features, head, labels, normalizer=11.0, chunk_size=chunk_size
    )
    expected = F.cross_entropy((reference @ head.T).float(), labels, reduction='sum') / 11.0
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(features.grad, reference.grad, atol=2e-4, rtol=0.02)
    assert head.grad is None
    assert torch.isfinite(features.grad).all()
