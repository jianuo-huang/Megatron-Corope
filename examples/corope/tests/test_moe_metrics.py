"""Suppressing diagnostics must never suppress loss-free router updates."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domino_moe import DominoMoE


@pytest.mark.parametrize("counts", [[0, 0, 0, 0], [0, 3, 9, 0], [4, 4, 4, 4]])
def test_router_state_is_independent_of_metric_collection(counts):
    observed = DominoMoE(
        8,
        dict(
            routed_experts=4,
            expert_intermediate_size=8,
            top_k=2,
            routed_scale=1.0,
            bias_update_rate=0.001,
            shared_experts=1,
        ),
    )
    observed.routing_bias.copy_(torch.tensor([0.03, -0.01, -0.02, 0.0]))
    quiet = copy.deepcopy(observed)
    # Exercise both suppressed steps and a later observed step. In particular,
    # ignored/empty batches must not increment the balance-update counter.
    for step in range(4):
        for model in (observed, quiet):
            model.pending_counts.copy_(torch.tensor(counts))
        expected = observed.update_balance()
        actual = quiet.update_balance(collect_metrics=step == 3)
        for key, value in observed.state_dict().items():
            torch.testing.assert_close(value, quiet.state_dict()[key], rtol=0, atol=0)
        assert torch.count_nonzero(quiet.pending_counts) == 0
        if step == 3:
            assert actual.keys() == expected.keys()
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual == {}
