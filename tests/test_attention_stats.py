import torch

from caveat.models.discrete.household_discCNN_CVAE_attention import (
    HouseholdDiscCNNCVAEAttention,
)


def test_attention_stats_size1_and_size2():
    # Case size1: one active member -> metrics should be skipped (no valid rows)
    B, W, M, H = 1, 5, 3, 2
    weights = torch.zeros(B * W, H, M, M)
    weights[:, :, 0, 0] = 1.0  # only self
    kpm = torch.tensor([[False, True, True]]).repeat(B * W, 1)

    stats = HouseholdDiscCNNCVAEAttention._compute_attention_stats(weights, kpm)
    assert stats is not None
    assert stats["per_head"] is None
    assert stats["mean"] is None
    assert stats["active_members_mean"] == 1.0
    assert stats["frac_size1"] == 1.0

    # Case size2: two active members -> metrics should exist and be finite
    weights = torch.zeros(B * W, H, M, M)
    weights[:, :, 0, 0] = 0.7
    weights[:, :, 0, 1] = 0.3
    weights[:, :, 1, 0] = 0.4
    weights[:, :, 1, 1] = 0.6
    kpm = torch.tensor([[False, False, True]]).repeat(B * W, 1)

    stats = HouseholdDiscCNNCVAEAttention._compute_attention_stats(weights, kpm)
    assert stats is not None
    assert stats["per_head"] is not None
    assert stats["mean"] is not None
    for v in stats["mean"].values():
        assert torch.isfinite(torch.tensor(v))
test_attention_stats_size1_and_size2()