import torch

from caveat.models import library
from caveat.models.discrete.household_discCNN_CVAE_fullattention_interactions import (
    AttentionBlockPairwiseFullDay,
    HouseholdDiscCNNCVAEFullAttentionInteractions,
)


def _loop_reference_forward(
    block: AttentionBlockPairwiseFullDay,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    B, C, M, W = x.shape
    mask_4d = mask.bool().view(B, 1, M, 1).float()

    y = block.down(x)
    d = y.shape[1]
    tokens = y.permute(0, 2, 3, 1).contiguous()  # [B, M, W, d]
    tokens_norm = block.norm1(tokens)
    mask_bool = mask.bool()

    msg = torch.zeros_like(tokens)
    pair_count = torch.zeros(B, M, 1, 1, dtype=tokens.dtype, device=tokens.device)

    for i in range(M):
        valid_i = mask_bool[:, i]
        if not valid_i.any():
            continue
        q = tokens_norm[:, i]
        for j in range(M):
            if i == j:
                continue
            valid_pair = valid_i & mask_bool[:, j]
            if not valid_pair.any():
                continue
            kv = tokens_norm[:, j]
            out_ij, _ = block.mha(q, kv, kv, need_weights=False)
            msg[:, i] += out_ij * valid_pair[:, None, None].to(tokens.dtype)
            pair_count[:, i] += valid_pair[:, None, None].to(tokens.dtype)

    tokens = msg / pair_count.clamp(min=1.0)
    has_pairs = (pair_count > 0).float()
    h = block.norm2(tokens)
    tokens = (tokens + block.ff(h)) * has_pairs

    y = tokens.permute(0, 3, 1, 2).contiguous()
    y = block.up(y)
    if block.use_gate and block.gate is not None:
        alpha = torch.sigmoid(block.gate).view(1, -1, 1, 1)
    else:
        alpha = block.gate_fixed.view(1, -1, 1, 1)
    return (x + alpha * y) * mask_4d


def test_pairwise_fullattention_shapes_and_masking():
    torch.manual_seed(0)
    block = AttentionBlockPairwiseFullDay(
        channels=8,
        attn_dim=8,
        num_heads=2,
        dropout_attn=0.0,
        dropout_ff=0.0,
        dropout_resid=0.0,
        use_gate=True,
    )
    block.eval()
    x = torch.randn(2, 8, 4, 12)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)

    out = block(x, mask)
    assert out.shape == x.shape
    assert torch.allclose(out[:, :, 2:, :], torch.zeros_like(out[:, :, 2:, :]), atol=1e-6)


def test_pairwise_fullattention_size1_is_residual_only():
    torch.manual_seed(1)
    block = AttentionBlockPairwiseFullDay(
        channels=8,
        attn_dim=8,
        num_heads=2,
        dropout_attn=0.0,
        dropout_ff=0.0,
        dropout_resid=0.0,
        use_gate=True,
    )
    block.eval()
    x = torch.randn(1, 8, 3, 10)
    mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    mask_4d = mask.view(1, 1, 3, 1).float()

    out = block(x, mask)
    expected = x * mask_4d
    assert torch.allclose(out, expected, atol=1e-6)


def test_pairwise_fullattention_vectorized_matches_loop_reference():
    torch.manual_seed(2)
    block = AttentionBlockPairwiseFullDay(
        channels=8,
        attn_dim=8,
        num_heads=2,
        dropout_attn=0.0,
        dropout_ff=0.0,
        dropout_resid=0.0,
        use_gate=True,
    )
    block.eval()
    x = torch.randn(2, 8, 5, 9)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

    out_vec = block(x, mask)
    out_loop = _loop_reference_forward(block, x, mask)
    max_abs = (out_vec - out_loop).abs().max().item()
    assert max_abs < 1e-5


def test_fullattention_model_registration_and_forward_smoke():
    assert (
        library["household_discCNN_CVAE_fullattention_interactions"]
        is HouseholdDiscCNNCVAEFullAttentionInteractions
    )

    torch.manual_seed(3)
    model = HouseholdDiscCNNCVAEFullAttentionInteractions(
        in_shape=(16, 6),
        encodings=6,
        latent_dim=4,
        hidden_size=16,
        num_layers=2,
        kernel_size=4,
        stride=2,
        padding=1,
        dropout=0.0,
        max_members=3,
        time_steps=16,
        interaction_emb_dim=4,
        attn_dim=8,
        attn_heads=2,
        attn_ff_mult=2,
        attn_dropout_attn=0.0,
        attn_dropout_ff=0.0,
        attn_dropout_resid=0.0,
        attn_use_gates=True,
        enc_attn_indices=[0, 1],
        dec_attn_indices=[0, 1],
        enc_post_attn=True,
        dec_post_attn=True,
        use_hierarchical=False,
        encoder_conditioning=False,
        decoder_conditioning=False,
        latent_conditioning=False,
        label_cardinalities=[],
        log_attention=False,
        log_gates=False,
    )
    model.eval()
    batch = {
        "x": torch.randint(0, 6, (2, 3, 16)),
        "interactions": torch.randint(0, 2, (2, 3, 16)),
        "mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32),
    }
    outputs = model.forward(batch, return_attn=True)
    log_probs, interaction_logits, mu, logvar, z, mask, attn_maps = outputs
    assert log_probs.shape == (2, 3, 16, 6)
    assert interaction_logits.shape == (2, 3, 16)
    assert mu.shape[:2] == (2, 3)
    assert logvar.shape[:2] == (2, 3)
    assert z.shape[:2] == (2, 3)
    assert mask.shape == (2, 3)
    assert isinstance(attn_maps, list)
