import torch

from nanochat_muon_lab.muonclip import (
    apply_muonclip_weights_,
    clip_factors,
    rms_normalize_last_dim,
)


def test_clip_factors():
    x = torch.tensor([50.0, 100.0, 200.0])
    got = clip_factors(x, tau=100.0)
    assert torch.allclose(got, torch.tensor([1.0, 1.0, 0.5]))


def test_mha_balanced_clip_scales_qk_product():
    q = torch.ones(4, 3)
    k = torch.ones(4, 3)
    gamma = apply_muonclip_weights_(
        q, k, torch.tensor([400.0, 100.0]), num_q_heads=2, tau=100.0
    )
    # head 0: sqrt(.25)=.5 on Q and K; head 1 unchanged
    assert torch.allclose(gamma, torch.tensor([0.25, 1.0]))
    assert torch.allclose(q[:2], torch.full((2, 3), 0.5))
    assert torch.allclose(k[:2], torch.full((2, 3), 0.5))
    assert torch.allclose(q[2:], torch.ones(2, 3))


def test_gqa_query_only_does_not_touch_shared_k():
    q = torch.ones(8, 3)  # 4 Q heads, head_dim=2
    k = torch.ones(4, 3)  # 2 KV heads, head_dim=2
    k0 = k.clone()
    gamma = apply_muonclip_weights_(
        q, k, torch.tensor([200.0, 100.0, 400.0, 50.0]),
        num_q_heads=4, num_kv_heads=2, tau=100.0, gqa_mode="query_only"
    )
    assert torch.allclose(k, k0)
    assert torch.allclose(gamma, torch.tensor([0.5, 1.0, 0.25, 1.0]))
    assert torch.allclose(q[:2], torch.full((2, 3), 0.5))
    assert torch.allclose(q[4:6], torch.full((2, 3), 0.25))


def test_qk_rms_norm_is_approximately_invariant_to_positive_head_scaling():
    torch.manual_seed(0)
    x = torch.randn(3, 11) * 10.0  # stay far above epsilon-dominated regime
    y1 = rms_normalize_last_dim(x)
    y2 = rms_normalize_last_dim(0.2 * x)
    assert torch.allclose(y1, y2, atol=2e-4, rtol=2e-4)


def test_muonclip_rejects_nonfinite_negative_and_shape_mismatch():
    import pytest
    with pytest.raises(ValueError, match="NaN/Inf"):
        clip_factors(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="nonnegative"):
        clip_factors(torch.tensor([-1.0]))
    with pytest.raises(ValueError, match="input dimensions"):
        apply_muonclip_weights_(
            torch.ones(4, 3), torch.ones(4, 5), torch.ones(2), num_q_heads=2
        )
    with pytest.raises(ValueError, match="divisible"):
        apply_muonclip_weights_(
            torch.ones(6, 3), torch.ones(4, 3), torch.ones(3),
            num_q_heads=3, num_kv_heads=2,
        )
