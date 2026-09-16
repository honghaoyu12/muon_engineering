import torch

from nanochat_muon_lab.muon_math import (
    coefficients,
    frobenius_snap,
    gram_newton_schulz_reference,
    merge_heads,
    muoneq_rows,
    muonplus_normalize,
    normuon_rows,
    split_heads,
    standard_newton_schulz,
)


def test_gns_matches_standard_without_restart_fp64():
    torch.manual_seed(0)
    x = torch.randn(7, 13, dtype=torch.float64)
    coeffs = coefficients("keller", 5)
    a = standard_newton_schulz(x, coeffs)
    b = gram_newton_schulz_reference(x, coeffs, reset_after=())
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


def test_gns_restart_is_algebraically_close_fp64():
    torch.manual_seed(1)
    x = torch.randn(8, 17, dtype=torch.float64)
    coeffs = coefficients("dion_ns", 5)
    a = standard_newton_schulz(x, coeffs)
    b = gram_newton_schulz_reference(x, coeffs, reset_after=(2,))
    torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-8)


def test_muoneq_equalizes_row_norms():
    x = torch.tensor([[1.0, 0.0], [0.0, 4.0], [3.0, 4.0]])
    y = muoneq_rows(x)
    row_norms = y.norm(dim=-1)
    assert torch.allclose(row_norms, row_norms.mean().expand_as(row_norms), atol=1e-6)


def test_frob_snap_target():
    torch.manual_seed(2)
    x = torch.randn(5, 11)
    y = frobenius_snap(x)
    assert abs(float(y.norm()) - 5**0.5) < 1e-5


def test_muonplus_row_and_col():
    torch.manual_seed(3)
    x = torch.randn(4, 7)
    yr = muonplus_normalize(x, "row")
    yc = muonplus_normalize(x, "col")
    torch.testing.assert_close(yr.float().norm(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(yc.float().norm(dim=-2), torch.ones(7), atol=1e-5, rtol=1e-5)


def test_normuon_preserves_frobenius_norm():
    torch.manual_seed(4)
    u = torch.randn(6, 9)
    v = torch.zeros(6, 1)
    out, _ = normuon_rows(u, v, 0.9)
    torch.testing.assert_close(out.float().norm(), u.float().norm(), atol=1e-5, rtol=1e-5)


def test_head_split_roundtrip():
    torch.manual_seed(5)
    x = torch.randn(12, 8)
    y, shape = split_heads(x, num_heads=6, heads_per_group=2)
    assert y.shape == (3, 4, 8)
    z = merge_heads(y, shape)
    torch.testing.assert_close(z, x)


def test_fixed_coefficient_schedules_do_not_invent_extra_steps():
    import pytest
    with pytest.raises(ValueError, match="fixed 5-step"):
        coefficients("dion_ns", 6)
    with pytest.raises(ValueError, match="fixed 5-step"):
        coefficients("nanochat_pe", 6)
    assert len(coefficients("dion_ns", 3)) == 3


def test_muoneq_zero_row_remains_zero_and_nonzero_rows_equalize():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 2.0]])
    y = muoneq_rows(x)
    assert torch.equal(y[0], torch.zeros_like(y[0]))
    nonzero = y[1:].norm(dim=-1)
    torch.testing.assert_close(nonzero[0], nonzero[1])


def test_current_keller_ema_is_positive_scalar_equivalent_to_historical_accumulator_for_constant_beta():
    """Document the baseline naming subtlety.

    Historical Muon examples used B_t = beta B_{t-1} + G_t and the Nesterov input
    G_t + beta B_t.  The current KellerJordan/Muon implementation uses the normalized EMA
    M_t = beta M_{t-1} + (1-beta) G_t and input (1-beta)G_t + beta M_t.
    With constant beta and zero initial state, M_t=(1-beta)B_t, so the latter input is exactly
    (1-beta) times the former. A positively homogeneous polar normalization therefore gives the
    same direction in exact arithmetic.
    """
    torch.manual_seed(11)
    beta = 0.95
    B = torch.zeros(5, 9, dtype=torch.float64)
    M = torch.zeros_like(B)
    for _ in range(4):
        G = torch.randn_like(B)
        B = beta * B + G
        M = beta * M + (1.0 - beta) * G
        historical = G + beta * B
        current = (1.0 - beta) * G + beta * M
        torch.testing.assert_close(M, (1.0 - beta) * B, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(current, (1.0 - beta) * historical, rtol=1e-12, atol=1e-12)
