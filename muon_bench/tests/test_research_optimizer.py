"""CPU tests for the high-level research optimizer without requiring a NanoChat checkout."""
import sys
import types

import pytest
import torch

try:
    import nanochat.optim  # noqa: F401
except ModuleNotFoundError:
    nanochat_mod = types.ModuleType("nanochat")
    optim_mod = types.ModuleType("nanochat.optim")

    class _StubMuonAdamW:
        def __init__(self, param_groups):
            self.param_groups = param_groups
            self.state = {}

    optim_mod.MuonAdamW = _StubMuonAdamW
    nanochat_mod.optim = optim_mod
    sys.modules["nanochat"] = nanochat_mod
    sys.modules["nanochat.optim"] = optim_mod

from nanochat_muon_lab.research_optimizer import ResearchMuonAdamW, _defaults


def make_opt():
    return ResearchMuonAdamW.__new__(ResearchMuonAdamW)


def group(**kw):
    g = dict(
        kind="muon", lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.1,
        polar_dtype="fp32",
    )
    _defaults(g)
    g.update(kw)
    return g


def test_polar_dtype_fp32_really_casts_input_to_fp32():
    opt = make_opt()
    g = group(orthogonalizer="standard", coefficients="keller", polar_dtype="fp32")
    x = torch.randn(6, 10, dtype=torch.bfloat16)
    out = opt._orthogonalize(x, g)
    assert out.dtype == torch.float32


def test_group_validation_rejects_bad_enums_and_backend_precision_claims():
    base = group()
    bad = dict(base, orthogonalizer="typo")
    with pytest.raises(ValueError, match="orthogonalizer"):
        ResearchMuonAdamW._validate_group(bad)
    bad = dict(base, orthogonalizer="gns_official", polar_dtype="bf16")
    with pytest.raises(ValueError, match="owns its internal working precision"):
        ResearchMuonAdamW._validate_group(bad)


def test_batched_classic_headwise_is_finite_and_shape_preserving():
    torch.manual_seed(1)
    opt = make_opt()
    g = group(
        orthogonalizer="gns_reference", coefficients="dion_ns", gns_reset_after=(2,),
        num_heads=4, heads_per_group=1, muoneq=True, muonplus_mode="frob_snap",
        normuon=True, normuon_mode="row", beta2=0.9,
    )
    ResearchMuonAdamW._validate_group(g)
    grad = torch.randn(3, 8, 12)
    param = torch.randn(3, 8, 12)
    mom = torch.zeros_like(param)
    var = torch.zeros(3, 8, 1)
    before = param.clone()
    opt._classic_batch(grad, param, mom, var, g)
    assert param.shape == before.shape
    assert torch.isfinite(param).all() and torch.isfinite(mom).all() and torch.isfinite(var).all()
    assert not torch.allclose(param, before)


def test_batched_dion3_works_for_wide_and_tall_matrices():
    torch.manual_seed(2)
    opt = make_opt()
    for shape in [(3, 6, 10), (3, 10, 6)]:
        g = group(
            update_rule="fractional_ef", nesterov=False, ortho_fraction=0.5,
            fraction_lr_compensation=True, orthogonalizer="gns_reference",
            coefficients="dion_ns", gns_reset_after=(2,), normuon=True,
            normuon_mode="row", beta2=0.9,
        )
        ResearchMuonAdamW._validate_group(g)
        grad = torch.randn(*shape)
        param = torch.randn(*shape)
        err = torch.zeros_like(param)
        var = torch.zeros(shape[0], min(shape[-2:]), 1)
        opt._fractional_ef_batch(grad, param, err, var, g)
        assert torch.isfinite(param).all() and torch.isfinite(err).all() and torch.isfinite(var).all()


def test_dion3_error_feedback_decays_only_selected_rows():
    opt = make_opt()
    opt._project = types.MethodType(lambda self, x, g: torch.zeros_like(x), opt)
    grad = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]])
    p = torch.ones_like(grad)
    e = torch.zeros_like(grad)
    g = group(update_rule="fractional_ef", nesterov=False, ortho_fraction=0.5,
              fraction_lr_compensation=False, weight_decay=0.0, momentum=0.9)
    opt._fractional_ef_batch(grad, p, e, None, g)
    # top-2 rows are selected and consumed => multiplied by mu; unselected remain exact residual.
    torch.testing.assert_close(e[0, 0], grad[0, 0] * 0.9)
    torch.testing.assert_close(e[0, 1], grad[0, 1] * 0.9)
    torch.testing.assert_close(e[0, 2:], grad[0, 2:])


def test_dion3_f1_matches_non_nesterov_classic_first_step_direction():
    # On the first step classic EMA differs from E<-E+G by a positive scalar (1-beta); a homogeneous
    # polar map removes that scalar, so f=1 should give the same parameter update when Nesterov/WD/
    # adaptive normalization are off.
    torch.manual_seed(7)
    opt = make_opt()
    grad = torch.randn(2, 6, 10)
    p0 = torch.randn_like(grad)

    classic = group(weight_decay=0.0, nesterov=False, orthogonalizer="gns_reference",
                    coefficients="keller", gns_reset_after=(), normuon=False)
    dion = group(update_rule="fractional_ef", nesterov=False, ortho_fraction=1.0,
                 fraction_lr_compensation=False, weight_decay=0.0,
                 orthogonalizer="gns_reference", coefficients="keller", gns_reset_after=(),
                 normuon=False)
    p1, p2 = p0.clone(), p0.clone()
    m1, e2 = torch.zeros_like(grad), torch.zeros_like(grad)
    opt._classic_batch(grad, p1, m1, None, classic)
    opt._fractional_ef_batch(grad, p2, e2, None, dion)
    torch.testing.assert_close(p1, p2, rtol=2e-5, atol=2e-5)


def test_fraction_lr_compensation_does_not_change_weight_decay_strength():
    opt = make_opt()
    opt._project = types.MethodType(lambda self, x, g: torch.zeros_like(x), opt)
    base = torch.full((2, 6, 10), 2.0)
    grad = torch.randn_like(base)

    outs = []
    for compensate in (False, True):
        p = base.clone()
        e = torch.zeros_like(p)
        g = group(
            update_rule="fractional_ef", nesterov=False, ortho_fraction=0.25,
            fraction_lr_compensation=compensate, normuon=False,
        )
        opt._fractional_ef_batch(grad, p, e, None, g)
        outs.append(p)
    expected = base * (1.0 - 0.02 * 0.1)
    assert torch.allclose(outs[0], expected)
    assert torch.allclose(outs[1], expected)


def test_keller_shape_scale_changes_update_but_not_decoupled_weight_decay():
    """Guard against accidentally multiplying WD by Keller's matrix shape factor."""
    p = torch.full((10, 5), 2.0)
    u = torch.zeros_like(p)
    raw_lr = 0.02
    wd = 0.1
    scale = (10 / 5) ** 0.5
    ResearchMuonAdamW._decay_and_update(
        p, u, raw_lr=raw_lr, scaled_lr=raw_lr * scale, wd=wd, mode="decoupled"
    )
    expected = torch.full_like(p, 2.0 * (1.0 - raw_lr * wd))
    torch.testing.assert_close(p, expected)


def test_dion3_rejects_unimplemented_factored_normuon_instead_of_silently_using_row_mode():
    g = group(update_rule="fractional_ef", nesterov=False, normuon=True, normuon_mode="factored")
    with pytest.raises(ValueError, match="currently supports NorMuon only in row mode"):
        ResearchMuonAdamW._validate_group(g)


def test_defaults_reject_conflicting_fraction_aliases():
    g = {"ortho_fraction": 0.25, "rank_fraction": 0.5}
    with pytest.raises(ValueError, match="Conflicting ortho_fraction"):
        _defaults(g)


def test_official_gns_contract_uses_list_coefficients_and_preserves_shape(monkeypatch):
    captured = {}

    class FakeGramNewtonSchulz:
        def __init__(self, *, ns_epsilon, ns_coefficients, gram_newton_schulz_reset_iterations):
            captured["epsilon"] = ns_epsilon
            captured["coefficients"] = ns_coefficients
            captured["restarts"] = gram_newton_schulz_reset_iterations

        def __call__(self, x):
            return x.clone()

    fake = types.ModuleType("gram_newton_schulz")
    fake.GramNewtonSchulz = FakeGramNewtonSchulz
    monkeypatch.setitem(sys.modules, "gram_newton_schulz", fake)

    opt = make_opt()
    opt._official_gns_cache = {}
    g = group(
        orthogonalizer="gns_official", polar_dtype="backend", coefficients="dion_ns",
        gns_reset_after=(2,),
    )
    ResearchMuonAdamW._validate_group(g)
    x = torch.randn(2, 6, 10)
    y = opt._orthogonalize(x, g)
    assert y.shape == x.shape
    assert captured["epsilon"] == pytest.approx(1e-7)
    assert all(isinstance(row, list) for row in captured["coefficients"])
    assert captured["restarts"] == [2]


def test_official_gns_shape_change_fails_loudly(monkeypatch):
    class BadGramNewtonSchulz:
        def __init__(self, **kwargs):
            pass

        def __call__(self, x):
            return x[..., :-1]

    fake = types.ModuleType("gram_newton_schulz")
    fake.GramNewtonSchulz = BadGramNewtonSchulz
    monkeypatch.setitem(sys.modules, "gram_newton_schulz", fake)

    opt = make_opt()
    opt._official_gns_cache = {}
    g = group(
        orthogonalizer="gns_official", polar_dtype="backend", coefficients="dion_ns",
        gns_reset_after=(2,),
    )
    with pytest.raises(RuntimeError, match="changed tensor shape"):
        opt._orthogonalize(torch.randn(2, 6, 10), g)


def test_cautious_weight_decay_uses_shape_scaled_effective_lr_like_nanochat():
    p = torch.full((10, 5), 2.0)
    # Positive update makes the cautious mask true everywhere for positive params.
    u = torch.ones_like(p)
    raw_lr = 0.02
    scale = (10 / 5) ** 0.5
    scaled_lr = raw_lr * scale
    wd = 0.1
    ResearchMuonAdamW._decay_and_update(
        p, u, raw_lr=raw_lr, scaled_lr=scaled_lr, wd=wd, mode="cautious"
    )
    expected = torch.full_like(p, 2.0)
    expected.sub_(scaled_lr * wd * expected)
    expected.add_(u, alpha=-scaled_lr)
    torch.testing.assert_close(p, expected)


def test_legacy_dion3_update_rule_is_canonicalized_to_fractional_ef():
    g = group(update_rule="dion3", nesterov=False)
    ResearchMuonAdamW._validate_group(g)
    assert g["update_rule"] == "fractional_ef"


def test_schedule_specific_standard_ns_normalization_eps(monkeypatch):
    import nanochat_muon_lab.research_optimizer as ro

    seen = []

    def fake_standard(x, coeffs, **kwargs):
        seen.append(kwargs)
        return x

    monkeypatch.setattr(ro, "standard_newton_schulz", fake_standard)
    opt = make_opt()

    keller = group(orthogonalizer="standard", coefficients="keller", polar_dtype="fp32")
    opt._orthogonalize(torch.randn(4, 8), keller)
    assert seen[-1]["eps"] == pytest.approx(1e-7)
    assert seen[-1]["safety_factor"] == pytest.approx(1.0)

    pe = group(orthogonalizer="standard", coefficients="nanochat_pe", polar_dtype="fp32")
    opt._orthogonalize(torch.randn(4, 8), pe)
    assert seen[-1]["eps"] == pytest.approx(1e-6)
    assert seen[-1]["safety_factor"] == pytest.approx(1.01)


def test_invalid_polar_eps_fails_loudly():
    g = group(polar_eps=0.0)
    with pytest.raises(ValueError, match="polar_eps"):
        ResearchMuonAdamW._validate_group(g)


def test_official_gns_custom_epsilon_is_forwarded_and_changes_cache_key(monkeypatch):
    constructed = []

    class FakeGramNewtonSchulz:
        def __init__(self, *, ns_epsilon, ns_coefficients, gram_newton_schulz_reset_iterations):
            constructed.append(ns_epsilon)
        def __call__(self, x):
            return x

    fake = types.ModuleType("gram_newton_schulz")
    fake.GramNewtonSchulz = FakeGramNewtonSchulz
    monkeypatch.setitem(sys.modules, "gram_newton_schulz", fake)
    opt = make_opt()
    opt._official_gns_cache = {}
    g1 = group(orthogonalizer="gns_official", polar_dtype="backend", gns_reset_after=(2,), polar_eps=1e-6)
    g2 = group(orthogonalizer="gns_official", polar_dtype="backend", gns_reset_after=(2,), polar_eps=2e-6)
    opt._official_gns(g1)
    opt._official_gns(g1)
    opt._official_gns(g2)
    assert constructed == [pytest.approx(1e-6), pytest.approx(2e-6)]


@pytest.mark.parametrize("restart", [(0,), (5,), (-1,)])
def test_invalid_gns_restart_indices_fail_loudly(restart):
    g = group(orthogonalizer="gns_reference", gns_reset_after=restart, ns_steps=5)
    with pytest.raises(ValueError, match="restart indices"):
        ResearchMuonAdamW._validate_group(g)


def test_standard_backend_rejects_gns_restart_as_noop_config():
    g = group(orthogonalizer="standard", gns_reset_after=(2,))
    with pytest.raises(ValueError, match="only meaningful for GNS"):
        ResearchMuonAdamW._validate_group(g)
