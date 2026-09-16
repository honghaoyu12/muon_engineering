import sys
import types

import pytest
import torch
from torch import nn

# Standalone stub for the NanoChat optimizer base. These tests audit grouping/configuration only.
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

from nanochat_muon_lab.setup import setup_research_optimizer, set_attention_grouping


class Attn(nn.Module):
    def __init__(self, d=12, nq=4, nkv=2):
        super().__init__()
        hd = d // nq
        self.c_q = nn.Linear(d, nq * hd, bias=False)
        self.c_k = nn.Linear(d, nkv * hd, bias=False)
        self.c_v = nn.Linear(d, nkv * hd, bias=False)
        self.c_proj = nn.Linear(d, d, bias=False)
        self.ve_gate = nn.Linear(12, nkv, bias=False)


class MLP(nn.Module):
    def __init__(self, d=12):
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d, bias=False)
        self.c_proj = nn.Linear(4 * d, d, bias=False)


class Block(nn.Module):
    def __init__(self, d=12):
        super().__init__()
        self.attn = Attn(d)
        self.mlp = MLP(d)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(n_embd=12, n_head=4, n_kv_head=2)
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(50, 12)
        self.transformer.h = nn.ModuleList([Block(), Block()])
        self.lm_head = nn.Linear(12, 50, bias=False)
        self.value_embeds = nn.Embedding(5, 12)
        self.resid_lambdas = nn.Parameter(torch.ones(2))
        self.x0_lambdas = nn.Parameter(torch.ones(2))
        self.smear_gate = nn.Linear(12, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.ones(()))
        self.backout_lambda = nn.Parameter(torch.ones(()))


def muon_groups(opt):
    return [g for g in opt.param_groups if g.get("kind") == "muon"]


def test_shape_only_baseline_matches_native_shape_partition():
    model = FakeModel()
    opt = setup_research_optimizer(model, preset="kj_reference")
    groups = muon_groups(opt)
    native = list(model.transformer.h.parameters())
    native_shapes = sorted({tuple(p.shape) for p in native})
    assert [tuple(g["params"][0].shape) for g in groups] == native_shapes
    assert [len(g["params"]) for g in groups] == [
        sum(tuple(p.shape) == shape for p in native) for shape in native_shapes
    ]
    assert all(g["logical_role"] == "mixed" for g in groups)
    assert opt.muon_lab_role_split is False


def test_role_split_control_has_no_headwise_geometry():
    model = FakeModel()
    opt = setup_research_optimizer(model, preset="role_split_dion_gns")
    groups = muon_groups(opt)
    assert opt.muon_lab_role_split is True
    assert {g["logical_role"] for g in groups} == {
        "q", "k", "v", "o", "mlp_up", "mlp_down", "ve_gate"
    }
    assert all(g["num_heads"] is None for g in groups)


def test_production_preset_is_role_aware_and_gqa_correct():
    model = FakeModel()
    opt = setup_research_optimizer(model, preset="production_candidate", ortho_fraction=0.25)
    groups = muon_groups(opt)
    by_role = {}
    for g in groups:
        by_role.setdefault(g["logical_role"], []).append(g)

    assert set(by_role) == {"q", "k", "v", "o", "mlp_up", "mlp_down", "ve_gate"}
    assert all(g["num_heads"] == 4 for g in by_role["q"])
    assert all(g["num_heads"] == 2 for g in by_role["k"] + by_role["v"])
    assert all(g["num_heads"] is None for g in by_role["o"])
    mlp = by_role["mlp_up"] + by_role["mlp_down"]
    assert all(g["update_rule"] == "fractional_ef" for g in mlp)
    assert all(g["ortho_fraction"] == 0.25 for g in mlp)
    assert all(g["fraction_lr_compensation"] for g in mlp)
    assert all(not g["nesterov"] for g in mlp)


def test_override_precedence_beta2_and_fraction_and_deprecated_alias():
    model = FakeModel()
    opt = setup_research_optimizer(
        model,
        preset="production_candidate",
        ortho_fraction=0.25,
        beta2=0.9,
        override={"ortho_fraction": 0.5, "beta2": 0.95},
    )
    groups = muon_groups(opt)
    assert all(g["beta2"] == 0.95 for g in groups)
    assert all(g["ortho_fraction"] == 0.5 for g in groups if g["update_rule"] == "fractional_ef")

    opt2 = setup_research_optimizer(
        FakeModel(), preset="production_candidate", override={"rank_fraction": 0.125}
    )
    assert all(
        g["ortho_fraction"] == 0.125
        for g in muon_groups(opt2) if g["update_rule"] == "fractional_ef"
    )


def test_override_can_make_qk_only_per_head():
    model = FakeModel()
    opt = setup_research_optimizer(
        model, preset="dion_gns", override={"per_head_roles": ["q", "k"]}
    )
    groups = muon_groups(opt)
    assert any(g["logical_role"] == "q" and g["num_heads"] == 4 for g in groups)
    assert any(g["logical_role"] == "k" and g["num_heads"] == 2 for g in groups)
    assert all(g["num_heads"] is None for g in groups if g["logical_role"] == "v")


def test_invalid_override_key_role_and_head_group_fail_loudly():
    with pytest.raises(ValueError, match="Unknown Muon-lab override"):
        setup_research_optimizer(FakeModel(), override={"normuon_beta22": 0.9})
    with pytest.raises(ValueError, match="Unknown logical role"):
        setup_research_optimizer(FakeModel(), override={"dion3_roles": ["not_a_role"]})
    with pytest.raises(ValueError, match="must divide"):
        setup_research_optimizer(
            FakeModel(), preset="head_dion_gns", override={"heads_per_group": 3}
        )


def test_dynamic_grouping_validates_divisibility():
    opt = setup_research_optimizer(FakeModel(), preset="head_dion_gns")
    set_attention_grouping(opt, heads_per_group_q=2, heads_per_group_kv=2)
    q = [g for g in muon_groups(opt) if g["logical_role"] == "q"]
    assert all(g["heads_per_group"] == 2 for g in q)
    with pytest.raises(ValueError, match="must divide"):
        set_attention_grouping(opt, heads_per_group_q=3, heads_per_group_kv=2)


def test_dion3_does_not_silently_enable_normuon():
    model = FakeModel()
    opt = setup_research_optimizer(
        model,
        preset="role_split_dion_gns",
        override={"dion3_roles": ["mlp_up"], "normuon": False},
    )
    mlp = [g for g in opt.param_groups if g.get("logical_role") == "mlp_up"]
    assert mlp and all(g["update_rule"] == "fractional_ef" for g in mlp)
    assert all(g["normuon"] is False for g in mlp)


def test_legacy_dion3_roles_alias_canonicalizes_to_fractional_roles():
    opt = setup_research_optimizer(
        FakeModel(), preset="role_split_dion_gns",
        override={"dion3_roles": ["mlp_up"]},
    )
    assert opt.muon_lab_config["fractional_roles"] == ("mlp_up",)
    assert "dion3_roles" not in opt.muon_lab_config
    groups = [g for g in muon_groups(opt) if g["logical_role"] == "mlp_up"]
    assert groups and all(g["update_rule"] == "fractional_ef" for g in groups)


def test_param_group_signature_detects_effective_lr_and_grouping_changes():
    a = setup_research_optimizer(FakeModel(), preset="dion_gns", matrix_lr=0.02)
    b = setup_research_optimizer(FakeModel(), preset="dion_gns", matrix_lr=0.03)
    c = setup_research_optimizer(FakeModel(), preset="role_split_dion_gns", matrix_lr=0.02)
    assert a.muon_lab_param_group_signature != b.muon_lab_param_group_signature
    assert a.muon_lab_param_group_signature != c.muon_lab_param_group_signature
    # Signature is JSON-friendly and contains matrix shapes rather than tensor objects.
    import json
    json.dumps(a.muon_lab_param_group_signature, sort_keys=True)


def test_heads_per_group_without_per_head_roles_is_rejected_as_noop():
    with pytest.raises(ValueError, match="has no effect"):
        setup_research_optimizer(FakeModel(), override={"heads_per_group": 2})


def test_fraction_lr_compensation_without_fractional_roles_is_rejected_as_noop():
    with pytest.raises(ValueError, match="has no effect"):
        setup_research_optimizer(FakeModel(), override={"fraction_lr_compensation": True})


def test_overlapping_per_head_and_fractional_roles_is_rejected():
    with pytest.raises(ValueError, match="does not combine per-head and fractional_ef"):
        setup_research_optimizer(
            FakeModel(),
            override={"per_head_roles": ["q"], "fractional_roles": ["q"]},
        )


def test_update_rule_cannot_be_smuggled_through_json_override():
    with pytest.raises(ValueError, match="Unknown Muon-lab override"):
        setup_research_optimizer(FakeModel(), override={"update_rule": "fractional_ef"})
