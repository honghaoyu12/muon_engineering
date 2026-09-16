import pytest
import torch

from nanochat_muon_lab.diagnostics import (
    block_frobenius_stats,
    finite_summary,
    polar_residual,
    selection_counts,
    tensor_rms,
)


def test_tensor_rms_definition():
    x = torch.tensor([3.0, 4.0])
    assert torch.allclose(tensor_rms(x), torch.sqrt(torch.tensor(12.5)))


def test_block_frobenius_stats_zero_cv_for_equal_heads():
    # Four heads, one row/head, equal norm 2.
    x = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    s = block_frobenius_stats(x, num_heads=4)
    assert torch.allclose(s["mean"], torch.tensor(2.0))
    assert torch.allclose(s["cv"], torch.tensor(0.0))


def test_polar_residual_zero_for_identity():
    assert torch.allclose(polar_residual(torch.eye(4)), torch.tensor(0.0))


def test_selection_counts_and_range_check():
    idx = torch.tensor([[0, 2], [2, 3]])
    assert torch.equal(selection_counts(idx, 4), torch.tensor([1, 0, 2, 1]))
    with pytest.raises(ValueError):
        selection_counts(torch.tensor([4]), 4)


def test_finite_summary_detects_nonfinite():
    s = finite_summary(torch.tensor([1.0, float("inf"), float("nan")]))
    assert not s["all_finite"]
    assert s["finite_count"] == 1
    assert s["numel"] == 3
    assert s["max_abs_finite"] == 1.0
