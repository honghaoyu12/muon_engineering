# Diagnostics Definitions

This file defines the quantities referenced by the experiment ladder. The helper implementation is
`nanochat_muon_lab/diagnostics.py`. These diagnostics are **not automatically inserted into the
NanoChat training loop** because tensor capture, synchronizations, and reductions can contaminate
throughput measurements. Run them on sampled tensors outside timing windows unless a measurement is
explicitly intended to include their cost.

## Update RMS

For a tensor \(X\) with \(N\) entries,

\[
\operatorname{RMS}(X)=\sqrt{\frac{1}{N}\sum_i X_i^2}.
\]

Use this for update-magnitude comparisons. Do not substitute Frobenius norm without accounting for
matrix size, because \(\|X\|_F=\sqrt{N}\,\operatorname{RMS}(X)\).

## Per-head / per-group norm dispersion

For optimizer blocks \(B_h\), define \(n_h=\|B_h\|_F\). Report the population mean, population
standard deviation, minimum, maximum, and

\[
\operatorname{CV}=\frac{\operatorname{std}_h(n_h)}{\operatorname{mean}_h(n_h)}.
\]

For GQA, Q uses `n_head`, whereas K/V use `n_kv_head`. `heads_per_group` must match the optimizer
view used in that experiment.

## Polar residual

For a wide/semi-row-orthogonal matrix \(U\in\mathbb{R}^{m\times n}\), \(m\le n\),

\[
r_{\rm polar}=\frac{\|UU^\top-I_m\|_F}{\sqrt{m}}.
\]

For a tall matrix use \(U^\top U\) and normalize by \(\sqrt{n}\). This is a numerical diagnostic,
not a training objective.

## Fractional-selection coverage

If \(i_t\) are selected canonical coordinates, maintain a count \(c_j\) for every selectable
coordinate. Useful summaries are the fraction never selected, minimum/median/maximum count, and the
maximum age since last selection. Keep these statistics by layer and matrix role when possible;
aggregating globally can hide starvation in one layer.

**Canonical orientation matters.** Fractional-EF selection always operates on the smaller matrix
dimension. A tall original matrix is transposed before selection. Therefore, for a tall MLP matrix,
a "row" in fractional-EF diagnostics (and in the optional row-NorMuon state) corresponds to an
original **column**, not an original output neuron. This is intentionally different from applying
ordinary row-NorMuon to the untransposed full matrix.

## Finite-value diagnostics

At minimum record whether all entries are finite and the maximum absolute finite value for sampled
gradients, momentum/error buffers, polar outputs, and final updates. A single NaN/Inf makes a run a
numerical failure even if the scalar loss has not yet become non-finite.
