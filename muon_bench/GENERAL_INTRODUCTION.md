# NanoChat Muon Improvement Program

**Purpose.** This package is a research-to-production benchmark framework for improving Muon on the current NanoChat pretraining stack. It is designed around one rule: **do not bundle unrelated optimizer changes until each change has earned its place.** The reference path is Keller–Jordan Muon; the candidate path progressively incorporates Gram Newton–Schulz (GNS), attention-head grouping, MuonEq, post-polar normalization, NorMuon, and Dion3-lineage fractional error-feedback updates.

**Snapshot date:** 2026-09-16. NanoChat moves quickly; pin the NanoChat commit for every experiment.

---

## 1. Why this package exists

Current NanoChat's optimizer is no longer “plain Muon.” Its Muon path already combines Nesterov momentum, MuonEq-style row equilibration, Polar Express coefficients, a Muon+-inspired Frobenius correction, factored NorMuon-style variance normalization, cautious weight decay, and Keller-like matrix-shape LR scaling. That is a good tuned optimizer, but a poor *scientific baseline* if the question is which features should go into a production large-scale Muon.

This package therefore keeps two references:

1. **`native`** — current NanoChat master. This tells us what we must beat in the target codebase.
2. **`kj_reference`** — the matrix transform used by the current `KellerJordan/Muon` reference implementation, embedded inside the same NanoChat model, AdamW auxiliary groups, data pipeline, LR schedule, and distributed optimizer scaffold. This tells us what each later Muon feature contributes.

The phrase “Keller reference” here means the **current reference-repository matrix transform**: normalized EMA/Nesterov momentum with constant beta 0.95, five quintic NS steps with `(3.4445, -4.7750, 2.0315)`, bf16 NS evaluation, Keller matrix-shape scaling, and standard decoupled weight decay. Older historical Muon snippets used the unnormalized accumulator `B_t=beta B_{t-1}+G_t`. With constant beta and zero initial state, the current EMA is `M_t=(1-beta)B_t` and its Nesterov input is `(1-beta)` times the historical input, so the normalized polar direction is the same in exact arithmetic. Finite precision/epsilon make the two implementations not bitwise identical. This is why the attribution-clean reference fixes beta at 0.95. NanoChat architecture, data, batch/LR schedule, and AdamW parameter groups remain NanoChat's.

---

## 2. Target production structure

The intended mature candidate is not a monolithic new optimizer. It is a parameter-role-aware composition:

```text
Q/K/V matrices
    gradient
      -> momentum
      -> split by attention head (or head group)
      -> optional MuonEq-R
      -> GNS / polar polynomial
      -> conservative post-polar normalization
      -> NorMuon normalization
      -> LR/update RMS scaling
      -> parameter update

MLP matrices
    gradient
      -> fractional-EF residual buffer
      -> select fraction of smaller dimension
      -> MuonEq-R on selected block
      -> GNS / polar polynomial
      -> conservative post-polar normalization
      -> NorMuon normalization
      -> update selected coordinates
      -> error feedback on residual

O projection and other block matrices
      -> full-matrix Muon-family path initially

Embeddings / LM head / scalar parameters / value embeddings / smear parameters
      -> NanoChat AdamW path
```

The code makes each transformation independently switchable. A feature that does not help by itself or in its intended interaction should be removed from the final stack.

---

## 3. Keller–Jordan reference

For a gradient matrix `G_t`, the **current KellerJordan/Muon reference-repository** normalized EMA form is

\[
M_t = \beta M_{t-1} + (1-\beta)G_t,
\]

and the default Nesterov input to the polar map is

\[
\widetilde M_t=(1-\beta)G_t+\beta M_t.
\]

The Keller quintic NS map used by the reference implementation uses

\[
(a,b,c)=(3.4445,-4.7750,2.0315),
\]

for five iterations. For a wide working matrix `X`,

\[
A=X X^\top,
\qquad
X\leftarrow aX+(bA+cA^2)X.
\]

Tall matrices are transposed internally. Keller's reference code normalizes by the Frobenius norm, casts the NS working matrix to bf16, and then applies the matrix-shape factor

\[
s_{KJ}=\sqrt{\max(1,m/n)}.
\]

A subtle but important implementation point is preserved in this package: **Keller's shape factor scales the optimizer update, not the weight-decay coefficient.** Current NanoChat multiplies its fused kernel LR by the shape factor before both update and cautious WD; that is not identical to the Keller reference-repository decoupled-WD semantics used here. Accordingly, the package keeps standard decoupled WD on the raw base LR for the Keller reference, while its optional `cautious` mode deliberately uses the shape-scaled effective LR to reproduce NanoChat-style cautious-WD semantics.

---

## 4. Gram Newton–Schulz

GNS is an implementation-level improvement to polynomial polar iteration. Instead of repeatedly operating on the large rectangular matrix, it accumulates polynomial action through the small square Gram matrix. For `X in R^(n x m)`, `n <= m`, define

\[
R_0=XX^\top,\qquad Q_0=I.
\]

For polynomial coefficients `(a_t,b_t,c_t)`, GNS updates a small Gram-state and accumulated `Q`; after the chosen iterations it materializes `QX`. In exact arithmetic it evaluates the same polynomial composition as standard NS.

The official implementation exposes restart points to control half-precision Gram-state error; `(2,)` is the package's provisional five-step starting point. Restart placement is **coefficient- and hardware-dependent** and should be numerically validated/autotuned on the target GPU before a serious run. The package contains:

- `gns_reference`: pure PyTorch, auditable, slow, used for tests;
- `gns_official`: calls the official `gram-newton-schulz` package. Its current README specifies Frobenius normalization with epsilon `1e-7`, then an internal `float16` cast, with PyTorch >=2.7.1 / CUDA >=12.9 and H100 or B200/B300 named as requirements. The wrapper therefore marks precision as `backend` and does not claim bf16 control. H200 must be validated explicitly on your environment. Profile performance separately by matrix shape/aspect ratio because GNS benefit is shape-dependent.

### Coefficient schedules are kept separate

Do not conflate “GNS” with “Polar Express.” GNS is an evaluation algorithm; coefficients define the polynomial. This package exposes three schedules:

- `keller`: repeated Keller quintic;
- `nanochat_pe`: the five coefficients currently hard-coded in NanoChat;
- `dion_ns`: the five-step sequence used by current Microsoft Dion Muon and historical modded-nanogpt-style Muon recipes; the GNS README also uses this sequence in an autotuning example, but that does not make it Polar Express.

The current NanoChat PE coefficients and the Dion NS sequence are **not identical**. Therefore “switch standard NS to GNS and also change coefficients” is two changes. The experiment ladder separates them.

**Official-backend aspect-ratio note.** The current `gram-newton-schulz` implementation dispatches non-square matrices to the Gram kernel but routes square matrices through its standard Newton–Schulz path. This matters for performance interpretation: a full square projection and a narrow per-head projection can exercise different backend kernels even when both are configured as `gns_official`. Profile by role/shape rather than assuming every `gns_official` call executes the Gram kernel. Also, `nanochat_pe_gns` means the **same coefficient table on the GNS solver**, not numerical reproduction of native NanoChat Polar Express: native NanoChat and official GNS use different finite-precision normalization/solver details.

For the local `standard` backend, the reference normalizer is schedule-aware: Keller uses `||X||_F + 1e-7`, while current NanoChat PE uses `1.01 ||X||_F + 1e-6`. The official GNS backend retains its own normalization/precision semantics instead.

---

## 5. Per-head Q/K/V orthogonalization

NanoChat's Q/K/V projections are already separate parameters:

- `c_q`: `(n_head * head_dim, d_model)`;
- `c_k`: `(n_kv_head * head_dim, d_model)`;
- `c_v`: `(n_kv_head * head_dim, d_model)`.

This makes head-wise Muon especially clean. For Q,

\[
W_Q = [W_Q^{(1)};\ldots;W_Q^{(H_Q)}],
\]

and each head block is sent independently through the polar transform. For GQA, K and V use `n_kv_head`, not `n_head`.

The output projection `c_proj` is **not** treated per head in the initial package because its head decomposition lies on the input axis. That should be an explicit later experiment, not inferred from the Q/K/V rule.

Current NanoChat's standard `base_train.py` configuration presently sets `n_kv_head = n_head`, so the default benchmark is MHA even though the architecture and this package support GQA. A separate GQA smoke/transfer run is required before claiming GQA production readiness.

The package also supports `heads_per_group > 1`, allowing a later schedule such as

\[
1\to 2\to 4\to H
\]

heads per orthogonalization block. The default ladder first establishes static per-head behavior, then tests a phase switch to full matrix.

---

## 6. MuonEq

This package implements **MuonEq-R (row equilibration only)** and applies it **before** the polar map. MuonEq variants based on column or row+column equilibration are outside this package unless added as explicit new experiments. For a matrix/block with `n_rows = X.shape[-2]`, for each row `i`,

\[
r_i=\|X_{i,:}\|_2,
\qquad
r_*=\frac{\|X\|_F}{\sqrt{n_{\rm rows}}},
\qquad
X_{i,:}\leftarrow X_{i,:}\frac{r_*}{r_i+\epsilon}.
\]

In head-wise mode, equilibration is performed independently within each head/group block. In fractional-EF mode, selection happens first and MuonEq is applied to the selected submatrix. This ordering is intentional: equalizing all rows *before* the fractional-EF magnitude-based selection would corrupt the selection signal.

---

## 7. Muon+ and the important naming distinction

There are two materially different things commonly called “Muon+” in code discussions:

1. **NanoChat-style Frobenius snap**: after the approximate polar map, rescale the whole block so
   \[
   \|U\|_F=\sqrt{\min(m,n)}.
   \]
   This is a small correction for finite-iteration under/over-shoot.

2. **Full Muon+ row/column normalization**: normalize rows, columns, or compositions such as `row_col` / `col_row` after the polar transform.

The package supports both, but the proposed production candidate defaults to **`frob_snap`**, not full row/column Muon+. Full Muon+ and NorMuon both alter post-polar neuron/coordinate scaling, so enabling both without an interaction ablation is not justified.

---

## 8. NorMuon

The row form implemented here maintains a per-row second-moment state after the polar transform. The package default is `beta2=0.9` to stay close to current NanoChat; literature/reference NorMuon configurations commonly use `0.95`, so `0.9` versus `0.95` is an explicit tuning check before promotion:

\[
v_{i,t}=\beta_2 v_{i,t-1}+(1-\beta_2)\operatorname{mean}_j U_{ij}^2,
\]

then applies `v^{-1/2}` and globally rescales to preserve the pre-normalization Frobenius norm. In head-wise mode, norm preservation is performed per independent head/group block.

The code also contains a `factored` mode matching the basic shape-aware idea in current NanoChat: use row statistics for tall matrices and column statistics for wide matrices. For head-wise Q/K/V, use the row form; it has much cleaner semantics and much smaller state than a separate per-column state for every head.

---

## 9. Fractional error feedback (Dion2/Dion3 lineage)

The package implements a canonical **selected-submatrix + error-feedback** rule as `update_rule="fractional_ef"` rather than pretending fractional Muon means “delete 75% of an ordinary Muon update.” This is the mechanism we want to isolate before adaptive normalization. For a residual `E`:

1. accumulate the new gradient: `E <- E + G`;
2. orient the matrix so selection acts along the smaller dimension;
3. select `k = ceil(f * min(m,n))` rows by largest oriented-row L1 residual magnitude;
4. polar-transform only the selected block;
5. optionally apply NorMuon;
6. apply the package's chosen global decoupled WD semantics;
7. update selected coordinates;
8. multiply only the selected residual rows by `mu`; unselected residual rows remain, providing error feedback.

The package starts with `f=1/4` **only on MLP matrices**. Q/K/V remain full fraction because head-wise orthogonalization already makes their blocks small and because combining two geometry changes immediately would make attribution poor.

When `fraction_lr_compensation=True`, the selected optimizer update follows the empirical fractional-update transfer heuristic used in the Dion3 work

\[
\eta_f=\eta_1/\sqrt{f}.
\]

The implementation deliberately keeps **weight decay tied to the raw base LR**, not the compensated update LR. This is a benchmark design choice chosen to isolate fractional-update LR compensation from regularization; do not present it as a unique requirement of the Dion3 paper. Otherwise changing `f` would silently retune regularization. Treat the compensation rule as a starting point, not a replacement for an LR sweep.

### Important naming/implementation boundary

Current Microsoft Dion documents `Dion3` as `NorDion2`: Dion2-style submatrix selection/error feedback combined with NorMuon. Its current README also states that `mu` feeds Dion2's error-feedback decay but Dion3's momentum. Consequently, this harness uses the deliberately narrower name **`fractional_ef`** for the independent error-feedback rule. `fractional_ef + row NorMuon` is a production candidate and a useful Dion3-family experiment, but it is **not asserted to be state/update-equivalent to the current Microsoft `Dion3` class**. Exact-Dion3 parity is a separate validation branch.

There is a second parity distinction for tall matrices. Fractional-EF canonicalizes the matrix so selection runs along the smaller dimension. If the original matrix is tall, the canonical selected rows are original **columns**. The optional row-NorMuon state in this harness follows that canonical selected orientation. It should therefore be described as *selected-coordinate row normalization*, not as ordinary output-neuron row NorMuon on the untransposed matrix. `DIAGNOSTICS.md` defines the same convention for coverage statistics.

---

## 10. MuonClip: why it is not enabled on stock NanoChat

MuonClip/QK-Clip is a valuable large-scale stability mechanism in Kimi-style training, but **stock NanoChat has a crucial architectural difference**. Current NanoChat does

```python
q, k = norm(q), norm(k)
q = q * 1.2
k = k * 1.2
```

after Q/K projection and RoPE. Here `norm` is RMS normalization over the head dimension. Therefore a pure scalar rescaling of a Q or K weight/head is largely removed by the subsequent activation normalization:

\[
\operatorname{RMSNorm}(\alpha q)\approx \operatorname{RMSNorm}(q),\quad \alpha>0.
\]

Consequently, copying Kimi's post-optimizer weight-rescaling MuonClip into stock NanoChat would give a misleading “feature enabled” experiment whose intended action is mostly neutralized.

**Recommendation for NanoChat:** log attention-logit diagnostics, but leave weight-space MuonClip off in the optimizer-only ladder. The package nevertheless includes `muonclip.py`: it implements balanced MHA Q/K rescaling and a conservative GQA query-only rule for a branch where QK activation normalization is disabled or an explicit post-normalization scale is available. If you test that branch, label it as an architecture + optimizer stability experiment.

---

## 11. Distributed semantics

Current NanoChat does not use ordinary DDP for gradients. Its combined optimizer performs synchronization itself:

- AdamW large tensors: reduce-scatter gradient slices, local update, all-gather parameters;
- Muon matrices: stack same-shape matrices, distribute whole matrices across ranks with reduce-scatter across the parameter batch, compute each owned full matrix locally, then all-gather updated matrices.

This package subclasses current `MuonAdamW` and overrides only `_compute_muon`. As a result, the experiment retains NanoChat's optimizer-owned synchronization, state sharding, and communication overlap.

For clean pre-geometry baselines (`kj_reference`, `kj_gns`, `dion_gns`, `nanochat_pe_gns`), the package reproduces native NanoChat's **shape-only matrix grouping**. Once role-specific behavior is required, matrices are split by role+shape. Because that split itself can change collective sizes and batching efficiency, `role_split_dion_gns` is a mandatory systems control: compare it with `dion_gns` before attributing a timing difference to per-head geometry. Within each rank's owned chunk, matrices remain batched through momentum/error feedback, per-head reshape, GNS, post-normalization, NorMuon, and the update. fractional-EF top-k indices are also computed independently within one batched kernel path. This is important: a Python loop over individual matrices would turn a GNS systems experiment into a benchmark of Python dispatch overhead.


### Timing comparability

Native NanoChat and the research path do not have identical kernel fusion. Native NanoChat executes a compiled/fused Muon step, while this package keeps the research composition explicit so that feature semantics can be audited. Consequently, **BPB versus tokens/steps is the clean algorithmic metric**. Within-research timing comparisons are useful when grouping and fusion boundaries are held fixed, but raw native-vs-research optimizer milliseconds should not be interpreted as an algorithm-only speedup. A winning recipe must be fused/ported before a production throughput claim.

This is deliberately different from directly replacing the optimizer with Microsoft's Dion package. The Dion codebase is the right source of algorithms and production systems ideas, but dropping it directly into NanoChat would simultaneously alter distributed semantics and invalidate an optimizer-only benchmark.

Once the winning algorithm is known, port it into the actual production training stack using Dion/GNS-style batching, megabatching, sharding, and fused kernels.

---

## 12. Proposed production candidate

The package's `production_candidate` preset means:

- Q/K/V: per-head, full fraction;
- O: full matrix;
- MLP up/down: `fractional_ef`, `f=1/4`, with error feedback;
- GNS official backend;
- Dion NS coefficient sequence, provisional restart `(2,)` pending target-GPU restart autotuning;
- MuonEq-R;
- Frobenius snap;
- row NorMuon;
- standard decoupled WD;
- Keller-style shape scaling initially, to stay close to NanoChat LR semantics;
- Nesterov for full-rank classic groups, no Keller Nesterov for fractional-EF groups;
- embeddings/head/scalars: unchanged NanoChat AdamW.

This is **a candidate to earn**, not the baseline to start with. It is intentionally called a composed NanoChat candidate, not “exact Dion3”; exact current-Microsoft-Dion3 parity requires a dedicated port/comparison because its momentum semantics differ from this isolated fractional-EF rule. Checkpoint state must not be resumed across incompatible presets/grouping/state layouts; the patched trainer records and checks the research configuration and recorded world size before loading optimizer state. Changes in distributed topology/layout beyond world size are not migrated automatically and should start from fresh optimizer state unless an explicit state-resharding procedure is added.

---

## 13. Sources and implementation anchors

- Keller–Jordan Muon reference: https://github.com/KellerJordan/Muon
- NanoChat optimizer: https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py
- NanoChat GPT/QK norm: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
- Gram Newton–Schulz: https://github.com/Dao-AILab/gram-newton-schulz
- Microsoft Dion: https://github.com/microsoft/dion
- Dion3 paper: https://arxiv.org/abs/2608.11612
- NorMuon: https://arxiv.org/abs/2510.05491
- Muon+: https://arxiv.org/abs/2602.21545
- MuonEq: https://arxiv.org/abs/2603.28254
- Kimi K2 technical report / MuonClip: https://arxiv.org/abs/2507.20534
- Kimi K3 technical report (per-head Muon): use the official Moonshot/Kimi technical-report release for the exact version used in your internal citation set
- Group Muon / attention-head grouping study: https://arxiv.org/abs/2605.08933

Pin all repositories/commits in run metadata. “Latest master” is not a reproducible dependency.
