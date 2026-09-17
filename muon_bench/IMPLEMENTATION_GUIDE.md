# Implementation Guide — Modular Muon for NanoChat

**Snapshot:** 2026-09-17\
**Target:** current NanoChat optimizer/training structure  
**Design principle:** preserve NanoChat's distributed communication and AdamW parameter groups; change only the Muon compute semantics needed for controlled experiments.

---

## 1. Architecture of the implementation

The package intentionally separates five layers:

```text
NanoChat training loop
        |
        v
setup_research_optimizer(...)
        |
        +-- classify Transformer matrices by logical role (role-aware presets)
        |      q / k / v / o / mlp_up / mlp_down / ve_gate
        |
        +-- preserve NanoChat AdamW groups
        |
        v
ResearchMuonAdamW
        |
        +-- parent NanoChat communication / ZeRO-2-style ownership
        |
        +-- classic Muon batch path
        |      momentum -> optional Nesterov -> projection stack -> update
        |
        +-- fractional_ef batch path
               error accumulation -> selection -> projection stack -> selected update

projection stack
        MuonEq -> head/group view -> NS/GNS -> Muon+ post-normalization -> merge

optional adaptive stack
        NorMuon -> LR/shape scale -> WD -> parameter update
```

The important separation is that **GNS is an evaluation backend**, while `keller`, `dion_ns`, and `nanochat_pe` are different coefficient schedules. A GNS speed claim must not be contaminated by silently changing the polynomial.

---

## 2. Why subclass NanoChat `MuonAdamW`

Current NanoChat deliberately places gradient synchronization and sharded matrix ownership inside its optimizer rather than using an ordinary DDP optimizer flow. Replacing the entire optimizer with another library would change both optimization and distributed execution.

`ResearchMuonAdamW` therefore subclasses NanoChat's `MuonAdamW` and overrides the Muon compute phase while leaving the parent class responsible for:

- AdamW updates for non-Muon groups;
- reduce-scatter/all-reduce behavior;
- rank ownership of matrix chunks;
- asynchronous all-gather and copy-back;
- single-GPU behavior.

This makes the benchmark much closer to an optimizer-only comparison. For attribution-clean pre-geometry presets the package preserves NanoChat's native **shape-only** grouping; role+shape splitting is introduced only when a feature needs role metadata, with an explicit role-split control preset.

---

## 3. Parameter taxonomy

`setup.py` reconstructs all trainable parameters inside `model.transformer.h` and asserts that its role-aware classification is **exactly** the same parameter set that native NanoChat would give to Muon. Clean reference/GNS presets then regroup that set by **shape only**, matching native NanoChat; role-sensitive presets group by role+shape.

Roles are:

| Role | NanoChat parameter | Default production treatment |
|---|---|---|
| `q` | `block.attn.c_q.weight` | per-Q-head full-fraction Muon/GNS |
| `k` | `block.attn.c_k.weight` | per-KV-head full-fraction Muon/GNS |
| `v` | `block.attn.c_v.weight` | per-KV-head initially; V-full is an ablation |
| `o` | `block.attn.c_proj.weight` | full matrix initially |
| `mlp_up` | `block.mlp.c_fc.weight` | full Muon first; fractional_ef after validation |
| `mlp_down` | `block.mlp.c_proj.weight` | full Muon first; fractional_ef after validation |
| `ve_gate` | value-embedding attention gate | full matrix Muon path |

Everything outside Transformer blocks remains in NanoChat-style AdamW groups: embeddings, LM head, value embeddings, residual/x0 scalars, smear gate/lambdas, etc.

### GQA handling

Q uses:

```text
num_heads = model.config.n_head
```

K and V use:

```text
num_heads = model.config.n_kv_head
```

The code does not incorrectly reshape K/V using the Q-head count. Current NanoChat's default benchmark construction uses `n_kv_head=n_head`, so the standard run is MHA; run a separate GQA smoke/transfer configuration before claiming GQA deployment readiness.

---

## 4. Keller–Jordan reference path

`kj_reference` follows the **current `KellerJordan/Muon` reference-repository transform**, not every historical Muon code snippet. Historical unnormalized momentum and current normalized EMA differ by a positive scalar when beta is constant and state starts at zero, so their normalized polar directions agree in exact arithmetic. The package fixes beta=0.95 in the attribution reference and tests this relation explicitly.

The reference is deliberately conservative:

```text
beta = 0.95
Nesterov = on
NS steps = 5
coefficients = (3.4445, -4.7750, 2.0315) repeated
NS working dtype = bf16
shape scale = sqrt(max(1, rows/cols))
WD = standard decoupled WD
```

The update path is

\[
M_t=\beta M_{t-1}+(1-\beta)G_t,
\]

\[
\widetilde M_t=(1-\beta)G_t+\beta M_t,
\]

\[
U_t=P_5(\widetilde M_t),
\]

\[
W_{t+1}=(1-\eta_t\lambda_t)W_t-\eta_t s_{KJ}U_t.
\]

Two audit details are intentional:

1. the Frobenius normalization for the Keller NS path is evaluated from the bf16 working tensor, matching the finite-precision reference behavior rather than silently upgrading the baseline; and
2. `s_KJ` scales the update, **not** the decoupled weight-decay coefficient. The later `cautious` mode is different by design: it matches current NanoChat and uses the same shape-scaled effective Muon LR for both the update and the masked cautious-WD term.

`kj_reference` should remain frozen once experiments begin.

---

## 5. GNS backends

### `gns_reference`

Pure PyTorch correctness implementation. Use it for unit/numerical tests, never for throughput claims.

### `gns_official`

Uses `gram_newton_schulz.GramNewtonSchulz`. The optimizer sends the entire batch of same-shape matrices/head blocks to the official backend rather than looping one matrix at a time. This is important for meaningful kernel timing. The current upstream algorithm normalizes with epsilon `1e-7` and then casts to float16 internally; `polar_dtype="backend"` therefore means precisely “do not claim wrapper-level bf16/fp32 control.” The upstream README currently names H100 and B200/B300, PyTorch >=2.7.1, and CUDA >=12.9; H200 must be validated explicitly.

The package starts with restart `(2,)` for five-step runs, but this is **provisional**. Restart behavior interacts with coefficients, matrix shapes, precision, and hardware. Autotune/validate restart placement for each coefficient schedule on the target GPU before making quality or throughput claims. The official backend owns its internal normalization/working precision; `polar_dtype="backend"` records that fact rather than making a false bf16/fp32 claim. The adapter passes coefficients as a list-of-lists, passes restart indices as a list, and fails loudly if the external backend changes tensor shape.

---

## 6. Coefficient schedules

### Keller

```text
keller
```

Five repeated Keller quintic steps by default.

### Current Dion / modded-nanogpt sequence

```text
dion_ns
```

```text
(4.0848, -6.8946, 2.9270)
(3.9505, -6.3029, 2.6377)
(3.7418, -5.5913, 2.3037)
(2.8769, -3.1427, 1.2046)
(2.8366, -3.0525, 1.2012)
```

This sequence has appeared in Dion/current Muon implementations and in a GNS example. It should **not** be called Polar Express merely because a GNS example uses it.

### Current NanoChat Polar Express

```text
nanochat_pe
```

The exact five-step schedule is copied into `muon_math.py` as a dated reference. This is a separate polynomial ablation from `dion_ns`. Fixed schedules may be truncated for diagnostics (`ns_steps <= 5`) but are never extended by inventing additional repeated coefficients. `nanochat_pe_gns` is a coefficient-table-on-GNS experiment, not a bitwise/native-NanoChat PE reproduction.

For the local `standard` backend, normalization is schedule-aware: Keller uses epsilon `1e-7`; current NanoChat PE uses `1.01 * ||X||_F + 1e-6`. `polar_eps` may override the epsilon for numerical diagnostics, but should not be casually changed in attribution baselines. GNS reference/official paths retain their own normalization semantics.

---

## 7. Per-head and grouped attention Muon

For a Q tensor with row-stacked heads,

\[
X\in\mathbb{R}^{(H d_h)\times d_{model}},
\]

`split_heads` reshapes it to

\[
(H/g)\times (g d_h)\times d_{model},
\]

where `g = heads_per_group`.

Thus:

- `g=1`: per-head orthogonalization;
- `1<g<H`: Group-Muon-style intermediate coupling;
- `g=H`: full projection matrix.

The grouping is a view/reshape around the projection stack. Momentum and NorMuon state remain attached to the original parameter rows, so a head-wise → full phase switch can retain state.

O projection remains full matrix initially because its natural head decomposition is on the input dimension, not the output dimension used by Q/K/V.

---

## 8. MuonEq ordering

For classic Muon:

```text
momentum/Nesterov tensor
    -> split into chosen optimizer blocks
    -> MuonEq-R inside each block
    -> polar solve
```

For `fractional_ef`:

```text
error buffer
    -> select top fraction using ORIGINAL residual magnitude
    -> MuonEq-R on the selected block
    -> polar solve
```

Do not equilibrate every residual row before fractional selection; that would alter the magnitude signal used for selection.

---

## 9. Muon+ modes

`muonplus_mode` supports:

```text
none
frob_snap
row
col
row_col
col_row
```

`frob_snap` is deliberately distinguished from full paper-style row/column Muon+. It rescales an approximate polar result to the Frobenius norm of an exact semi-orthogonal matrix:

\[
U\leftarrow U\frac{\sqrt{\min(m,n)}}{\|U\|_F}.
\]

The default combined candidate uses `frob_snap + NorMuon`, because full Muon+ row/column normalization and NorMuon both change post-polar coordinate scaling and should first be tested separately.

---

## 10. NorMuon state

Row mode maintains one fp32 second-moment scalar per output row:

\[
v_{i,t}=\beta_2v_{i,t-1}+(1-\beta_2)\operatorname{mean}_j U_{ij}^2.
\]

After multiplying by `v^{-1/2}`, the update is globally norm-corrected so the pre-NorMuon Frobenius norm is preserved.

For per-head/grouped Q/K/V, this norm preservation is performed independently for each optimizer block. The state is still stored row-aligned, so grouping can change without reallocating state.

A factored row/column mode is included for full matrices, but the main per-head experiments use row mode. The default `beta2=0.9` is NanoChat-oriented; because reference NorMuon recipes often use `0.95`, sweep/compare `0.9` and `0.95` before fixing the production value.

---

## 11. Fractional-EF MLP path (Dion2/Dion3 lineage)

The initial implementation intentionally restricts fractional selection to non-head-wise groups. MLP up/down are the main target.

For each same-shape matrix batch:

1. `E <- E + G`;
2. transpose only if necessary so selection uses the smaller matrix dimension;
3. score each selectable row using L1 residual magnitude;
4. select `k = ceil(f * r)` rows independently per matrix;
5. run MuonEq/polar/Muon+ on the selected block;
6. apply row NorMuon using selected variance states;
7. apply the package's **global decoupled WD using the raw LR**;
8. update only selected rows, with optional `1/sqrt(f)` update-LR compensation;
9. multiply selected residual rows by `momentum`; leave unselected residual rows untouched.

The code separates

```text
raw_lr       -> weight decay
update_lr    -> selected optimizer update
```

For tall matrices the implementation transposes to a canonical orientation so selection is always row-wise on the smaller dimension. Consequently, optional row-NorMuon on a tall fractional group tracks the selected canonical rows, which correspond to **original columns**. Do not interpret that state as ordinary output-row/neuron NorMuon.

so changing `f` cannot silently change regularization strength. This is an isolation/design choice of the benchmark, not a claim that Dion3 uniquely mandates this WD convention.

`fractional_ef` groups do **not** use Keller Nesterov. Combining the two would be a new algorithm. Current Microsoft `Dion3` also bundles NorMuon-family behavior and documents different `mu` semantics, so this isolated rule must not be presented as exact Microsoft Dion3.

---

## 12. Batching and performance semantics

NanoChat groups Muon matrices by shape. The clean reference/GNS presets preserve that exact **shape-only** partition. Role-sensitive presets necessarily split by role+shape; this can change collective sizes and launch/batching behavior even before per-head reshaping. `role_split_dion_gns` therefore exists as a systems control and should precede `head_dion_gns`. Within each resulting group, the research optimizer preserves the batch through:

- momentum/error accumulation;
- per-head reshape;
- MuonEq;
- GNS;
- Muon+;
- NorMuon;
- parameter update.

Fractional-EF top-k selection is also batched, with a separate index set per matrix.

This removes a major benchmark artifact: a correctness-oriented Python loop over matrices can make a mathematically faster GNS method appear slower end to end.

The package is still **not the final fused production kernel**. After the winning algorithm is established, the final engineering pass should consider cross-group megabatching, fused normalization/update kernels, better overlap, and topology-aware sharding.

**Performance interpretation:** native NanoChat uses compiled/fused Muon kernels, while `ResearchMuonAdamW` is a correctness-first explicit composition. Use BPB-vs-tokens/steps for algorithmic conclusions. For systems conclusions, compare runs with the same grouping/fusion boundary and profile kernels/collectives separately. Do not present raw native-vs-research `optimizer_ms` as an algorithm-only speedup. Fuse/port the winning recipe before claiming production end-to-end throughput gains.

---

## 13. MuonClip

`muonclip.py` provides:

```python
clip_factors(max_logits, tau=100.0)
apply_muonclip_weights_(...)
```

For ordinary MHA, if

\[
\gamma_h=\min(1,\tau/s_h),
\]

then the helper applies balanced scaling

\[
W_Q^{(h)}\leftarrow \sqrt{\gamma_h}W_Q^{(h)},\qquad
W_K^{(h)}\leftarrow \sqrt{\gamma_h}W_K^{(h)}.
\]

For GQA, K heads are shared. The implementation therefore only provides the conservative `query_only` policy:

\[
W_Q^{(h)}\leftarrow\gamma_h W_Q^{(h)},
\]

while shared K weights remain unchanged.

### Why it is not wired into stock NanoChat

Current NanoChat RMS-normalizes Q and K activations after projection. For positive scalar `c`, approximately

\[
\operatorname{RMSNorm}(c q)\approx\operatorname{RMSNorm}(q),
\]

so rescaling an entire Q/K head weight does not provide the Kimi-style control intended by MuonClip. The test suite explicitly checks this scale-invariance behavior.

To test genuine MuonClip, make it a labeled architecture branch where QK activation normalization is disabled or where a separately learned/post-normalization head scale is clipped. Do not mix that branch into optimizer-only comparisons.

---

## 14. Presets

| Preset | Meaning |
|---|---|
| `kj_reference` | current KellerJordan/Muon reference-repository transform |
| `kj_moonlight_scale` | KJ transform, alternative RMS update scaling |
| `kj_gns` | Keller coefficients through official GNS |
| `dion_gns` | Dion NS coefficient schedule through GNS, native shape-only grouping |
| `nanochat_pe_gns` | NanoChat PE coefficient table through GNS, native shape-only grouping |
| `role_split_dion_gns` | same algorithm as `dion_gns`, role+shape grouping control |
| `head_dion_gns` | + per-head Q/K/V |
| `head_eq_dion_gns` | + MuonEq-R |
| `head_eq_norm_dion_gns` | + Frobenius snap + row NorMuon |
| `production_candidate` | + `fractional_ef` on MLP up/down; row NorMuon inherited from parent preset |

The polynomial winner should be selected by Phase C of the experiment ladder. `production_candidate` uses `dion_ns` merely as a concrete starting configuration.

---

## 15. Configuration schema and JSON overrides

Configuration precedence is **preset < explicit setup arguments (`ortho_fraction`, `beta2`) < JSON override**. Unknown keys and unknown logical roles are errors. `fractional_roles` is canonical; `dion3_roles` is a deprecated compatibility alias. `rank_fraction` is accepted only as a deprecated JSON alias for `ortho_fraction`; conflicting old/new values are rejected. `heads_per_group` must be a positive divisor of every affected role's head count.

`polar_dtype` means `bf16`, `fp32`, or `native` for the standard/reference backends. Official GNS must use `backend`, because the external kernel owns its internal precision path.


The patched trainer accepts:

```bash
--muon-lab-override-json='<JSON object>'
```

Examples:

```bash
# Q/K only head-wise
--muon-lab-override-json='{"per_head_roles":["q","k"]}'

# Full Muon+ without NorMuon
--muon-lab-override-json='{"muonplus_mode":"row_col","normuon":false}'

# Factored NorMuon on full matrices
--muon-lab-override-json='{"normuon":true,"normuon_mode":"factored","per_head_roles":[]}'

# Moonlight/Kimi update RMS scaling; retune matrix LR
--muon-lab-override-json='{"lr_scale":"moonshot_rms"}'

# Change grouping from one head to two heads per optimizer block
--muon-lab-override-json='{"heads_per_group":2}'
```

Invalid interactions are rejected early, including `fractional_ef`+Nesterov, fractional+head-wise in the initial implementation, factored NorMuon+head-wise, fractional+factored-NorMuon, and cautious-WD+fractional-EF.

---

## 16. Dynamic head phase switch

The copied trainer accepts:

```bash
--muon-lab-head-switch-frac=<fraction>
```

At the boundary it sets `heads_per_group = num_heads` for Q/K/V groups, which converts them from head/group-wise to full-projection orthogonalization without clearing momentum or row-NorMuon state.

### M1 runtime-state and provenance contract

Revision 0.6.0 adds the package-level contract that makes dynamic grouping and resume identity
explicit:

- `runtime_state.py` separates immutable initial grouping, mutable active grouping, expected lazy
  state schemas, and observed rank-local checkpoint signatures;
- transitions validate every affected Q/K/V group before mutation, advance stage/count once, and
  reject duplicate execution;
- restore cross-checks explicit runtime grouping against serialized optimizer `param_groups`;
- structural signatures intentionally exclude scheduled LR, momentum, and weight decay so they can
  be checked both before and after `optimizer.load_state_dict`;
- `provenance.py` supplies canonical JSON/SHA-256 fingerprints, explicit resume modes, shared versus
  rank-local seed policy, immutable manifest publication, monotonic event logs, and rank-signature
  completeness checks.

`setup_research_optimizer` now initializes initial grouping, runtime grouping, and the live structural
signature. The trainer must build the expected-state schema after it has resolved world size. New
integrations should use `transition_attention_grouping_`, which rejects duplicate transitions by default. The current generated trainer intentionally retains
the low-level `set_attention_grouping` mutation path until M2 can switch it together with explicit
checkpoint metadata restore and validation.

This is the M1 foundation only. The generated trainer still uses its v0.5 checkpoint metadata path.
Persisting/cross-checking these new artifacts around optimizer load, plus RNG/GradScaler/dataloader
continuity, belongs to M2 and must land before Q-BASE qualification.

For a current NanoChat warmdown ratio of 0.65, `0.35` is a meaningful first switch point because warmdown begins at 35% of training. It is a hypothesis, not a fixed recommendation; compare multiple boundaries and static controls.

---

## 17. Momentum policy

Research presets default to constant Keller-style `0.95`:

```bash
--muon-lab-momentum-policy=constant
```

To isolate a feature under NanoChat's native time-varying Muon momentum schedule:

```bash
--muon-lab-momentum-policy=nanochat
```

Do not mix the policies within a comparison block without labeling the change.

---

## 18. Checkpoint and phase-change rules

Safe with existing state shapes:

- changing `heads_per_group` among valid divisors;
- turning a run from per-head to full only when the optimizer configuration itself is otherwise unchanged.

Do **not** change mid-run without a migration plan:

- `normuon_mode` row ↔ factored;
- classic ↔ `fractional_ef`;
- a parameter's logical role;
- fraction in a way that is intended to reset error feedback;
- adding/removing optimizer states and expecting an old checkpoint to load transparently.

Checkpoint/resume equivalence is a required gate before a long run. The patched trainer stores NanoChat's `user_config` and refuses to load research optimizer state when the saved research preset/fraction/momentum policy/head-switch/override configuration does not match the requested run. It checkpoints a canonical, sorted JSON representation of the **effective** research configuration (after preset/argument/override resolution) **plus a parameter-group signature containing effective LRs, WD/AdamW hyperparameters, matrix algorithm metadata, head grouping, and parameter shapes**, and refuses resume if that changes; older lab checkpoints fall back to semantic comparison of their legacy CLI fields. It also refuses research optimizer-state resume when the recorded world size differs. This is intentionally conservative: a cross-preset optimizer state can have different parameter-group ordering and state tensors even when model weights are compatible. Changes in TP/EP/PP/FSDP layout are not migrated by this package; start with fresh optimizer state unless you implement and validate explicit state resharding.

---

## 19. Recommended installation workflow

```bash
# 1. Pin NanoChat
cd /path/to/nanochat
# git checkout <recorded commit>

# 2. Install the lab non-destructively
python /path/to/nanochat_muon_bench/install_into_nanochat.py

# 3. Run package CPU tests from the package directory
cd /path/to/nanochat_muon_bench
python -m pytest -q

# 4. Back in NanoChat, smoke test KJ reference
cd /path/to/nanochat
python -m scripts.base_train_muon_lab \
  --muon-lab-preset=kj_reference \
  <small smoke-test flags>

# 5. Reproduce native through the copied harness
python -m scripts.base_train_muon_lab \
  --muon-lab-preset=native \
  <same flags>

# 6. Install/verify official GNS and enter Phase C only after A/B pass
```

---

## 20. What has and has not been validated here

Validated in the artifact environment:

- module syntax;
- pure math behavior;
- GNS reference math/restart tests;
- batch head-wise path;
- batch fractional-EF wide/tall orientation;
- LR-compensation/WD separation;
- strict configuration/override precedence;
- native shape-only grouping and role-split control;
- Keller historical/current momentum scalar-equivalence and WD-scale regression guards;
- installer transactional behavior and research resume guards;
- GQA parameter grouping;
- MuonClip helper semantics;
- current package tests: **45 passing**, including mocked external-GNS API/shape contract tests and package-version consistency.

Not validated here:

- end-to-end training on your exact NanoChat commit;
- official GNS CUDA kernels on H200;
- multi-rank collective timing/correctness on your cluster;
- checkpoint/resume against your production launcher;
- statistical superiority of any candidate configuration.

Those are explicitly the first gates in the experiment ladder.
