# NanoChat Muon Experiment Ladder

This document defines the benchmark sequence. The goal is **not** to run every combination. The goal is to promote a feature only when it improves either training efficiency, convergence, stability, or a clearly defined production constraint.

## 0. Experimental principles

Use the same NanoChat commit, tokenizer/data shards, model architecture, global batch, sequence length, seed set, LR/warmdown schedule, evaluation code, and AdamW auxiliary parameter settings within each comparison block. Log the exact git commit of NanoChat and this package.

Primary quality metric:

- validation BPB versus **training tokens/steps**.

Primary production metric:

- validation BPB versus **wall-clock time**.

Secondary metrics:

- steps to fixed BPB targets;
- wall-clock to fixed BPB targets;
- total step time and optimizer-only time;
- tokens/s;
- peak allocated memory;
- final CORE metric for promoted candidates;
- run failure/divergence rate.

Do not call a feature a speedup merely because optimizer milliseconds fall. End-to-end tokens/s and time-to-quality matter.

**Fusion confound:** native NanoChat uses compiled/fused Muon kernels while this research path is deliberately explicit/correctness-first. Therefore native-vs-research optimizer milliseconds are not a clean algorithm-only comparison. Use BPB-vs-tokens/steps for algorithmic claims, use `role_split_dion_gns` and other within-harness controls for systems attribution, and fuse/port the winner before claiming production throughput superiority.

---

# Phase A — correctness and reproducibility gates

### A0. Pin and record environment

Record:

```text
nanochat commit
muon-lab commit / zip checksum
PyTorch version
CUDA version
GPU model
world size
GNS package version/commit
GNS build requirements / detected GPU compute capability
flash-attention mode (FA3 or fallback)
COMPUTE_DTYPE
```

Use bf16 for the main benchmark. The Keller reference and current NanoChat paths were designed around bf16-safe NS behavior; fp16 is a different numerical experiment.

### A1. Pure math tests

Already included:

```bash
cd nanochat_muon_bench
PYTHONPATH=. python -m pytest -q
```

Expected in this package build: `59 passed`. The CPU suite covers pure math, batched classic and fractional-EF semantics, GQA role grouping, LR-compensation/WD separation, and MuonClip helper behavior.

The current upstream GNS README names H100 and B200/B300 and requires PyTorch >=2.7.1 plus CUDA >=12.9. Because H200 is not explicitly named there, installation plus a numerical/kernel smoke test on H200 is an explicit gate rather than an assumed supported configuration.

Before a large run, add GPU tests for:

1. standard KJ NS vs pure-PyTorch GNS in fp32;
2. standard KJ NS vs official GNS in bf16 on representative NanoChat gradient matrices;
3. finite-value stress test over at least 1,000 collected/synthetic matrices;
4. head split/merge exactness for Q and GQA K/V shapes;
5. `fractional_ef` `f=1` reduction compared with non-Nesterov full-rank update direction;
6. checkpoint save/resume consistency at the same world size; verify that mismatched world-size/config resumes fail loudly.

Acceptance gate: no NaN/Inf, no silent shape mismatch, and direction/residual discrepancies explained by expected finite-precision differences.

### A2. Native reproduction

Run unmodified NanoChat and `base_train_muon_lab.py --muon-lab-preset native` with the same seed/config. They should agree within ordinary run nondeterminism. This proves the harness itself is not changing the native path.

### A3. Checkpoint/resume gate

For `kj_reference`, one GNS preset, and `production_candidate`, compare an uninterrupted short run with a save/resume run that stops at the same final step. Verify model weights/metrics within expected determinism and inspect optimizer state. The patched trainer deliberately refuses to resume research optimizer state across a different preset, orthogonalized fraction, momentum policy, head-switch setting, or JSON override. Never use `--resume-from-step` to migrate between optimizer algorithms; start a new run from model weights only if that is the intended experiment.

---

# Phase B — establish the two baselines

## B0. Native NanoChat

This is the engineering baseline to beat.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train_muon_lab \
  --run muonlab_native_s0 \
  --muon-lab-preset native \
  <your normal NanoChat benchmark args>
```

Current native NanoChat bundles PE + MuonEq + Frobenius snap + factored NorMuon + cautious WD + momentum schedule. Do **not** infer individual feature value from this run.

## B1. Keller matrix-transform reference

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train_muon_lab \
  --run muonlab_kj_s0 \
  --muon-lab-preset kj_reference \
  --muon-lab-momentum-policy constant \
  <same benchmark args>
```

Interpretation: the current `KellerJordan/Muon` reference-repository matrix transform under NanoChat's architecture and training harness. Older unnormalized historical momentum is direction-equivalent after normalization only for constant beta (up to finite precision), which is why this reference holds `beta=0.95` fixed. It keeps NanoChat's LR schedule and WD schedule, but uses constant `beta=0.95`, standard decoupled WD, Keller NS5 and Keller shape scaling.

**Important fairness choice:** use the same nonzero `--weight-decay` across B1 and later research rungs if the target is production training. If you also want the literal historical default with WD=0, run it as a *side diagnostic*, not as the sole baseline.

### Seeds

For the first cheap model/depth:

- exploratory: 1 seed;
- promote decision: 3 seeds;
- final top candidates: 5 seeds if variance is nontrivial.

Do not spend 5 seeds on obviously losing rungs.

---

# Phase C — GNS as an implementation change

## C1. Keller polynomial through GNS

Preset:

```text
kj_gns
```

Question: **Can GNS preserve KJ training quality while reducing orthogonalization/optimizer time?**

This is the cleanest GNS test because the update polynomial stays Keller-like. It is still not bitwise “only a kernel swap”: the current official GNS path normalizes with `1e-7` and internally casts to float16, whereas the Keller reference path in this harness follows the bf16 reference-repository convention. Use pure/reference numerical checks to establish mathematical agreement and treat the official run as the hardware solver experiment.

Measure:

- val BPB vs steps;
- optimizer step ms;
- total step ms;
- tokens/s;
- GNS kernel time if profiler markers are available;
- numerical residual / update cosine against standard KJ on occasional captured matrices.
- matrix shape/aspect ratio and whether the external backend took its Gram or square-matrix standard-NS path, when that dispatch can be observed from profiling/source-version metadata.

Gate:

- no meaningful quality regression at matched steps;
- measurable optimizer or end-to-end throughput improvement.

Treat restart `(2,)` only as a starting point. Before the promoted run, autotune/validate GNS restart placement for the KJ coefficients on the target GPU and representative matrix shapes. If no stable/accurate restart works, keep KJ on standard NS. The official GNS backend owns its internal normalization/precision, so profile matrix shapes/aspect ratios separately and do not assume identical finite-precision behavior to the reference implementation.

## C2. Polynomial choice after GNS is validated

Do **not** assume one five-step polynomial is best merely because it appears in a particular implementation. From the successful C1 GNS backend, compare at least:

```text
dion_gns         # current Dion/modded-nanogpt five-step NS coefficients
nanochat_pe_gns  # NanoChat PE coefficient table evaluated by GNS (not native PE numerics)
```

Keep the same momentum policy, LR/WD schedule, GNS backend, restart policy, and parameter grouping. This is a **polynomial ablation**, not a GNS speed experiment.

Questions:

- Which schedule gives the best BPB-versus-step behavior in NanoChat?
- Does either schedule show materially different finite-precision stability on your H200 stack?
- Does the winner transfer to the next model size without coefficient-specific retuning?

The later named `*_dion_gns` presets are convenience starting points. If `nanochat_pe_gns` wins C2, carry that winner forward with `--muon-lab-override-json='{"coefficients":"nanochat_pe"}'` rather than forcing the Dion sequence into later phases.

---

# Phase D — attention geometry and grouping-control gate

## D0. Role-split systems control

Before turning on per-head geometry, run:

```text
role_split_dion_gns
```

(or the same winning Phase-C coefficient schedule via override). `dion_gns`/`nanochat_pe_gns` preserve native NanoChat's shape-only matrix partition, whereas role-aware Q/K/V experiments split matrices by role+shape. That split can change reduce-scatter/all-gather group sizes and batching efficiency by itself. D0 must match the Phase-C algorithm and differ **only** in grouping. If D0 changes throughput, use D0—not the shape-only run—as the timing control for D1. Quality should be mathematically unchanged apart from numerical/distributed ordering noise.

## D1. Per-head Q/K/V

Convenience preset:

```text
head_dion_gns
```

Conceptually this means **the winning Phase-C polynomial + role split + per-head geometry**. If `nanochat_pe` won C2, override the coefficient schedule accordingly. Q uses `n_head`; K and V use `n_kv_head`. O stays full matrix.

Compare:

- role-split full Q/K/V (D0 control);
- role-split per-head Q/K/V.

For quality-only reporting you may also show the shape-only Phase-C control, but do not attribute a wall-clock difference between shape-only and per-head runs solely to head-wise geometry.

If full per-head QKV is ambiguous, split the ablation:

- Q/K per-head, V full;
- Q/K/V per-head.

That is worth doing because V need not benefit from the same geometry as Q/K.

Log additional statistics every evaluation interval if practical:

- per-head gradient Frobenius norm distribution;
- per-head momentum norm distribution;
- per-head update RMS distribution;
- coefficient of variation across heads;
- Q/K/V update norm ratios.

Promotion criterion: improvement in BPB at matched steps or matched time with no stability cost.

---

# Phase E — MuonEq

## E1. Add MuonEq-R

Convenience preset:

```text
head_eq_dion_gns
```

Again, carry forward the winning Phase-C polynomial rather than treating the preset name as a scientific requirement. This package implements **MuonEq-R only**; column/row-column MuonEq variants require a separately defined implementation/experiment.

Question: **Does transient equilibration improve finite-step polar quality and/or convergence once the head-wise decomposition is fixed?**

Diagnostics on sampled matrices:

- row-norm coefficient of variation before/after equilibration;
- singular-value condition proxy;
- stable rank;
- polar residual before/after the finite-step solver;
- update cosine to an SVD polar reference on small sampled matrices.

MuonEq should be cheap. A quality-neutral result is still acceptable if its runtime cost is negligible and it improves numerical robustness, but do not keep it solely on theory if it consistently worsens training.

---

# Phase F — post-polar normalization interaction

Do **not** jump directly to “Muon+ + NorMuon.” Run the following branch from the best Phase-E configuration:

| ID | Muon+ mode | NorMuon | Purpose |
|---|---|---|---|
| F0 | none | off | post-polar control |
| F1 | `frob_snap` | off | finite-iteration norm correction |
| F2 | none | row | NorMuon alone |
| F3 | `frob_snap` | row | conservative combined candidate |
| F4 | `row_col` | off | full Muon+ test |
| F5 | `col_row` | off | full Muon+ alternate |
| F6 | best full Muon+ | row | **interaction only if F4/F5 win alone** |

The provided preset `head_eq_norm_dion_gns` is F3 when the Dion coefficient schedule is used. Its default `beta2=0.9` is NanoChat-oriented; for any promoted NorMuon branch also compare `beta2=0.95` before fixing this hyperparameter. Carry the polynomial winner forward via an override if needed.

Why this branch matters: full Muon+ and NorMuon both alter post-polar coordinate scaling. If F3 wins, you have evidence for a mild global norm correction plus adaptive row scaling. If full Muon+ wins alone, consider it as a substitute rather than automatically stacking it with NorMuon.

Log:

- pre/post normalization Frobenius norm;
- per-row update RMS CV;
- max/min row update RMS ratio;
- NorMuon variance-state range;
- validation BPB.

---

# Phase G — fractional MLP error feedback (Dion2/Dion3 lineage)

Start from the best full-rank stack. Apply the canonical `fractional_ef` rule only to `mlp_up` and `mlp_down`.

The package's `production_candidate` uses the canonical configuration name `ortho_fraction`:

```text
ortho_fraction_MLP = 1/4
ortho_fraction_QKV = 1
ortho_fraction_O   = 1
```

Run at least:

| ID | fraction | LR compensation | Note |
|---|---:|---:|---|
| G0 | 1.0 | no | full-rank control |
| G1 | 0.50 | `1/sqrt(f)` | intermediate |
| G2 | 0.25 | `1/sqrt(f)` | main candidate |
| G3 | 0.125 | `1/sqrt(f)` | aggressive, only if G2 strong |

Also do a small LR sweep around the heuristic. For `f=1/4`, the transfer rule suggests 2x the **update LR** for fractional groups; test something like `{1.5x, 2.0x, 2.5x}` on the cheap model before concluding. In this package that compensation does **not** multiply decoupled weight decay; WD remains tied to the raw base LR so the fraction sweep does not silently become a regularization sweep.

### Fractional-update diagnostics

For every MLP matrix (aggregate by layer/role):

- selection count per row/column;
- maximum “age” since last selection;
- fraction of coordinates never selected by milestone;
- residual/error-buffer norm;
- selected-block norm versus unselected residual norm;
- update RMS;
- optimizer step time.

A serious failure mode is starvation: the same rows are repeatedly selected while others accumulate large stale residuals. Error feedback should mitigate this, but measure it.

### Important semantics

`fractional_ef` does **not** use Keller Nesterov. It accumulates the residual, selects, updates, and only then decays selected residual rows. This isolates the paper-lineage fractional error-feedback mechanism; do not call this rung exact current Microsoft Dion3.

### G4. Exact-current-Dion3 parity branch (required before using that exact label)

Current Microsoft Dion exports `Dion3 is NorDion2`: submatrix selection/error feedback plus NorMuon, with `mu` documented as momentum for Dion3 rather than Dion2's error-feedback decay. Therefore, after the attribution-clean fractional rung is understood, do one explicit parity study if production plans to claim/port **Microsoft Dion3**:

- pin a Microsoft Dion commit;
- match coefficient/GNS backend, selected fraction, LR scaling, `mu`, NorMuon beta2/epsilon, WD, and parameter grouping;
- compare one-step captured-matrix updates and short-training curves;
- document any intentionally retained NanoChat-specific semantic difference.

The package's `production_candidate` is a composed `fractional_ef + row NorMuon` candidate, not proof of exact parity.

---

# Phase H — head grouping schedule

Only start this after the best static head-wise/full geometry is known.

### H1. Two-stage switch

Use the head-wise preset plus:

```bash
--muon-lab-head-switch-frac 0.35
```

if your warmdown ratio is the current NanoChat default 0.65, because warmdown starts at `1 - 0.65 = 0.35` of total steps.

Compare switch fractions:

```text
0.25, 0.35, 0.50, 0.70
```

and static controls:

```text
per-head throughout
full throughout
```

Do not assume warmdown start is optimal; it is simply a meaningful first phase boundary.

### H2. Gradual grouping

If H1 works, implement a staged schedule, e.g. for 16 Q heads:

```text
0–25%:   1 head/group
25–50%:  2 heads/group
50–70%:  4 heads/group
70–100%: 16 heads/group
```

K/V grouping must respect `n_kv_head` under GQA. Current NanoChat's default benchmark configuration uses `n_kv_head=n_head` (MHA), so add at least one explicit `n_kv_head<n_head` GQA smoke/transfer run before treating the GQA path as production validated.

Key question: does increasing coupling scale later improve terminal BPB without sacrificing the early head-wise gain?

---

# Phase I — LR scaling choice

The package initially retains Keller-like shape scaling to stay close to NanoChat's matrix-LR semantics. Once the algorithm is fixed, compare:

- Keller shape scaling: `sqrt(max(1,m/n))`;
- Moonlight/Kimi RMS scaling: `0.2*sqrt(max(m,n))`;
- no automatic scaling + role-specific retuned LR.

This requires a new LR sweep. **Do not swap scaling rules at fixed `matrix_lr=0.02` and interpret the result as a clean algorithm comparison**; the effective update magnitude changes dramatically.

---

# Phase J — cautious weight decay

Only after the core stack stabilizes compare:

- standard decoupled WD;
- cautious WD on full-rank classic groups.

Fractional-EF groups should initially retain the package's chosen global decoupled-WD semantics so the fraction experiment does not also become a regularization experiment. Cautious WD + fractional selection is a separate interaction with unclear semantics for unselected coordinates.

---

# Phase K — MuonClip / attention stability

On **stock current NanoChat**, do not test Kimi-style Q/K weight rescaling as if it were meaningful: QK RMS normalization after projection largely cancels scalar Q/K weight rescaling.

Instead:

1. monitor sampled maximum/percentile QK logits;
2. monitor per-head Q/K projected RMS before QK norm;
3. monitor post-QK-norm logits;
4. record any loss spikes.

If you later disable QK norm or introduce an explicit post-normalization head scale, then run a genuine MuonClip ablation with the architecture change clearly labeled. Never mix that result into the optimizer-only ladder without noting the model change.

---

# Recommended benchmark scales

## Tier 1 — smoke/correctness

- NanoChat depth 4–8;
- tens to a few hundred steps;
- one seed;
- purpose: crashes, NaNs, shape/state bugs, profiler sanity.

## Tier 2 — cheap ranking

Use NanoChat's small reference-style regime (e.g. d12 if that is your internal standard), normal sequence length and meaningful token horizon. Run one seed for every rung, then three seeds for plausible improvements.

## Tier 3 — transfer

Promote the best 2–3 candidates to at least two larger depths/model sizes. Do not infer production scaling from only d12.

## Tier 4 — production pilot

Only the best static stack and at most one dynamic/fractional variant. At this point measure actual cluster utilization, communication overlap, optimizer kernel occupancy, and checkpoint/restart reliability. Treat any change in world size or parallel topology as a new optimizer-state experiment unless an explicit state migration is implemented.

---

# Decision table

A feature is promoted when it satisfies at least one of the following without a material regression elsewhere:

1. fewer steps/tokens to the same BPB;
2. less wall-clock to the same BPB;
3. same quality with materially higher throughput/lower memory;
4. prevents a real observed instability at negligible healthy-run cost.

A feature is rejected or deferred when:

- it needs a large LR retune merely to match the parent rung and offers no systems benefit;
- the gain is within seed noise;
- it improves optimizer kernel time but not end-to-end time;
- it makes checkpointing/distributed behavior fragile;
- it duplicates another normalization without additive evidence.

---

# Minimum logging schema

Every run should emit at least:

```text
run_id
seed
nanochat_commit
optimizer_preset
all optimizer feature flags
model depth / d_model / n_head / n_kv_head
sequence length
world size / GPU type
matrix LR and actual role LR multipliers
momentum policy
WD policy + schedule
ortho_fraction per role
head group size per role / phase
step
tokens
train loss EMA
val BPB
CORE (promoted runs)
wall-clock training time
step_ms
optimizer_ms (profile windows)
tokens_per_second
peak_memory_bytes
NaN/Inf flag
```

For fractional-EF runs add selection/residual statistics. For head-wise runs add per-head norm dispersion. For GNS add solver/kernel timing and occasional polar residual diagnostics.

---

# Suggested first execution order

If compute is limited, run exactly this sequence first:

```text
native NanoChat
KJ reference
KJ + GNS (restart/precision validation first)
polynomial branch: Dion NS vs NanoChat PE coefficient table on GNS
role-split control (same algorithm, no per-head)
+ per-head QKV (and Q/K-only sub-ablation if useful)
+ MuonEq
post-polar branch: none / frob / NorMuon / frob+NorMuon / full Muon+
+ fractional_ef f=1/4 on MLP
per-head -> grouped/full schedule
LR-scaling retune: Keller vs Moonlight/Kimi RMS
```

Stop a branch early if it loses clearly. The purpose of the ladder is to avoid spending large-scale compute on a combinatorial optimizer soup.
