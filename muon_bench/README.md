# NanoChat Muon Benchmark Package

Snapshot: **2026-09-16**.  **Package revision:** `0.5.0` (fourth audit). This is a correctness-first research harness for benchmarking Muon-family changes inside the current NanoChat training stack.

The package has two goals that should not be conflated:

1. preserve an **attribution-clean current KellerJordan/Muon repository reference transform** inside NanoChat; and
2. make production-oriented features independently switchable: GNS, alternative NS polynomials, per-head Q/K/V, MuonEq, Muon+/Frobenius correction, NorMuon, Dion3-lineage fractional error-feedback MLP updates, and architecture-aware MuonClip helpers.

## Deliverables

- `GENERAL_INTRODUCTION.md` — conceptual introduction, equations, feature interactions, and recommendations.
- `EXPERIMENT_LADDER.md` — staged benchmark program, metrics, gates, seed policy, and failure diagnostics.
- `IMPLEMENTATION_GUIDE.md` — exact code structure, operation ordering, installation, flags, presets, and NanoChat integration details.
- `DIAGNOSTICS.md` — exact definitions for RMS, head/group dispersion, polar residual, finite checks, and fractional-selection coverage.
- `AUDIT_REPORT.md` — fourth-pass bug/consistency audit and remaining validation boundary.
- `SOURCE_SNAPSHOT.md` — upstream integration assumptions checked on 2026-09-16; experiments must still pin a commit.
- `nanochat_muon_lab/muon_math.py` — auditable pure-PyTorch reference math.
- `nanochat_muon_lab/research_optimizer.py` — role-aware, batched NanoChat `MuonAdamW` subclass.
- `nanochat_muon_lab/setup.py` — NanoChat parameter classification and benchmark presets.
- `nanochat_muon_lab/muonclip.py` — MHA/GQA-aware MuonClip weight-rescaling helpers for architectures where the rescaling is meaningful.
- `install_into_nanochat.py` — non-destructive installer; creates `scripts/base_train_muon_lab.py` and leaves native `base_train.py` untouched.
- `tests/` — CPU correctness and semantics tests.

## Install into a pinned NanoChat checkout

From the NanoChat repository root:

```bash
python /path/to/nanochat_muon_bench/install_into_nanochat.py
```

The installer deliberately fails if the expected current-NanoChat patch points are not found exactly once. That is preferable to silently benchmarking against an upstream script whose semantics have changed.

The performance presets use the official `gram-newton-schulz` package. Install the version appropriate for your CUDA/PyTorch/GPU environment before running `gns_official` presets. The included pure-PyTorch GNS implementation is for correctness testing, not performance claims. The current upstream GNS README specifies PyTorch >=2.7.1, CUDA >=12.9, and H100 or B200/B300 as named tested hardware. H200 is Hopper but is not explicitly named there, so treat H200 support as a required smoke-test gate rather than an assumption.

For package development/tests outside a NanoChat checkout, editable installation is optional but convenient:

```bash
python -m pip install -e /path/to/nanochat_muon_bench --no-deps
# On an offline cluster with build requirements already installed:
python -m pip install -e /path/to/nanochat_muon_bench --no-deps --no-build-isolation
cd /path/to/nanochat_muon_bench
python -m pytest -q
```

`--no-deps` is intentional: the benchmark should inherit the pinned NanoChat/PyTorch/CUDA environment rather than trying to replace a working GPU stack.

## First smoke test

```bash
python -m scripts.base_train_muon_lab \
  --depth=4 --max-seq-len=512 --device-batch-size=1 \
  --eval-tokens=512 --core-metric-every=-1 \
  --total-batch-size=512 --num-iterations=20 \
  --muon-lab-preset=kj_reference
```

Then reproduce native NanoChat through the copied harness:

```bash
python -m scripts.base_train_muon_lab \
  --muon-lab-preset=native \
  <the exact same NanoChat flags>
```

## Canonical presets

```text
native                    current NanoChat optimizer, unchanged
kj_reference              current KellerJordan/Muon reference transform (constant beta=0.95)
kj_moonlight_scale        KJ transform + Moonlight/Kimi RMS update scaling (LR retune required)
kj_gns                    KJ coefficients evaluated with official GNS
dion_gns                  Dion/modded-nanogpt 5-step NS coefficients through GNS, native shape grouping
nanochat_pe_gns            NanoChat PE coefficient table through GNS, native shape grouping
role_split_dion_gns        dion_gns algorithm, but role+shape grouping control
head_dion_gns             dion_gns + per-head Q/K/V
head_eq_dion_gns          + MuonEq-R
head_eq_norm_dion_gns     + Frobenius snap + row NorMuon
production_candidate      + fractional_ef on MLP up/down + inherited row NorMuon
```

`production_candidate` is a **starting hypothesis**, not a claim that the full stack is already validated. The experiment ladder is designed to earn each component sequentially. `role_split_dion_gns` is a required systems control before attributing a timing change to per-head geometry, because role-aware grouping can change the number and size of NanoChat's optimizer collectives.

## Useful overrides without editing code

The patched trainer accepts a JSON dictionary:

```bash
# Q/K per-head, leave V full
--muon-lab-override-json='{"per_head_roles":["q","k"]}'

# Full paper-style Muon+ row->column normalization, NorMuon off
--muon-lab-override-json='{"muonplus_mode":"row_col","normuon":false}'

# Switch to Moonlight/Kimi RMS scaling (requires a fresh matrix-LR sweep)
--muon-lab-override-json='{"lr_scale":"moonshot_rms"}'

# Override the fractional-EF selected fraction and NorMuon beta2 explicitly
--muon-lab-override-json='{"ortho_fraction":0.25,"beta2":0.95}' 
```

For the dynamic head-granularity experiment:

```bash
--muon-lab-head-switch-frac=0.35
```

This retains optimizer state and changes Q/K/V from the configured head grouping to a full projection matrix when the specified fraction of training is reached.

## Naming and numerical-solver caveats

`kj_reference` follows the **current** `KellerJordan/Muon` normalized-EMA/Nesterov form. Older historical Muon snippets used an unnormalized momentum accumulator. With constant `beta` and zero initial state the two preconditioner inputs differ by the positive scalar `1-beta`, so their normalized polar directions agree in exact arithmetic; they are not bitwise identical in finite precision. The benchmark therefore fixes `beta=0.95` for the reference path unless a momentum-schedule experiment is explicitly labeled. Current NanoChat's native schedule is materially different: it ramps 0.85 -> 0.97 over 400 steps, stays at 0.97, then reaches 0.90 during LR warmdown.

`nanochat_pe_gns` means **NanoChat's PE coefficient table evaluated by the official GNS backend**. It is not numerically identical to native NanoChat PE: the official GNS package owns its internal normalization/precision/restart path. Likewise, the preset restart `(2,)` is a starting configuration, not a universal constant. Autotune/validate restart placement for each coefficient schedule on the target GPU before a serious speed/quality run.

The official GNS implementation is a GPU performance dependency with hardware/software requirements of its own. The reference implementation in this package is the correctness oracle; no throughput claim should be made from it.

## Dion naming: fractional mechanism vs current Microsoft `Dion3`

The canonical internal update rule in this package is **`fractional_ef`**. It implements the selected-submatrix/error-feedback mechanism that we want to ablate independently. Current Microsoft Dion describes `Dion3`/`NorDion2` as a combined Dion2-selection + NorMuon method, and its current README notes that `mu` has different semantics in Dion2 and Dion3. Therefore this NanoChat harness does **not** claim API/state equivalence to the current `microsoft.dion.Dion3` class.

The production preset combines `fractional_ef` with row NorMuon, making it conceptually close to the current Dion3 family while retaining attribution-clean switches. If an experiment or report needs the label **exact Microsoft Dion3**, run an explicit parity/port validation against the pinned Dion implementation rather than inferring equivalence from this harness. The canonical role key is `fractional_roles`; `dion3_roles` remains a deprecated JSON alias for older configs.

## Configuration precedence

For research presets the precedence is: **preset defaults < explicit setup arguments (`ortho_fraction`, `beta2`) < JSON override**. Unknown keys, unknown roles, invalid head-group divisors, unsupported feature interactions, and out-of-range fractions are rejected rather than silently ignored. The old JSON key `rank_fraction` is accepted only as a deprecated alias for `ortho_fraction`; `dion3_roles` is likewise a deprecated alias for `fractional_roles`. Conflicting old/new spellings are errors.


## Benchmark interpretation: algorithm vs systems

Current NanoChat uses compiled/fused Muon kernels, whereas this research optimizer is deliberately a correctness-first composition of explicit PyTorch operations around NanoChat's existing collective scaffold. Therefore:

- **BPB vs tokens/steps** is the primary algorithmic comparison.
- Timing comparisons **within the research harness** are useful when the compared runs use the same grouping/fusion boundary (for example `dion_gns` vs `kj_gns`, or `role_split_dion_gns` vs `head_dion_gns`).
- A raw `native` vs research `optimizer_ms` difference is **not** an algorithm-only speed measurement because kernel fusion/dispatch differs.
- Before claiming a production throughput win, port/fuse the winning recipe and rerun end-to-end throughput and time-to-quality measurements.

The experiment ladder treats `role_split_dion_gns` as a required control because role-aware grouping itself changes collective/batch boundaries.

## MuonClip caveat for stock NanoChat

Current NanoChat RMS-normalizes Q and K **after projection** and then rescales the normalized activations. A positive scalar rescaling of a Q/K head's weights is therefore largely canceled by the activation normalization. Consequently, Kimi-style Q/K **weight-rescaling** MuonClip is intentionally not wired into the stock optimizer path.

`nanochat_muon_lab/muonclip.py` implements the MHA rule and a conservative GQA query-only rule for a model branch where QK activation normalization is disabled or where an explicit post-normalization head scale is being clipped. The experiment ladder treats this as an architecture-aware stability experiment, not a free optimizer toggle.

## Validation status

Build-time CPU validation currently passes **59/59 tests**, covering:

- Keller/NS and GNS reference equivalence checks;
- GNS restart/epsilon/backend-contract behavior;
- MuonEq;
- Muon+ normalization modes;
- NorMuon norm preservation;
- per-head split/merge;
- batched classic head-wise updates;
- batched fractional-EF updates on both tall and wide matrices;
- separation of fractional LR compensation from weight-decay strength;
- strict configuration/override precedence and role validation;
- native shape-only grouping parity plus the role-split control;
- current-vs-historical Keller momentum scalar equivalence at constant beta;
- checkpoint/installer transactional guards and effective parameter-group signatures;
- executable diagnostics definitions (RMS, head/group dispersion, polar residual, selection coverage, finite checks);
- GQA-aware role grouping;
- MuonClip MHA/GQA semantics and the QK-normalization scale-invariance caveat.

The package has **not** been executed end-to-end on your exact pinned NanoChat checkout/H200 environment here. Research optimizer-state resume is intentionally restricted to the same recorded world size and matching Muon-lab configuration; topology/layout changes require a fresh optimizer state unless you add an explicit migration. Before interpreting training results, run the correctness and reproduction gates in `EXPERIMENT_LADDER.md`.
