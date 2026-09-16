# NanoChat Muon Benchmark Package Review

**Review date:** 2026-09-16  
**Package reviewed:** `muon_bench` revision `0.5.0`  
**NanoChat checkout inspected:** `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (`master`, July 3, 2026)

## Executive Summary

`muon_bench` is a serious research harness rather than an ad hoc optimizer fork. Its strongest design choice is attribution discipline: the package separates the Keller-Jordan reference transform, GNS backend, coefficient schedule, attention-head grouping, MuonEq, Muon+, NorMuon, and fractional error feedback into independently selectable stages.

The package is well documented and has strong defensive configuration behavior. It preserves NanoChat's optimizer-owned distributed communication and AdamW groups, handles GQA head counts explicitly, distinguishes the isolated `fractional_ef` mechanism from exact Microsoft Dion3 semantics, and makes important numerical and timing caveats visible.

The main limitation is validation scope. The CPU/unit tests cover the pure math and configuration boundaries, but the highest-risk paths remain unexecuted here: real NanoChat integration, CUDA kernels, official GNS, single-rank optimizer stepping, multi-rank collectives, and checkpoint/resume. The package should therefore be treated as a strong correctness-oriented research prototype until those gates pass.

## Findings

Findings are ordered by practical severity. `P1` means the issue can invalidate a benchmark or weaken experiment safety; `P2` means important validation or reproducibility work is missing; `P3` means a lower-severity maintainability or usability concern.

### P1: Dynamic head-switch metadata becomes stale

`set_attention_grouping()` mutates `heads_per_group` in live optimizer groups at [setup.py:366](nanochat_muon_lab/setup.py#L366). The generated trainer invokes it after the configured switch threshold at [install_into_nanochat.py:159](install_into_nanochat.py#L159).

The optimizer's canonical configuration and parameter-group signature are captured only during construction at [setup.py:356](nanochat_muon_lab/setup.py#L356). They are not recomputed after the phase switch. A checkpoint created after switching can therefore contain active full-matrix grouping while its `muon_lab_effective_config_json` still describes the initial per-head grouping.

This may not immediately prevent loading because PyTorch also serializes optimizer parameter-group fields, but it weakens the resume guard and makes the experiment record inaccurate. A resumed run can appear configuration-identical while the metadata does not describe the state actually used at save time.

**Recommendation:** record `initial_grouping` and `active_grouping` separately, or update the canonical config/signature transactionally when the switch occurs. Add a checkpoint test that saves after the switch and verifies that the recorded configuration matches the live groups.

### P1: Distributed and GPU paths lack executable coverage

The research optimizer overrides NanoChat's private `_compute_muon()` contract at [research_optimizer.py:459](nanochat_muon_lab/research_optimizer.py#L459), including rank ownership, sharded state allocation, all-gather buffer reuse, and copy-back. The parent implementation is in [nanochat/optim.py:362](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/optim.py#L362).

The current tests exercise helper methods with ordinary tensors, but do not run the full optimizer contract. Missing execution coverage includes:

- a real single-GPU `optimizer.step()` through the inherited communication scaffold;
- a multi-rank reduce-scatter/all-gather update;
- groups with more ranks than parameters, including padded ownership;
- real checkpoint save/resume with optimizer state;
- the compiled NanoChat model and trainer;
- official GNS on CUDA;
- FP8 training, if the benchmark is expected to support NanoChat's `--fp8` path.

The package itself acknowledges these missing gates in [AUDIT_REPORT.md:38](AUDIT_REPORT.md#L38). Until they pass, the implementation should not be considered a validated drop-in NanoChat optimizer.

**Recommendation:** add a GPU smoke-test script and a small `torchrun` integration test. Verify parameter equality across ranks after gather, state shapes, padding behavior, finite values, and one-step parity between world size 1 and a multi-rank run where numerical ordering permits it.

### P1: Native-vs-research timing is structurally confounded

Native NanoChat executes a compiled/fused Muon step, while `ResearchMuonAdamW` performs explicit Python/PyTorch operations in [research_optimizer.py:251](nanochat_muon_lab/research_optimizer.py#L251). This changes kernel fusion, launch count, temporary allocations, and synchronization behavior.

The package documents this correctly at [README.md:130](README.md#L130), but timing fields can still be misread. A raw native-versus-research `optimizer_ms` difference is not an algorithm-only speed comparison.

**Recommendation:** use BPB versus tokens/steps for algorithmic claims, compare timing only within equivalent research grouping/backend controls, and fuse the winning algorithm before reporting production throughput or time-to-quality improvements.

### P2: Resume validation does not fingerprint the training schedule

The installer compares the preset, effective optimizer configuration, and world size at [install_into_nanochat.py:100](install_into_nanochat.py#L100). It does not compare all state-affecting trainer inputs, including:

- `num_iterations` and target-token-derived horizon;
- `warmup_steps`;
- `warmdown_ratio`;
- `final_lr_frac`;
- batch-size and LR scaling inputs;
- tokenizer/data-shard identity.

These values affect future LR, momentum, and weight-decay schedules. A user can resume with an identical optimizer signature but a different future optimization trajectory.

**Recommendation:** store a canonical schedule/data fingerprint in checkpoint metadata and reject or explicitly warn on mismatches. The fingerprint should include the resolved iteration count, LR and WD schedule parameters, batch configuration, tokenizer/data version, and relevant model/runtime settings.

### P2: Official GNS is contract-tested but not version-pinned or numerically validated

The official backend is imported dynamically at [research_optimizer.py:194](nanochat_muon_lab/research_optimizer.py#L194). The package does not declare a `gram-newton-schulz` version constraint, and unit tests substitute a fake implementation rather than exercising the real backend.

That dependency strategy is reasonable for avoiding an unconditional CUDA installation, but reproducible experiments require recording the exact GNS package version or commit. Numerical parity also needs to be measured on representative NanoChat matrix shapes and dtypes, including square and non-square dispatch paths.

The experiment ladder correctly calls for these checks at [EXPERIMENT_LADDER.md:67](EXPERIMENT_LADDER.md#L67).

**Recommendation:** add a GPU validation command that reports the imported package version, backend dispatch path where observable, finite-value status, update cosine/residual versus the reference implementation, and measured kernel timing by shape.

### P2: The test extra is incomplete for standalone installation

[pyproject.toml:11](pyproject.toml#L11) declares only `pytest` under the test extra, while tests import PyTorch immediately, for example at [tests/test_research_optimizer.py:5](tests/test_research_optimizer.py#L5).

The documentation intentionally expects the package to inherit NanoChat's pinned PyTorch/CUDA environment, so this is not necessarily a design error. It does mean that a fresh `pip install -e '.[test]'` does not produce a runnable test environment.

**Recommendation:** document the inherited-environment requirement directly in `pyproject.toml` or provide separate extras such as `test-cpu` and `test-nanochat`. Avoid forcing a PyTorch wheel that could replace a working CUDA stack, but make the expected installation path explicit.

### P2: Installer tests validate synthetic fixtures, not the pinned checkout

The installer tests use a minimal synthetic `base_train.py` fixture at [tests/test_installer.py:6](tests/test_installer.py#L6). This is useful for testing transactionality and rollback, but it does not prove that the actual NanoChat source matches the patch anchors.

The installer intentionally depends on exact textual anchors through `replace_once()` at [install_into_nanochat.py:26](install_into_nanochat.py#L26). That is safer than silently patching changed code, but it makes a real-checkout smoke test essential.

**Recommendation:** run the installer in CI against the exact NanoChat SHA used by experiments, import the generated trainer, and at least parse/compile it. Keep the fixture tests for rollback coverage, but do not treat them as source compatibility coverage.

### P3: Experiment provenance is described but not fully captured automatically

The experiment ladder asks users to record NanoChat SHA, package checksum, PyTorch/CUDA versions, GPU, world size, GNS version, and compute mode at [EXPERIMENT_LADDER.md:35](EXPERIMENT_LADDER.md#L35). The generated trainer records CLI configuration and resolved optimizer setup, but the environment facts are not all emitted automatically into checkpoint metadata.

Manual recording is easy to omit and makes later benchmark comparisons harder to audit.

**Recommendation:** add an environment manifest to run metadata containing NanoChat SHA, package revision/checksum, Python/PyTorch/CUDA versions, GPU names and compute capabilities, world size, GNS version, Flash Attention mode, and compute dtype.

## Code and Design Assessment

### Optimizer architecture

Subclassing NanoChat's `MuonAdamW` is the right integration boundary. It preserves the existing gradient synchronization and optimizer-owned sharding instead of introducing a second distributed execution model. The explicit `_project()` pipeline at [research_optimizer.py:251](nanochat_muon_lab/research_optimizer.py#L251) is easy to audit:

```text
optional head grouping
    -> MuonEq
    -> standard/GNS polar map
    -> Muon+ post-normalization
    -> optional NorMuon
    -> LR scaling, WD, parameter update
```

The separation between solver backend and coefficient schedule is particularly valuable. It prevents a GNS performance experiment from silently changing the polynomial being evaluated.

### Parameter grouping

The role classifier in [setup.py:200](nanochat_muon_lab/setup.py#L200) correctly distinguishes Q, K, V, output projection, MLP up/down, and value-embedding gates. Q uses `n_head`; K and V use `n_kv_head`, matching NanoChat's GQA layout.

Preserving shape-only grouping for clean baselines is a strong experimental choice. The explicit `role_split_dion_gns` control also recognizes that role-aware grouping changes collective boundaries and batching behavior independently of the mathematical update.

### Fractional error feedback

The canonical name `fractional_ef` is preferable to claiming exact Microsoft Dion3 equivalence. The implementation makes selection, update-LR compensation, raw-LR weight decay, and residual decay separate. It also documents the tall-matrix convention: after canonical transposition, selected rows correspond to original columns.

That said, the behavior should be validated against a pinned external Dion implementation before using any Dion-specific result in a paper or report. The package explicitly reserves exact Dion3 claims for that future parity check, which is the correct boundary.

### Numerical semantics

The package is careful about Keller versus NanoChat normalization conventions, bf16 working tensors, epsilon choices, GNS backend-owned precision, restart placement, shape scaling, and weight-decay semantics. These distinctions are easy to lose in optimizer experiments and are a major strength of the implementation.

The remaining risk is not that the semantics are undocumented; it is that most of them are only tested with CPU/reference tensors. The GPU backend can still differ in finite precision, dispatch, or unsupported-shape behavior.

### MuonClip

Keeping MuonClip out of the stock NanoChat optimizer path is correct. NanoChat RMS-normalizes Q/K activations after projection at [nanochat/gpt.py:102](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L102), so scalar weight rescaling is largely canceled. Providing architecture-aware helpers without pretending they are a useful stock optimizer toggle is a good design decision.

## Test Coverage

The package has broad CPU-level coverage for:

- coefficient schedule bounds;
- standard NS and reference GNS behavior;
- MuonEq and Frobenius snap;
- row/column Muon+ modes;
- NorMuon norm preservation;
- head split/merge;
- classic batched updates;
- fractional wide/tall updates and error-feedback decay;
- fractional LR compensation versus WD strength;
- GQA role grouping;
- strict override validation;
- official-GNS constructor/output-shape contracts using fakes;
- installer transactionality and rollback;
- package metadata and checksum consistency.

The test inventory contains 57 test functions plus parametrized cases, consistent with the documented 59-case result in [AUDIT_REPORT.md:34](AUDIT_REPORT.md#L34).

The missing coverage is concentrated in the most integration-sensitive behavior:

- actual NanoChat model construction and role classification;
- generated trainer import and execution;
- `ResearchMuonAdamW.step()` with the real parent optimizer;
- multi-process communication and padding;
- same-world-size checkpoint/resume equivalence;
- post-switch checkpoint metadata;
- CUDA/bf16/FP8 numerical behavior;
- official GNS installation, dispatch, and performance;
- deterministic comparison of native and research paths.

## Verification Performed for This Review

- All package Python files passed AST parsing.
- Python bytecode compilation passed.
- `PACKAGE_MANIFEST.sha256` verification passed.
- Pytest collection could not complete in this environment because PyTorch is not installed. The failure was dependency-related (`ModuleNotFoundError: torch`), not a reported assertion failure.
- CUDA, official GNS, single-GPU, multi-rank, and checkpoint/resume execution were not available.
- The inspected NanoChat checkout is clean at `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

## Recommended Validation Order

1. Run the installer against the exact NanoChat SHA used for the experiment and compile/import the generated trainer.
2. Run a tiny native reproduction and compare it with `--muon-lab-preset native`.
3. Run a single-GPU `kj_reference` smoke test with finite checks and captured optimizer state.
4. Run a multi-rank smoke test covering all-gather, padding, and state ownership.
5. Test save/resume before and after a dynamic head-group switch.
6. Install official GNS, record its exact version, and compare representative shapes against the reference solver.
7. Only then run the staged experiment ladder, beginning with native and Keller baselines.
8. Fuse/port any winning recipe before making end-to-end throughput claims.

## Final Recommendation

Approve this package as a controlled research harness with the P1/P2 items tracked as validation and metadata work. Do not yet treat `production_candidate` as production-ready, do not call the fractional path exact Microsoft Dion3, and do not interpret native-versus-research optimizer timing as an algorithm-only speedup.

The design and documentation quality are strong enough to justify GPU validation. The next engineering priority is not adding more optimizer features; it is closing the real-checkout, distributed, CUDA, GNS, and checkpoint-resume validation boundary and fixing the stale dynamic-group metadata.
