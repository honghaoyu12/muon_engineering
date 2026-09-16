# Audit Report — NanoChat Muon Benchmark Package v0.5.0

**Audit date:** 2026-09-16  
**Audit pass:** fourth full pass

This revision re-audits implementation semantics, NanoChat integration boundaries, external GNS/Dion assumptions, configuration/checkpoint safety, documentation definitions, and package reproducibility.

## Major findings retained from v0.4.0

1. **Dion3 naming ambiguity removed.** The canonical internal rule is now `fractional_ef`. Current Microsoft Dion uses `Dion3` as `NorDion2` (fractional selection/error feedback + NorMuon) and documents different `mu` semantics for Dion2 vs Dion3. `dion3_roles` remains only as a deprecated config alias for canonical `fractional_roles`.
2. **Resume guard strengthened.** Checkpoints now compare the resolved research configuration **and the effective parameter-group signature** (LRs, WD/AdamW settings, matrix algorithm metadata, grouping/head information, and parameter shapes), not only the named preset/JSON flags.
3. **PE normalization made exact in the local reference path.** Keller standard NS defaults to epsilon `1e-7`; current NanoChat PE standard NS uses the current `1.01 * ||X||_F + 1e-6` convention. GNS keeps backend/reference semantics separate.
4. **Installer rollback strengthened.** New outputs are staged, previous lab outputs are backed up, and a partial commit attempts to restore the prior package/trainer. Source-anchor validation still occurs before any destination replacement.
5. **External GNS assumptions tightened.** Documentation now records current upstream float16/epsilon semantics and the fact that H200 is not explicitly named in the upstream hardware list; H200 is an explicit validation gate.
6. **Package metadata hardened.** Version is bumped consistently to `0.5.0`; editable test installation is documented without forcing replacement of the pinned CUDA/PyTorch stack.

## Important fixes retained from earlier passes

- `ortho_fraction` is canonical; `rank_fraction` is deprecated and conflicts fail loudly.
- Configuration precedence is `preset < explicit setup arguments < JSON override`; unknown keys/roles fail loudly.
- Fractional selection does not silently enable NorMuon.
- Unsupported fractional + factored-NorMuon and fractional + cautious-WD interactions are rejected.
- Native shape-only grouping is preserved for baseline/GNS/polynomial rungs; `role_split_dion_gns` isolates the role-grouping systems confound before per-head geometry.
- Keller reference finite precision and decoupled-WD semantics have regression guards.
- Current-NanoChat cautious WD uses the shape-scaled effective Muon LR and has a separate regression guard.
- Fixed five-step coefficient schedules cannot silently extend by repeating the final coefficient.
- GQA Q/K/V head metadata and dynamic grouping divisibility are validated.
- Official-GNS list/restart/output-shape contracts are checked at the wrapper boundary.
- MuonClip helpers reject malformed/nonfinite inputs and remain disabled on stock NanoChat because QK normalization largely cancels scalar weight rescaling.
- Algorithmic convergence claims are separated from fusion-dependent native-vs-research timing claims.

## Validation completed in the artifact environment

- **59/59 CPU tests pass** after this audit pass.
- All Python modules compile successfully.
- The tests now include version consistency, schedule-specific PE/Keller normalization epsilon, legacy-name canonicalization, parameter-group resume signatures, failed-reinstall preservation, external-GNS contract tests, fractional wide/tall/f=1/error-feedback behavior, LR/WD separation, grouping controls, MuonClip, and the previous math/configuration tests.

## Still required before expensive training

1. Run the installer against the **exact pinned NanoChat commit** chosen for experiments. Public `master` integration assumptions were rechecked on 2026-09-16, but this environment cannot execute that repository/GPU stack end-to-end.
2. Run single-GPU CUDA smoke tests, then multi-rank collective tests on the target H200 stack.
3. Install and validate official GNS on H200; upstream currently names H100/B200/B300 rather than H200 explicitly.
4. Numerically validate/autotune GNS restarts for each carried-forward coefficient table.
5. Run same-world-size save/resume equivalence. Treat topology changes as requiring explicit state migration or fresh optimizer state.
6. If the final method is to be called **Microsoft Dion3**, run an exact semantic parity/port check against a pinned Microsoft Dion implementation.
7. Treat BPB-vs-tokens/steps as the clean algorithmic comparison until the winning recipe is fused/ported for fair end-to-end throughput measurement.


## Additional v0.5.0 findings

- Added executable diagnostics definitions (`DIAGNOSTICS.md` + `nanochat_muon_lab/diagnostics.py`) so experiment terms such as update RMS, head-wise CV, polar residual, and selection coverage are mathematically unambiguous.
- Documented canonical-orientation semantics for fractional row-NorMuon: on a tall original matrix, selected canonical rows are original columns.
- Documented current official-GNS square-matrix dispatch: `gns_official` does not imply the Gram kernel is used for every shape.
- Tightened GNS restart/no-op configuration validation and wired `polar_eps` through the official backend.
