# Upstream Source Snapshot

**Checked:** 2026-09-16

This package is designed for a **pinned** NanoChat checkout. The audit compared integration assumptions with public `master` on the date above, but `master` is not a reproducible experiment identifier. Record the exact commit SHA in every run.

## NanoChat assumptions verified on current public source

- `GPT.setup_optimizer` sends all `transformer.h` parameters to Muon and groups them **by matrix shape**.
- The current AdamW auxiliary groups/hyperparameters match those mirrored in `setup.py`.
- `MuonAdamW._compute_muon(group, info, gather_list, rank)` still owns the Muon rank chunk and uses reduce-scatter/all-gather plus gather-copy-back in the same contract overridden by `ResearchMuonAdamW`.
- Native Muon is a fused/compiled momentum -> MuonEq -> Polar Express -> Frobenius snap -> factored variance -> cautious-WD/update path.
- Current native PE normalization is `X / (1.01 * ||X||_F + 1e-6)`; current MuonEq row norms clamp at `1e-6`.
- Current native Muon shape scaling is `sqrt(max(1, rows/cols))` and that scaled LR is used by both the Muon update and cautious-WD term.
- The current base-training Muon momentum schedule ramps `0.85 -> 0.97` over 400 steps, stays at `0.97`, then reaches `0.90` over LR warmdown; Muon WD cosine-decays to zero.
- Current Q/K activations are RMS-normalized after RoPE and then each multiplied by `1.2`, which is why scalar Q/K weight-rescaling MuonClip is not a meaningful stock-NanoChat optimizer-only toggle.

## GNS assumptions verified on current upstream README

- Algorithmic normalization uses epsilon `1e-7` and the documented fast path casts to `float16`.
- `GramNewtonSchulz` accepts a list-of-lists coefficient table and restart indices such as `[2]`.
- Upstream currently recommends one restart for five-step NS as a speed/stability starting point and provides restart autotuning.
- Named requirements are PyTorch >=2.7.1, CUDA >=12.9, and NVIDIA H100 or B200/B300. H200 is Hopper but is not explicitly named in that README, so this package treats H200 as a required empirical installation/kernel gate rather than asserting support.

## Microsoft Dion naming verified on current upstream README

- Dion2 applies submatrix selection/error feedback.
- Current `Dion3` is exported as an alias of `NorDion2`, combining Dion2-style selection/error feedback with NorMuon.
- The README explicitly notes that `mu` feeds Dion2's error-feedback decay but Dion3's momentum.
- Therefore this package uses the internal name `fractional_ef` for its attribution-clean selected-submatrix/error-feedback rule. `fractional_ef + row NorMuon` is a Dion3-family candidate, not an assertion of state/update equivalence to the current Microsoft `Dion3` class.
- Current Dion2/Dion3/Muon/NorMuon support single-device, DDP, and FSDP2, but not FSDP2+TP; only the legacy Dion implementation currently supports the TP combination.

## Primary upstream references

- NanoChat training script: https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py
- NanoChat optimizer: https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py
- NanoChat GPT / parameter grouping: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
- Keller–Jordan Muon: https://github.com/KellerJordan/Muon
- Gram Newton–Schulz: https://github.com/Dao-AILab/gram-newton-schulz
- Microsoft Dion / Dion3: https://github.com/microsoft/dion
- Dion3 paper: https://arxiv.org/abs/2608.11612
- NorMuon: https://arxiv.org/abs/2510.05491
- Muon+: https://arxiv.org/abs/2602.21545
- MuonEq: https://arxiv.org/abs/2603.28254
- Kimi K2 / MuonClip: https://arxiv.org/abs/2507.20534

## Integration boundary

The installer validates exact textual anchors and stops before committing new outputs if those anchors no longer match. Passing that check is **necessary but not sufficient**. Run the single-GPU, multi-rank, official-GNS, and same-world-size save/resume gates in `EXPERIMENT_LADDER.md` on the exact pinned commit and target environment.


## Additional audit facts used in v0.5.0

- Current official `gram-newton-schulz` exposes `ns_epsilon`, accepts explicit coefficient and restart lists, preserves the input dtype at the API boundary, and dispatches non-square matrices to the Gram implementation while square matrices use its standard Newton–Schulz path. Treat `gns_official` as a backend selection, not proof that every matrix shape ran the Gram kernel.
- Current Microsoft Dion distinguishes the standalone fractional selection/error-feedback family from the NorMuon-combined family. This package therefore uses `fractional_ef` for the independently ablated mechanism and reserves “exact Microsoft Dion3” for a dedicated parity port/comparison.
- In the package's canonical fractional orientation, tall matrices are transposed so selection occurs along the smaller dimension. Optional row-NorMuon state therefore follows selected canonical coordinates; for a tall original matrix those coordinates are original columns.
