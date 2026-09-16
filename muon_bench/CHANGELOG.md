# Changelog

## 0.5.0 — 2026-09-16

- Fourth full audit.
- Added executable, unit-tested diagnostic definitions for update RMS, head/group norm dispersion, polar residual, selection coverage, and finite checks.
- Documented canonical fractional orientation: for tall matrices, selected canonical rows correspond to original columns; optional row-NorMuon follows those selected coordinates.
- Wired `polar_eps` into the official GNS constructor and cache key.
- Added strict GNS restart validation and no-op configuration rejection.
- Documented current official-GNS dispatch: non-square matrices use the Gram path while square matrices use the backend's standard NS path.
- Strengthened strict configuration tests for head grouping, fractional LR compensation, overlap between per-head/fractional roles, and raw update-rule overrides.
- Verified package installation/import from an arbitrary working directory using `--no-build-isolation --no-deps` in the offline artifact environment.
- 59 CPU tests passing at freeze preparation.

## 0.4.0 — 2026-09-16

- Third full audit.
- Canonical internal fractional rule renamed from ambiguous `dion3` to `fractional_ef`; canonical role key is `fractional_roles`, with deprecated aliases retained at config boundaries.
- Added effective parameter-group signature to resume metadata.
- Added schedule-aware local standard-NS normalization epsilon (`1e-7` Keller, `1e-6` NanoChat PE with safety 1.01).
- Strengthened installer staging/rollback.
- Updated GNS/Dion source assumptions against current upstream documentation.
- Added tests for legacy canonicalization, normalization epsilon, parameter-group signatures, failed reinstall preservation, and package version consistency.

## 0.3.0

- Added role-split systems control, official-GNS wrapper hardening, cautious-WD regression tests, and checkpoint configuration guards.
