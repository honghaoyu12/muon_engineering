# Response to Review and Revision Plan — NanoChat Muon Benchmark Project

**Revision:** post-audit v2\
**Date:** 2026-09-16\
**Reviewed package:** `muon_bench` v0.5.0\
**Primary review:** `REVIEW.md`\
**NanoChat checkout inspected by the review:** `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`\
**Purpose:** convert the review and follow-up audit into a single implementation and validation specification for the next package revision.

---

## 1. Executive response

We accept the review's central conclusion.

The package is already strong enough at the **research-design** level to justify continued work: it separates the Keller–Jordan reference transform, orthogonalizer backend, polynomial schedule, parameter grouping, head-wise geometry, MuonEq, Muon+/NorMuon, and fractional error feedback into independently testable stages. The review also positively assesses the preservation of NanoChat's optimizer-owned distributed execution model, the GQA-aware role classifier, the neutral `fractional_ef` naming, and the decision not to treat MuonClip as a stock NanoChat toggle.

The main weakness has shifted. It is no longer lack of algorithmic modularity. It is that the highest-risk execution paths have not yet been validated end to end in the real NanoChat/CUDA/checkpoint environment. Distributed execution is a separate deferred qualification track because the initial optimizer campaign is intentionally single-GPU.

The next release should therefore be an **integration-validation release**, not a feature release.

The project should advance through

\[
\boxed{
\text{reference correctness}
\rightarrow
\text{real-system correctness}
\rightarrow
\text{controlled algorithmic evidence}
\rightarrow
\text{production evidence}
}
\]

and should not add new optimizer mechanisms until the current implementation can pass the relevant qualification gates defined below.

---

## 2. Evidence classification

Not every review item is the same kind of problem. The next release should distinguish confirmed implementation defects from missing validation and from claim/reporting constraints.

| Review item | Classification | Required action |
|---|---|---|
| Dynamic head-switch metadata becomes stale | **Confirmed provenance/implicit-restore bug** | Make runtime state explicit and validate it before any dynamic-grouping experiment |
| GPU path lacks executable coverage | **Blocking validation gap** | Close before substantive single-GPU research runs |
| Distributed paths lack executable coverage | **Deferred scaling gap** | Close before distributed or production-throughput claims |
| Native-vs-research timing is structurally confounded | **Claim limitation** | Enforce reporting rules; fuse winner before production claims |
| Resume validation omits training trajectory | **Confirmed resume/provenance gap** | Add trajectory identity and exact-resume checks |
| Official GNS is not pinned/numerically validated | **Blocking dependency-validation gap for GNS phases** | Pin and validate before Phase C |
| Test extra does not install PyTorch | **Packaging/usability issue** | Document inherited environment; add preflight checks |
| Installer tests use synthetic fixtures only | **Integration gap** | Add exact-checkout qualification |
| Provenance is not fully automatic | **Reproducibility gap** | Generate immutable run manifest + checkpoint/runtime metadata |
| FP8 unvalidated | **Deferred compatibility task** | Validate after bf16 unless immediately required |

### Severity nomenclature

The coding-agent review already defines `P1/P2/P3`. This document does **not** introduce a competing `P0/P1/P2` system.

Instead, every implementation task is assigned one of these gate labels:

- **BLOCKING-BASE** — required before the single-GPU native/Keller benchmark campaign;
- **DEFERRED-DISTRIBUTED** — required before distributed scaling, cluster-utilization, or production-throughput claims;
- **BLOCKING-GNS** — required before GNS experiments;
- **BLOCKING-DYNAMIC** — required before dynamic head-grouping experiments;
- **BLOCKING-PRODUCTION** — required before production throughput claims;
- **NONBLOCKING** — useful hardening that does not block the relevant scientific phase.

---

# Part I — Confirmed implementation fixes

## 3. Dynamic grouping must become explicit optimizer runtime state

### 3.1 Confirmed defect

Current v0.5.0 records the effective optimizer configuration and parameter-group signature at optimizer construction time. Later, `set_attention_grouping()` mutates live `heads_per_group` values without regenerating metadata that explicitly describes the active phase.

PyTorch optimizer state dictionaries normally serialize parameter-group fields, so the active grouping may currently be restored implicitly by `optimizer.load_state_dict(...)`. That behavior has not been qualified here, is not represented in checkpoint provenance, and is not validated against the requested grouping schedule. The confirmed defect is therefore an ambiguous provenance and implicit-restore contract, not a demonstrated loss of grouping state.

A post-switch checkpoint can contain:

```text
saved construction-time description:  heads_per_group = 1
live optimizer at checkpoint time:     heads_per_group = H
saved optimizer param groups:          implementation-dependent restore source
```

The next revision must make the active phase explicit, cross-check it against saved optimizer parameter groups, and validate the final live state after loading.

**Gate:** `BLOCKING-DYNAMIC`.

### 3.2 State model

Do not solve this by overwriting the original experiment definition after a transition. Keep three distinct objects.

#### A. Immutable initial optimizer definition

Example:

```yaml
initial_grouping:
  q_heads_per_group: 1
  k_heads_per_group: 1
  v_heads_per_group: 1
```

This answers: **what experiment was requested at step 0?**

#### B. Immutable grouping schedule

Example:

```yaml
grouping_schedule:
  mode: two_stage
  switch_fraction: 0.35
  resolved_switch_step: 3500
  target: full
```

A future staged version may look like:

```yaml
grouping_schedule:
  mode: staged
  stages:
    - {start_fraction: 0.00, q_hpg: 1,  kv_hpg: 1}
    - {start_fraction: 0.25, q_hpg: 2,  kv_hpg: 2}
    - {start_fraction: 0.50, q_hpg: 4,  kv_hpg: 4}
    - {start_fraction: 0.70, q_hpg: full, kv_hpg: full}
```

Fractions are user-facing inputs. Before training, every transition must be resolved to an absolute step using the final trajectory and stored in the immutable schedule. A trajectory-changing resume must not reinterpret an already-started schedule under a new horizon without an explicit grouping-schedule migration.

This answers: **what transition policy defines the experiment?**

#### C. Mutable runtime optimizer state

Example:

```yaml
runtime_grouping_state:
  q_heads_per_group: 16
  k_heads_per_group: 16
  v_heads_per_group: 16
  active_stage: 1
  transition_count: 1
  last_transition_step: 3500
```

This must be derived from the **live optimizer groups** at checkpoint time.

This answers: **what phase was the optimizer actually in when this checkpoint was created?**

### 3.3 Required helper API

Add explicit helpers, for example:

```python
get_active_grouping_state(optimizer) -> dict
apply_active_grouping_state_(optimizer, state: dict) -> None
build_live_group_signature(optimizer) -> dict
```

`apply_active_grouping_state_` must validate all affected groups before mutating any of them.

The transition operation should update the optimizer groups and runtime-state record atomically from the perspective of checkpointing.

### 3.4 Correct resume sequence

For a checkpoint that includes optimizer state:

1. construct the optimizer from the validated immutable initial definition;
2. validate that the requested grouping schedule, including resolved transition steps, matches the checkpoint schedule;
3. read the explicit checkpoint runtime grouping state and the saved optimizer `param_groups`;
4. cross-check both representations and reject disagreement;
5. apply the active grouping transactionally to the newly constructed optimizer;
6. build and compare a pre-load structural signature that excludes legitimately scheduled mutable values such as current LR, momentum, and WD;
7. call `optimizer.load_state_dict(...)`;
8. rebuild the active grouping and live structural signature from the loaded optimizer and compare them again with checkpoint metadata;
9. restore the transition stage/count so the training loop cannot emit or execute the same transition twice.

The post-load check is mandatory because `optimizer.load_state_dict(...)` may restore parameter-group fields and overwrite values configured before the call.

Conceptually:

\[
\text{initial definition}
\rightarrow
\text{cross-check saved runtime state}
\rightarrow
\text{restore and validate pre-load structure}
\rightarrow
\text{load state}
\rightarrow
\text{validate final live structure}.
\]

### 3.5 Required regression tests

Add all of the following:

- checkpoint before the switch;
- checkpoint immediately after the switch;
- resume after the switch;
- inconsistent requested schedule must reject exact optimizer-state resume;
- post-switch resume must not execute the transition again;
- explicit runtime grouping must agree with grouping fields in the saved optimizer `param_groups`;
- live group signatures generated both before and after optimizer-state load must match the expected structural signature;
- staged schedules must reject illegal/non-divisible Q/K/V group sizes transactionally.

---

## 4. Distinguish expected state schema from observed live state

The previous document mixed these concepts in the proposed state fingerprint.

Many optimizer states are allocated lazily. Therefore a construction-time fingerprint cannot truthfully contain every observed tensor dtype/layout.

Use two levels.

### 4.1 Expected state schema

Available at construction time. Include:

- parameter-group order;
- parameter shapes;
- logical roles;
- update rule;
- orthogonalizer and coefficient schedule;
- expected momentum/error-feedback state type;
- NorMuon mode;
- active head grouping;
- world size / ownership partition rule;
- expected state dtype policy.

### 4.2 Observed checkpoint-state signature

Generated immediately before saving a checkpoint. Include, for each owned state entry where practical:

- state tensor names;
- actual shapes;
- actual dtypes;
- device class;
- ownership index/rank mapping;
- active grouping state;
- finite-value status in validation/debug mode.

This gives the resume guard something that describes the state that actually exists, not merely what was expected to exist. Because optimizer state is sharded, each rank must either embed its observed signature in its optimizer shard or participate in a collective that gathers rank-local signatures for rank-0 checkpoint metadata. The chosen representation must detect a missing rank signature.

**Gate:** `BLOCKING-BASE` for basic state schema; `BLOCKING-DYNAMIC` for active grouping state.

---

# Part II — Qualification gates

## 5. Use dependency-specific gates, not one monolithic Q0–Q8 chain

A single linear qualification chain over-constrains early work. Native/Keller baselines do not depend on GNS, and static head-wise experiments do not depend on dynamic-grouping checkpoint logic.

The next release should therefore implement four qualification families.

### 5.1 Qualification result contract

Every blocking gate must have all of the following before it is executed:

- an exact command and environment identity;
- a declared set of required tests;
- quantitative acceptance thresholds stored in a versioned `qualification_thresholds.json`;
- machine-readable results plus a human-readable summary;
- an explicit pass/fail result.

Thresholds must be committed before the corresponding result data is inspected. Phrases such as “within expected nondeterminism” are explanatory only and cannot replace a numeric threshold. Local integration tests may skip when hardware is unavailable, but a qualification command must fail if any test required by that gate is skipped or not collected.

---

## 6. Q-BASE — qualification required before baseline/Keller benchmark runs

### Q-BASE-0 — exact checkout and environment preflight

On the exact NanoChat SHA selected for experiments:

```bash
python ../muon_bench/install_into_nanochat.py
python -m py_compile scripts/base_train_muon_lab.py
python -m scripts.base_train_muon_lab --help
python -m nanochat_muon_lab.preflight
```

The preflight should report, without modifying the environment:

```text
NanoChat SHA and dirty/clean status
muon_bench version / package identity
Python version
PyTorch version
CUDA runtime/version
GPU names
GPU compute capabilities
world size / torch.distributed availability
compute dtype
Flash Attention mode when detectable
GNS identity when requested
```

The installer must remain fail-loud on source drift.

### Q-BASE-1 — explicit reproducibility/seed policy

The experiment plan calls for one-, three-, and five-seed comparisons, but the current package specification does not define how the seed is chosen.

The next trainer revision must make this explicit.

Preferred design:

```text
--seed <integer>
```

Requirements:

- establish the model RNG policy before parameter initialization;
- make the policy explicit for every rank;
- use a shared model-initialization seed so all ranks start from identical parameters;
- derive rank-local RNG streams only after synchronized model initialization, and only for operations that require independent streams;
- record the seed in immutable run provenance;
- document which RNG streams are controlled (`torch`, CUDA, Python/NumPy if used by the actual path);
- do not claim deterministic reproduction merely because the seed matches;
- preserve/reuse NanoChat's existing dataloader resume state rather than replacing it with the seed;
- checkpoint and restore RNG positions rather than attempting to reproduce them by reseeding.

If current NanoChat already has an internal seed policy in the chosen pinned SHA, expose and record that policy rather than layering a second conflicting mechanism on top.

### Q-BASE-2 — native harness reproduction

Run:

```text
unmodified scripts.base_train
```

and

```text
base_train_muon_lab --muon-lab-preset native
```

under the same configuration.

Acceptance criterion: behavior satisfies the preregistered native-reproduction thresholds for the chosen metric vector and run length. The threshold file must define, at minimum, the compared steps, validation-BPB tolerance, parameter or update comparison where enabled, and allowed runtime noise. The purpose is to prove that the copied/patched trainer does not alter the native path simply by existing.

### Q-BASE-3 — real single-GPU research optimizer step

Instantiate a real small NanoChat model and execute:

```text
forward
backward
optimizer.step()
```

for at least:

```text
kj_reference
one per-head preset
production_candidate / fractional path
```

GNS is not required for this gate; use a standard/reference orthogonalizer where necessary so the base integration can be qualified independently of the external GNS dependency.

Check:

- finite model parameters;
- finite allocated optimizer states;
- expected state shapes/dtypes;
- actual NanoChat role classification covers the intended matrix parameters;
- AdamW-only groups remain on their intended path;
- head reshaping is valid for real `n_head`/`n_kv_head` dimensions;
- every role with a finite, nonzero gradient and expected participation receives the intended update;
- parameters with zero/no gradient do not spuriously change except through explicitly applicable weight decay.

### Deferred-DISTRIBUTED-1 — distributed ownership and collective correctness

Run at least a two-rank test through the real parent optimizer communication scaffold.

After each step, verify gathered model parameters agree across ranks:

\[
W_i^{(0)} \approx W_i^{(1)}.
\]

Log:

- parameter/matrix ownership by rank;
- state ownership by rank;
- reduce-scatter input/output shapes;
- all-gather output shapes;
- copy-back targets.

#### Padding/zero-ownership subtests

Explicitly exercise both:

\[
N_{\rm matrices}\not\equiv0 \pmod{N_{\rm ranks}}
\]

and

\[
N_{\rm matrices}<N_{\rm ranks}.
\]

For example:

```text
5 matrices / 8 ranks
13 matrices / 8 ranks
```

Verify:

- ranks with zero real matrices are safe;
- padded entries never modify model parameters;
- stale buffer contents cannot leak into gathered updates;
- state shapes and ownership remain stable.

### Deferred-DISTRIBUTED-2 — world-size parity test with the same effective global gradient

A world-size comparison is meaningful only if both executions represent the same mathematical update.

Use one of these designs:

1. inject identical preconstructed gradients into the same tiny parameter set on world size 1 and world size 2; or
2. use exactly the same global batch, partitioned across ranks, and verify that gradient reduction produces the same effective global gradient before the optimizer transform.

Then compare the post-update parameters/directions.

Do **not** interpret differences from different data partitions as distributed-optimizer error.

Bitwise identity is not required when collective ordering changes floating-point accumulation, but any discrepancy must be bounded and explained.

### Q-BASE-6 — exact checkpoint/resume including dataloader state

Current NanoChat already saves/restores dataloader state as part of pretraining checkpointing. The Muon harness should extend that contract, not replace it.

For an exact-resume test compare:

```text
continuous run to step T
```

against

```text
run to step t
save checkpoint
resume
continue to step T
```

Validate:

- model parameters;
- optimizer state;
- loop/schedule state;
- CPU RNG state and CUDA RNG state for every rank/device;
- Python/NumPy RNG state if those streams are used by the qualified path;
- `GradScaler.state_dict()` whenever a scaler is active;
- the current and next training batches around the checkpoint boundary using NanoChat's existing `dataloader_state_dict` mechanism;
- run/trajectory fingerprints;
- expected numerical tolerance.

A data-identity fingerprint is not a substitute for the dataloader cursor/state. Both are needed:

\[
\boxed{
\text{dataset/tokenizer identity}
+
\text{dataloader resume state}
}.
\]

### Q-BASE-7 — short single-GPU integration smoke run

After the single-GPU tests pass, run a short non-GNS research preset on the available GPU and verify:

- successful forward/backward/optimizer completion;
- no NaN/Inf;
- expected local optimizer state shapes and dtypes;
- checkpoint creation and short exact resume;
- peak memory;
- basic profiler trace.

### Q-BASE exit

After `Q-BASE-0` through `Q-BASE-3`, `Q-BASE-6`, and the single-GPU `Q-BASE-7` pass, the project may begin substantive **Phase B native/Keller baseline experiments**. The deferred distributed track is not required for these optimizer comparisons.

---

## 7. Q-GNS — qualification required before Phase C or any promoted GNS run

### Q-GNS-0 — actual dependency pin

Recording a version is not the same as pinning it.

The run environment must install/use an exact GNS identity, such as:

```text
exact released package version
```

or

```text
exact Git commit SHA
```

`preflight` must verify that the imported implementation matches the expected identity before training starts. For a released package, verify the exact distribution version and artifact hash. For a VCS installation, verify PEP 610 `direct_url.json` or an equivalent immutable source identity. If the installed source cannot prove the configured identity, the GNS qualification must fail. A lockfile or container image digest should pin the surrounding PyTorch/CUDA environment.

Record:

```text
GNS version / commit
source/install identity
PyTorch version
CUDA version
GPU name
compute capability
coefficient table
restart positions
normalization epsilon
input/output dtype policy
```

### Q-GNS-1 — capture real pre-orthogonalizer tensors

Do not validate only on synthetic matrices of realistic shape.

Capture representative tensors **immediately before the orthogonalizer** from short NanoChat runs at multiple training phases:

```text
early
middle
late / warmdown
```

Include Q/K/V/O and MLP roles, and both square and rectangular matrices.

Supplement those captures with synthetic stress cases for unusually poor conditioning, row imbalance, and extreme aspect ratios.

### Q-GNS-2 — backend-equivalence reference

Compare official GNS against a matched pure-PyTorch implementation of the **same finite polynomial/restart scheme**.

This answers:

> Does the optimized GNS backend implement the intended finite polynomial update closely enough on the target dtype/hardware?

Measure:

\[
\cos(U_{\rm official},U_{\rm matched\ ref})
\]

and

\[
\frac{\|U_{\rm official}-U_{\rm matched\ ref}\|_F}
{\|U_{\rm matched\ ref}\|_F}.
\]

### Q-GNS-3 — polar-approximation reference

On small/medium matrices where it is affordable, separately compare the finite solver output against a high-precision SVD polar factor.

This answers a different question:

> How accurate is the selected finite-step polynomial as an approximation to the true polar factor?

Do not conflate backend parity with polar approximation error.

### Q-GNS-4 — residual/stability/runtime suite

For each role/shape family report:

- update cosine to the matched reference;
- relative Frobenius error;
- appropriate tall/wide polar residual;
- NaN/Inf rate;
- kernel/runtime distribution;
- backend dispatch path when observable;
- outlier matrices/shapes.

Run at least \(10^3\) representative matrices before long GNS training.

### Q-GNS-5 — restart validation

Restart `(2,)` is only a starting point.

For every coefficient schedule promoted beyond exploratory use, validate restart placement on the actual H200 stack and real captured tensors.

Do not silently reuse a restart selected for one polynomial as though it were universally optimal.

### Q-GNS exit

After `Q-GNS-0` through `Q-GNS-5` pass for the chosen schedule/backend, proceed to **Phase C GNS/polynomial experiments**.

---

## 8. Q-DYNAMIC — qualification required before Phase H

This gate is separate because static head-wise experiments do not require dynamic checkpoint semantics.

Required tests:

1. checkpoint before transition;
2. checkpoint immediately after transition;
3. exact resume after transition;
4. active live grouping equals saved runtime grouping;
5. restored live signature equals saved checkpoint signature;
6. inconsistent grouping schedule rejects exact optimizer-state resume;
7. resumed post-switch run does not execute the same transition twice;
8. staged schedule transitions are transactional and respect GQA divisibility;
9. continuous and resumed runs agree within expected tolerance.

Only after these pass should **Phase H dynamic grouping** begin.

---

## 9. Q-PRODUCTION — qualification required before production-speed claims

Even if the explicit research optimizer wins in BPB-vs-steps, do not claim a production speedup until:

- the winning recipe is fused/ported;
- the implementation class is comparable to the native/production baseline;
- end-to-end tokens/s is measured;
- wall-clock-to-fixed-BPB is measured;
- communication overlap is profiled;
- memory behavior is profiled;
- checkpoint/restart works in the fused implementation;
- the result transfers beyond the cheap ranking model.

---

# Part III — Resume and provenance model

## 10. Separate state compatibility from training-trajectory identity

### 10.1 State-compatibility fingerprint

This determines whether loading optimizer state is structurally/semantically valid.

Include immutable/structural fields such as:

- model parameter shapes and ordering;
- optimizer algorithm/preset;
- parameter-group boundaries/order;
- world size / ownership topology assumptions;
- logical roles;
- coefficient schedule;
- orthogonalizer backend;
- MuonEq/Muon+/NorMuon modes;
- fractional roles/fractions;
- expected state schema;
- initial grouping definition;
- checkpoint-local active grouping state.

A mismatch should be a hard error for exact optimizer-state resume unless an explicit state migration exists.

### 10.2 Training-trajectory fingerprint

This determines whether the resumed run follows the same intended future training trajectory.

Store **resolved values**, not merely raw CLI text, including:

- resolved `num_iterations`;
- resolved token horizon;
- warmup steps;
- resolved warmdown iterations/start step;
- final LR fraction;
- effective base LRs after any batch/depth scaling;
- weight-decay schedule parameters and effective base value;
- global batch size;
- device/microbatch size;
- gradient accumulation;
- sequence length;
- explicit seed policy/value;
- tokenizer identity/hash;
- dataset/shard identity/version;
- relevant compute precision/runtime flags.

Why resolved values matter: schedule behavior depends on quantities derived from the final iteration count and scaling rules, not only on the raw arguments used to produce them.

### 10.3 Resume modes

Use explicit semantics such as:

```text
--resume-mode exact
--resume-mode trajectory-change
--resume-mode weights-only
```

#### `exact`

- load model state;
- load optimizer state;
- restore dataloader/loop/runtime optimizer state;
- restore per-rank RNG positions and active `GradScaler` state before the next stochastic operation;
- require state-compatibility match;
- require trajectory match;
- reject mismatches by default.

#### `trajectory-change`

Use only for an intentional continuation experiment where optimizer-state compatibility remains valid but the future schedule/data trajectory is deliberately changed.

Requirements:

- explicit CLI selection;
- permanently record old and new trajectory identities;
- never present the continuation as an exact resume;
- preserve already-resolved grouping transition steps; changing those steps requires an explicit grouping-schedule migration, otherwise use `weights-only`.

#### `weights-only`

Use when changing optimizer algorithms/rungs.

Rule:

> Never migrate between experiment-ladder optimizer algorithms by silently loading old optimizer state.

Start a new run from model weights and initialize fresh optimizer state.

### 10.4 Fingerprint canonicalization

Every fingerprinted object must include a schema version and be serialized with one canonical JSON implementation: sorted keys, fixed separators, explicit normalization of tuples/enums/dtypes, and no device-specific object representations or unstable absolute paths. Hash canonical UTF-8 bytes with SHA-256. Unknown schema versions must fail closed for exact resume.

Dataset identity should use a versioned shard manifest containing stable shard identifiers, sizes, and content hashes where practical; tokenizer identity should hash the actual tokenizer artifacts. Hashing only a local path is insufficient.

---

## 11. Provenance should be immutable; runtime events should be append-only

The previous draft mixed mutable active grouping state into `run_manifest.json`. That would turn the run manifest into a moving target.

Use three artifacts instead.

### 11.0 Artifact location, ownership, and writes

All provenance artifacts must live in a unique run directory derived from an immutable run ID, not only NanoChat's depth-based default checkpoint directory. Starting a new run must refuse to reuse a nonempty run directory unless an explicit compatible resume is in progress.

Rank 0 exclusively creates `run_manifest.json` and appends `runtime_events.jsonl`. Manifest creation must use write-to-temporary-file, flush/fsync, and atomic replace, followed by a distributed barrier before training continues. Runtime events must include a monotonic sequence number and be flushed before a checkpoint that references them is considered complete.

Observed sharded-state signatures must either be embedded in each rank-local optimizer shard or gathered to rank 0. Checkpoint completion must fail if any expected rank signature or optimizer shard is missing.

### 11.1 `run_manifest.json` — immutable run identity

Write once at run start. Include:

#### Code identity

```text
NanoChat SHA
NanoChat dirty/clean status and diff hash/artifact when dirty
muon_bench version/SHA and dirty/clean status
package/archive SHA256 when applicable
```

Official benchmark runs should require clean code checkouts. Exploratory dirty runs must archive a patch and record its SHA-256; a dirty flag alone is not reproducible.

#### Environment

```text
Python version
PyTorch version
CUDA version
GPU names/compute capabilities
world size
compute dtype
FP8 state
Flash Attention mode
GNS exact identity when used
```

#### Model

```text
depth
d_model
n_head
n_kv_head
sequence length
architecture flags relevant to optimizer grouping
```

#### Optimizer experiment definition

```text
preset
resolved initial parameter-group definition
coefficient schedule
orthogonalizer
restart schedule
polar epsilon/dtype policy
MuonEq
Muon+
NorMuon/beta2
fractional roles/fractions
LR scaling
WD policy
momentum policy
initial grouping
grouping schedule
```

#### Training trajectory identity

```text
resolved iterations/token horizon
warmup
warmdown
final LR fraction
effective base learning rates
weight-decay schedule
batch/accumulation configuration
seed policy/value
```

#### Data identity

```text
tokenizer identity/hash
dataset/shard identity/version
```

### 11.2 `runtime_events.jsonl` — append-only event log

Examples:

```json
{"step": 3500, "event": "grouping_transition", "q_hpg": 16, "kv_hpg": 16}
```

Other useful events may include:

```text
resume event
trajectory-change acknowledgement
GNS fallback/dispatch anomaly
validation warning
```

Do not silently rewrite earlier events.

### 11.3 Checkpoint-local metadata

Each checkpoint should contain the minimal state needed to restore/audit that exact point:

- run-manifest SHA256;
- state-compatibility fingerprint/hash;
- trajectory fingerprint/hash;
- runtime grouping state;
- live parameter-group signature;
- rank-local observed state signature for every optimizer shard, or a complete gathered signature;
- transition stage/count;
- existing NanoChat dataloader and loop state;
- per-rank RNG states and active `GradScaler` state when applicable;
- resume mode/provenance if this run itself was resumed.

This avoids duplicating the full run manifest in every checkpoint while keeping each checkpoint auditable.

---

# Part IV — Performance claim discipline

## 12. Keep three classes of performance evidence separate

### A. Algorithmic optimization efficiency

Primary research metric:

\[
\boxed{\text{validation BPB versus tokens/steps}}
\]

Use this for the core ablation ladder.

### B. Within-harness systems evidence

Examples:

\[
\text{role-split full}\leftrightarrow\text{per-head}
\]

and

\[
f=1\leftrightarrow f=1/4.
\]

These are informative because much more of the implementation is shared.

### C. Production systems evidence

Metrics such as

\[
\text{tokens/s}
\]

and

\[
\text{wall-clock-to-target}
\]

should support production claims only after the winning method is fused/ported comparably to the native baseline.

Every result table should include an implementation-class field such as:

```text
native_fused
research_explicit
research_fused
```

A raw native-vs-explicit `optimizer_ms` difference must never be presented as an algorithm-only speed comparison.

---

# Part V — Research plan after qualification

## 13. Keep the attribution-first ladder

The review does not justify redesigning the scientific ladder.

Retain:

1. native NanoChat as the engineering baseline;
2. Keller reference as the scientific attribution baseline;
3. GNS backend separated from coefficient schedule;
4. role-split control before head-wise timing claims;
5. explicit GQA Q/K/V head counts;
6. MuonEq as a separate pre-polar ablation;
7. separate Frobenius snap / NorMuon / full Muon+ experiments;
8. `fractional_ef` as a neutral isolated mechanism;
9. dynamic coupling only after static geometry is understood;
10. MuonClip only if the architecture makes weight/head scaling meaningful.

### Phase dependencies

```text
Q-BASE passed
    -> Phase B: native + Keller baselines

Q-GNS passed
    -> Phase C: GNS + polynomial branch

static head/GQA smoke checks passed
    -> Phase D/E/F/G: head geometry, MuonEq, normalization, fractional MLP

Q-DYNAMIC passed
    -> Phase H: dynamic grouping

algorithm selected + Q-PRODUCTION passed
    -> production throughput claims
```

This is intentionally **not** one giant all-or-nothing gate.

---

## 14. Phase H should be framed as optimizer coupling scale

The dynamic experiment is better expressed as

\[
g_t=\text{number of heads coupled by one orthogonalization block}.
\]

For a two-stage schedule:

\[
g_t=
\begin{cases}
1, & t<t_s,\\
H, & t\ge t_s.
\end{cases}
\]

For a staged schedule:

\[
1\rightarrow2\rightarrow4\rightarrow H.
\]

The scientific question is therefore:

> Does the useful spatial scale of optimizer coupling increase over training?

The new runtime-state model and this research framing are aligned: `g_t` is both a scientific schedule and checkpointable optimizer state.

---

## 15. Phase G should remain `fractional_ef`, not exact Dion3

Continue using `fractional_ef` as the canonical isolated mechanism.

Do not claim exact Microsoft Dion3 unless a dedicated parity study matches, at minimum:

- coefficient/GNS backend;
- selected fraction;
- LR scaling;
- `mu` semantics;
- NorMuon beta2/epsilon;
- weight-decay semantics;
- parameter grouping;
- one-step captured-matrix updates;
- short training curves.

This is scientifically cleaner because fractional selection/error feedback can be evaluated independently of adaptive normalization.

---

## 16. FP8 policy

FP8 should not block the first proof that the optimizer is correct.

Use:

\[
\boxed{\text{bf16 integration qualification first}}
\]

and only then, if needed for the production target:

\[
\boxed{\text{FP8 transfer qualification}}.
\]

---

# Part VI — Concrete implementation plan

## 17. Files/components to modify

### `nanochat_muon_lab/setup.py`

Add helpers for:

- immutable initial grouping extraction;
- active grouping extraction from live groups;
- transactional active grouping restore;
- live parameter-group signature regeneration;
- expected state-schema generation.

Keep `set_attention_grouping()` as a low-level mutation helper only if a higher-level transition function updates runtime metadata atomically.

### New `nanochat_muon_lab/runtime_state.py`

Own:

- schema-versioned expected and observed optimizer-state descriptions;
- active grouping extraction, transactional restore, and post-load validation;
- transition-stage bookkeeping and duplicate-transition prevention;
- structural signatures that exclude legitimately scheduled mutable hyperparameters.

### New `nanochat_muon_lab/provenance.py`

Own:

- canonical JSON serialization and SHA-256 fingerprints;
- state-compatibility and resolved-trajectory fingerprints;
- immutable manifest creation;
- atomic rank-0 writes and append-only event records;
- rank-local signature aggregation/validation;
- schema-version checks.

### `install_into_nanochat.py`

Keep the generated patch thin: it should call package APIs rather than embedding provenance and state machinery as injected source text. Extend the generated trainer with:

- explicit seed policy/CLI if absent in the pinned NanoChat version;
- immutable run-manifest generation;
- trajectory fingerprint generation from **resolved** schedule values;
- state-compatibility fingerprint;
- checkpoint-local runtime grouping state;
- runtime-event logging;
- explicit resume modes;
- pre-load cross-checks and pre-/post-optimizer-load runtime-state validation;
- exact-resume checks that complement NanoChat's existing dataloader/loop-state resume.

### `nanochat_muon_lab/research_optimizer.py`

Add validation/debug instrumentation for:

- owned matrix indices;
- expected/observed state shapes;
- finite checks;
- collective/gather/copy-back assertions where practical;
- per-role execution counts;
- captured pre-orthogonalizer tensors for GNS validation.

Keep expensive checks off by default in normal benchmarks.

### New `nanochat_muon_lab/preflight.py`

Non-mutating environment and dependency checker.

### New integration tests/scripts

Suggested files:

```text
tests/integration/test_real_nanochat_single_gpu.py
tests/integration/test_distributed_muon.py
tests/integration/test_checkpoint_resume.py
tests/integration/test_dynamic_group_resume.py
tests/integration/test_seed_and_dataloader_resume.py
scripts/validate_gns_hardware.py
scripts/validate_distributed_optimizer.py
scripts/capture_orthogonalizer_inputs.py
```

Integration tests may skip automatically during ordinary local development when the required CUDA/multi-GPU/NanoChat/GNS environment is unavailable. Gate-specific qualification entry points must run in strict mode and fail when any required test is skipped or not collected.

### `pyproject.toml`

Do not make an arbitrary PyTorch wheel a mandatory dependency.

Document explicitly:

> `nanochat-muon-bench` intentionally inherits the pinned NanoChat PyTorch/CUDA environment. The test extra installs test tooling, not the CUDA framework stack.

---

## 18. Exact next-release test matrix

### CPU/unit

Retain the existing suite and add:

- runtime grouping serialization;
- active grouping restore;
- live signature regeneration;
- expected state-schema generation;
- state/trajectory fingerprint canonicalization;
- run-manifest hashing;
- runtime-event append behavior;
- resume-mode validation;
- duplicate-transition prevention;
- explicit seed-config serialization.

### Single GPU

At minimum:

```text
kj_reference
head-wise standard/reference solver configuration
production_candidate/fractional configuration
```

Then run `kj_gns` after Q-GNS dependency availability is established.

### Deferred distributed

At minimum:

```text
world_size=2, matrices divisible by ranks
world_size=2, matrices not divisible by ranks
world_size>number of matrices
world_size=8 H200 smoke
```

### Resume

Test:

```text
continuous vs exact resume, static optimizer
continuous vs exact resume, before grouping switch
continuous vs exact resume, after grouping switch
changed world size -> reject exact optimizer-state resume
changed optimizer definition -> reject
changed grouping schedule -> reject exact resume
changed trajectory -> reject exact resume by default
intentional trajectory-change -> explicit mode + provenance
weights-only restart under a new optimizer -> allowed as a new run
dataloader next-batch continuity across exact resume
per-rank RNG-position and active GradScaler continuity across exact resume
```

### GNS

Use the real pinned backend on target CUDA hardware and report results by role/shape, using both matched finite-polynomial and high-precision polar references.

---

# Part VII — Acceptance and stopping rules

## 19. Gate before Phase B baseline campaign

Require:

- [ ] exact NanoChat checkout/install qualification;
- [ ] preflight passes;
- [ ] explicit reproducibility/seed policy exists and is logged;
- [ ] native copied-trainer path reproduces unmodified native behavior within expected nondeterminism;
- [ ] single-GPU research optimizer step passes;
- [ ] exact static checkpoint/resume passes including dataloader, per-rank RNG-position, and active GradScaler continuity;
- [ ] short single-GPU non-GNS smoke run passes;
- [ ] immutable run provenance is generated automatically.

GNS, dynamic-switch, and distributed qualification are **not** prerequisites for Phase B.

---

## 20. Gate before Phase C GNS campaign

Require all relevant Phase-B prerequisites plus:

- [ ] exact GNS dependency identity is pinned and verified;
- [ ] real pre-orthogonalizer tensor suite exists;
- [ ] official backend matches the matched finite-polynomial reference within defined tolerance;
- [ ] small-matrix accuracy against high-precision polar reference is characterized;
- [ ] stress suite has no unexplained numerical failures;
- [ ] restart policy is validated for the chosen polynomial;
- [ ] runtime is measured by role/shape rather than only globally.

---

## 21. Gate before Phase H dynamic grouping

Require:

- [ ] initial config / schedule / runtime state are represented separately;
- [ ] checkpoint stores active grouping state and live signature;
- [ ] resume cross-checks explicit runtime grouping against saved optimizer groups and validates live grouping/signatures before and after optimizer-state load;
- [ ] post-switch exact resume matches continuous behavior within tolerance;
- [ ] inconsistent schedule is rejected;
- [ ] duplicate transition on resumed runs is impossible;
- [ ] GQA divisibility checks remain transactional through staged transitions.

---

## 22. Gate before production-speed claims

Require:

- [ ] winning research recipe has been fused/ported;
- [ ] implementation class is comparable to the production/native baseline;
- [ ] end-to-end tokens/s measured;
- [ ] wall-clock-to-fixed-BPB measured;
- [ ] communication overlap profiled;
- [ ] memory behavior profiled;
- [ ] final checkpoint/restart validated;
- [ ] result transfers beyond the cheap ranking model.

---

## 23. What should deliberately be deferred

Until the current validation boundary closes, do not add:

- another normalization family;
- another NS polynomial family;
- another momentum/controller mechanism;
- fractional Q/K/V updates;
- per-head O-projection Muon;
- more complex dynamic schedules than needed to test the current hypothesis;
- FP8-specific optimizer modifications unless immediately required.

The current search space is already large enough to answer the main research questions.

---

# Part VIII — Response to each coding-agent review finding

## Finding 1 — Dynamic head-switch metadata becomes stale

**Status:** accepted as a confirmed provenance and implicit-restore bug.\
**Action:** immutable initial definition + immutable grouping schedule + mutable checkpointed runtime grouping state; cross-check explicit state against saved optimizer groups, then validate live grouping/signatures both before and after state loading.\
**Gate:** `BLOCKING-DYNAMIC`.

## Finding 2 — Distributed and GPU paths lack executable coverage

**Status:** accepted.\
**Action:** real single-GPU step, exact resume, and a short single-GPU smoke test for the initial optimizer campaign. Two-rank collectives, padding/zero ownership, world-size parity, and 8×H200 throughput smoke are deferred until distributed scaling or production-throughput claims are in scope.\
**Gate:** `BLOCKING-BASE` for the single-GPU campaign; `DEFERRED-DISTRIBUTED` for distributed claims.

## Finding 3 — Native-vs-research timing is structurally confounded

**Status:** accepted as a claim limitation.\
**Action:** BPB-vs-tokens/steps for algorithmic claims; within-harness comparisons for provisional systems attribution; fused implementation for production speed claims.\
**Gate:** reporting discipline is immediate; fusion is `BLOCKING-PRODUCTION`.

## Finding 4 — Resume validation does not fingerprint the training schedule

**Status:** accepted.\
**Action:** state-compatibility + resolved training-trajectory fingerprints, explicit resume modes, and integration with existing NanoChat dataloader/loop-state, per-rank RNG-position, and active GradScaler resume.\
**Gate:** `BLOCKING-BASE` for exact-resume benchmark campaigns.

## Finding 5 — Official GNS is not pinned or numerically validated

**Status:** accepted.\
**Action:** exact dependency pin, H200 validation on real captured tensors, matched finite-polynomial reference, high-precision polar reference, stress/runtime characterization, restart validation.\
**Gate:** `BLOCKING-GNS`.

## Finding 6 — Test extra incomplete for standalone installation

**Status:** diagnosis accepted; remedy refined.\
**Action:** document inherited NanoChat PyTorch/CUDA environment and add non-mutating preflight checks; do not force-install an arbitrary torch wheel.\
**Gate:** `NONBLOCKING`, but preflight itself is part of `Q-BASE`.

## Finding 7 — Installer tests use synthetic fixtures

**Status:** accepted.\
**Action:** retain fixture rollback tests and add exact-SHA real-checkout install/compile/import qualification.\
**Gate:** `BLOCKING-BASE`.

## Finding 8 — Experiment provenance is not fully automatic

**Status:** accepted.\
**Action:** immutable `run_manifest.json`, append-only `runtime_events.jsonl`, and checkpoint-local state/runtime hashes.\
**Gate:** `BLOCKING-BASE` for the benchmark campaign.

---

# Part IX — Revised strategic position

## 24. What the project is trying to establish

The research question is not:

> Can we stack every recent Muon improvement into one optimizer?

It is:

> Which changes to the geometry, numerical polar solve, adaptive normalization, update frequency, and coupling scale materially improve NanoChat pretraining, and which of those gains survive a production-quality implementation?

The major scientific questions remain:

1. **Solver:** can GNS preserve the intended finite polar transformation efficiently on target hardware?
2. **Polynomial:** which coefficient schedule performs best under a fixed backend?
3. **Attention geometry:** is independent/grouped head geometry better than full-matrix coupling, and for which Q/K/V roles?
4. **Conditioning:** does MuonEq improve finite-step solver behavior enough to improve training?
5. **Post-polar adaptation:** should the final method use Frobenius correction, NorMuon, full Muon+, or a combination?
6. **Update frequency:** how much MLP orthogonalization can be removed with fractional error feedback without losing quality?
7. **Training phase:** should optimizer coupling scale increase during training?
8. **Production scaling:** once the algorithm is fixed, which LR-scaling and WD choices transfer best at scale?

The review does not invalidate these questions. It specifies the evidence standard required before their answers can be trusted.

---

## 25. Sequenced next engineering milestones

The next package work should add **no new optimizer feature**. Split it into three independently reviewable work packages.

### M1 — state and provenance core

Deliver:

1. `runtime_state.py` and `provenance.py`;
2. expected-state versus observed-state schemas;
3. active grouping serialization, transactional restore, and post-load validation;
4. canonical state-compatibility and resolved-trajectory fingerprints;
5. shared-initialization and rank-local seed policy representation;
6. resume-mode validation;
7. CPU tests for schema versions, hashing, transitions, and duplicate-transition prevention.

**Exit:** the pure state model is reviewable and fully covered without requiring NanoChat or CUDA execution.

### M2 — trainer integration and Q-BASE qualification

Deliver:

1. thin generated-trainer hooks into the state/provenance APIs;
2. immutable run manifest, append-only events, and unique run directories;
3. RNG/scaler/dataloader/loop checkpoint semantics;
4. exact-checkout preflight and install/compile/import qualification;
5. real single-GPU NanoChat smoke coverage;
6. static exact-resume coverage including RNG, GradScaler, and dataloader state;
7. a Q-BASE qualification report for the single-GPU research path;
8. a separate deferred distributed qualification plan.

**Exit:** Q-BASE passes for the pinned NanoChat environment. Phase B may begin.

### M3 — GNS qualification tooling and report

Deliver:

1. enforceable GNS dependency identity;
2. real pre-orthogonalizer tensor capture tooling;
3. matched finite-polynomial and high-precision polar references;
4. H200 stability, restart, dispatch, and runtime characterization;
5. a Q-GNS qualification report tied to exact code and environment identities.

**Exit:** Q-GNS passes for a specific schedule/backend. Phase C may begin.

Hardware qualification results are versioned artifacts tied to a code revision; they should not be hidden behind a package version bump or treated as passing merely because hardware-dependent tests skipped. Q-DYNAMIC remains a separate gate before Phase H, and Q-PRODUCTION remains separate from all research-harness milestones.

---

## 26. Final position

The package should continue to be treated as a **controlled research harness**, not a production-ready optimizer.

The correct response to the coding-agent review is not conceptual restart. The research decomposition is strong. The correct response is to make the implementation and experiment protocol much harder to misuse.

The project should now progress through:

\[
\boxed{
\text{reference correctness}
\rightarrow
\text{real-system correctness}
\rightarrow
\text{algorithmic evidence}
\rightarrow
\text{production evidence}
}
\]

The immediate goal is not to invent another optimizer component. It is to make the current research program trustworthy enough that both positive and negative experimental results can be believed.
