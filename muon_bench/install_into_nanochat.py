"""Install the benchmark harness into a clean, pinned NanoChat checkout.

Usage from the NanoChat repository root:
    python /path/to/nanochat_muon_bench/install_into_nanochat.py

The installer is transactional with respect to its own outputs: it validates every source patch
anchor and constructs the complete patched trainer in memory *before* replacing/copying anything.
Native ``scripts/base_train.py`` is never modified.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()

if not (ROOT / "nanochat" / "optim.py").exists() or not (ROOT / "scripts" / "base_train.py").exists():
    raise SystemExit("Run this script from the root of a NanoChat checkout.")

native = ROOT / "scripts" / "base_train.py"
text = native.read_text()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    n = source.count(old)
    if n != 1:
        raise RuntimeError(f"Patch point {label!r}: expected exactly one match, found {n}. Upstream changed.")
    return source.replace(old, new, 1)


# 1) CLI. `native` remains an untouched current-NanoChat comparison inside the copied trainer.
anchor = 'parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")\n'
text = replace_once(text, anchor, anchor + '''parser.add_argument("--muon-lab-preset", type=str, default="native",
    choices=["native", "kj_reference", "kj_moonlight_scale", "kj_gns", "dion_gns", "nanochat_pe_gns", "role_split_dion_gns", "head_dion_gns", "head_eq_dion_gns", "head_eq_norm_dion_gns", "production_candidate"],
    help="Muon benchmark preset; native uses unmodified NanoChat optimizer")
parser.add_argument("--muon-lab-ortho-fraction", type=float, default=0.25,
    help="fraction selected along the smaller matrix dimension for the Dion3-paper fractional EF rung")
parser.add_argument("--muon-lab-momentum-policy", type=str, default="constant", choices=["constant", "nanochat"],
    help="research presets: constant=current Keller reference beta=0.95; nanochat=use base_train momentum schedule")
parser.add_argument("--muon-lab-head-switch-frac", type=float, default=-1.0,
    help="if >=0, switch configured head-wise Q/K/V groups to full-matrix at this training fraction")
parser.add_argument("--muon-lab-override-json", type=str, default="{}",
    help="JSON dict overriding research-preset fields; unknown keys fail loudly")
''', "CLI")

# Explicit resume semantics are part of the provenance contract. Keep this optional for the
# small installer fixture used by package tests; the real NanoChat source contains the anchor.
resume_anchor = 'parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")\n'
if resume_anchor in text:
    text = replace_once(text, resume_anchor, resume_anchor + '''parser.add_argument("--muon-lab-resume-mode", "--resume-mode", dest="muon_lab_resume_mode",
    type=str, default="exact", choices=["exact", "trajectory-change", "weights-only"],
    help="resume contract: exact trajectory, changed trajectory, or model weights only")
''', "resume mode")

imports_anchor = 'from scripts.base_eval import evaluate_core\n'
if imports_anchor in text:
    text = replace_once(text, imports_anchor, imports_anchor + '''
from pathlib import Path
import hashlib
import platform
import subprocess
''', "M2 imports")

# Load only model weights for an explicit weights-only resume.
load_anchor = '    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)\n'
if load_anchor in text:
    text = replace_once(text, load_anchor, '''    load_optimizer_for_resume = args.muon_lab_resume_mode != "weights-only"
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device,
        load_optimizer=load_optimizer_for_resume, rank=ddp_rank
    )
''', "resume load mode")

# 2) Optimizer initialization.
old_init = '''optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)'''
new_init = '''if args.muon_lab_preset == "native":
    optimizer = model.setup_optimizer(
        # AdamW hyperparameters
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        scalar_lr=args.scalar_lr * batch_lr_scale,
        # Muon hyperparameters
        matrix_lr=args.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    )
else:
    from nanochat_muon_lab.setup import setup_research_optimizer
    optimizer = setup_research_optimizer(
        model,
        preset=args.muon_lab_preset,
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        scalar_lr=args.scalar_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
        ortho_fraction=args.muon_lab_ortho_fraction,
        override=__import__("json").loads(args.muon_lab_override_json),
    )
# Record distributed ownership topology and a canonical *effective* research config for safe resume.
user_config["muon_lab_world_size"] = ddp_world_size
if args.muon_lab_preset != "native":
    _effective_lab_config = {
        "optimizer": optimizer.muon_lab_config,
        "param_group_signature": optimizer.muon_lab_param_group_signature,
        "momentum_policy": args.muon_lab_momentum_policy,
        "head_switch_frac": args.muon_lab_head_switch_frac,
    }
    user_config["muon_lab_effective_config_json"] = __import__("json").dumps(
        _effective_lab_config, sort_keys=True, separators=(",", ":")
    )'''
text = replace_once(text, old_init, new_init, "optimizer initialization")

m2_runtime_setup = '\nfrom nanochat_muon_lab.runtime_state import (\n    apply_active_grouping_state_,\n    build_expected_state_schema,\n    build_live_group_signature,\n    build_observed_state_signature,\n    get_active_grouping_state,\n    validate_live_runtime_state,\n    validate_saved_grouping_state,\n)\nfrom nanochat_muon_lab.provenance import (\n    aggregate_rank_local_signatures,\n    append_runtime_event,\n    build_run_manifest,\n    build_seed_policy,\n    fingerprint_payload,\n    read_runtime_events,\n    state_compatibility_fingerprint,\n    trajectory_fingerprint,\n    verify_fingerprint,\n    write_immutable_manifest,\n)\n\n_runtime_grouping_state = get_active_grouping_state(optimizer)\n_initial_grouping = getattr(optimizer, "muon_lab_initial_grouping", {\n    "schema_version": 1,\n    "artifact_type": "initial_grouping",\n    "q_heads_per_group": _runtime_grouping_state.get("q_heads_per_group"),\n    "k_heads_per_group": _runtime_grouping_state.get("k_heads_per_group"),\n    "v_heads_per_group": _runtime_grouping_state.get("v_heads_per_group"),\n})\n_expected_state_schema = build_expected_state_schema(optimizer, world_size=ddp_world_size)\n_saved_provenance = meta_data.get("muon_lab_provenance", {}) if resuming else {}\n_saved_runtime = _saved_provenance.get("runtime_grouping_state")\n_saved_live_signature = _saved_provenance.get("live_group_signature")\nif resuming and args.muon_lab_resume_mode != "weights-only" and _saved_runtime is not None:\n    apply_active_grouping_state_(optimizer, _saved_runtime)\n    _runtime_grouping_state = get_active_grouping_state(optimizer)\n    if _saved_live_signature is not None:\n        validate_live_runtime_state(\n            optimizer,\n            expected_grouping_state=_saved_runtime,\n            expected_group_signature=_saved_live_signature,\n        )\n    if optimizer_data is None:\n        raise RuntimeError("Resume metadata requires optimizer state, but no optimizer shard was loaded")\n    validate_saved_grouping_state(_saved_runtime, optimizer_data)\n_live_group_signature = build_live_group_signature(optimizer)\n'
text = replace_once(text, new_init, new_init + m2_runtime_setup, "M2 runtime restore")

# 3) Resume guard. Optimizer state is group/order/world-size dependent; don't silently load it into
# a different research configuration. We compare the arguments that alter state shape/semantics.
old_resume = '''if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data'''
new_resume = '''if resuming:
    previous_cfg = meta_data.get("user_config", {})
    previous_preset = previous_cfg.get("muon_lab_preset", "native")
    if previous_preset != args.muon_lab_preset:
        raise RuntimeError(
            f"Refusing optimizer resume across Muon-lab presets: checkpoint={previous_preset!r}, current={args.muon_lab_preset!r}"
        )
    if args.muon_lab_preset != "native":
        # Preferred guard for checkpoints created by this audited package: compare the canonical
        # effective config, not raw JSON spelling/key order. Fall back to legacy CLI guards only
        # for older Muon-lab checkpoints that predate this metadata field.
        previous_effective = previous_cfg.get("muon_lab_effective_config_json")
        current_effective = user_config.get("muon_lab_effective_config_json")
        if previous_effective is not None:
            if previous_effective != current_effective:
                raise RuntimeError(
                    "Refusing optimizer resume with a changed effective Muon-lab configuration: "
                    f"checkpoint={previous_effective}, current={current_effective}"
                )
        else:
            guarded = {
                "muon_lab_ortho_fraction": args.muon_lab_ortho_fraction,
                "muon_lab_momentum_policy": args.muon_lab_momentum_policy,
                "muon_lab_head_switch_frac": args.muon_lab_head_switch_frac,
            }
            for key, current_value in guarded.items():
                if key in previous_cfg and previous_cfg[key] != current_value:
                    raise RuntimeError(
                        f"Refusing optimizer resume with changed {key}: checkpoint={previous_cfg[key]!r}, current={current_value!r}"
                    )
            if "muon_lab_override_json" in previous_cfg:
                old_override = __import__("json").loads(previous_cfg["muon_lab_override_json"])
                new_override = __import__("json").loads(args.muon_lab_override_json)
                if old_override != new_override:
                    raise RuntimeError(
                        f"Refusing optimizer resume with changed muon_lab_override_json: checkpoint={old_override!r}, current={new_override!r}"
                    )
    previous_world = previous_cfg.get("muon_lab_world_size")
    if previous_world is not None and int(previous_world) != int(ddp_world_size):
        raise RuntimeError(
            f"Refusing optimizer resume across world sizes: checkpoint={previous_world}, current={ddp_world_size}. "
            "Muon optimizer state is sharded by matrix ownership."
        )
    if args.muon_lab_resume_mode != "weights-only":
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data'''
text = replace_once(text, old_resume, new_resume, "optimizer resume")

# 4) Training-loop momentum/WD policy and optional geometry phase switch.
old_sched = '''        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay'''
new_sched = '''        if group['kind'] == 'muon':
            # Research rungs default to constant beta=0.95 unless the native schedule is requested.
            if args.muon_lab_preset == "native" or args.muon_lab_momentum_policy == "nanochat":
                group["momentum"] = muon_momentum
            else:
                group["momentum"] = 0.95
            group["weight_decay"] = muon_weight_decay

    # Optional per-head/grouped -> full-matrix phase experiment. State is row-aligned and retained.
    if args.muon_lab_preset != "native" and args.muon_lab_head_switch_frac >= 0:
        if step / num_iterations >= args.muon_lab_head_switch_frac:
            from nanochat_muon_lab.runtime_state import transition_attention_grouping_
            q_heads = model_config.n_head
            kv_heads = model_config.n_kv_head
            if get_active_grouping_state(optimizer)["active_stage"] == 0:
                transition_attention_grouping_(
                    optimizer, heads_per_group_q=q_heads, heads_per_group_kv=kv_heads,
                    transition_step=step, target_stage=1
                )
                if master_process:
                    _events = read_runtime_events(_muon_lab_events_path)
                    append_runtime_event(_muon_lab_events_path, {
                        "sequence_number": (_events[-1]["sequence_number"] + 1 if _events else 0),
                        "event": "grouping_transition", "step": step,
                        "q_heads_per_group": q_heads, "kv_heads_per_group": kv_heads,
                    })'''
text = replace_once(text, old_sched, new_sched, "training-loop Muon schedule")

provenance_block = '\n# M2 provenance/checkpoint integration. This runs after the resolved training horizon exists.\nfrom pathlib import Path\nimport platform\n\ndef _muon_lab_gpu_identity():\n    if not torch.cuda.is_available():\n        return []\n    return [{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]\n\n_muon_lab_trajectory = {\n    "num_iterations": num_iterations,\n    "total_tokens": total_tokens,\n    "warmup_steps": args.warmup_steps,\n    "warmdown_ratio": args.warmdown_ratio,\n    "final_lr_frac": args.final_lr_frac,\n    "total_batch_size": total_batch_size,\n    "device_batch_size": args.device_batch_size,\n    "max_seq_len": args.max_seq_len,\n    "world_size": ddp_world_size,\n    "momentum_policy": args.muon_lab_momentum_policy,\n    "head_switch_frac": args.muon_lab_head_switch_frac,\n}\n_muon_lab_trajectory_fp = trajectory_fingerprint(_muon_lab_trajectory)\n_muon_lab_state_fp = state_compatibility_fingerprint({\n    "model_config": model_config_kwargs,\n    "optimizer_config": getattr(optimizer, "muon_lab_config", {"preset": "native"}),\n    "group_signature": _live_group_signature,\n    "expected_state_schema": _expected_state_schema,\n    "world_size": ddp_world_size,\n})\n_muon_lab_manifest = build_run_manifest(\n    output_dirname,\n    code={"trainer": "base_train_muon_lab"},\n    environment={"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "world_size": ddp_world_size, "compute_dtype": str(COMPUTE_DTYPE).removeprefix("torch."), "gpu": _muon_lab_gpu_identity()},\n    model=model_config_kwargs,\n    optimizer={"preset": args.muon_lab_preset, "config": getattr(optimizer, "muon_lab_config", {"preset": "native"}), "initial_grouping": _initial_grouping},\n    trajectory=_muon_lab_trajectory,\n    data={"tokenizer_vocab_size": vocab_size},\n)\n_muon_lab_manifest_path = Path(checkpoint_dir) / "run_manifest.json"\n_muon_lab_events_path = Path(checkpoint_dir) / "runtime_events.jsonl"\nif _muon_lab_manifest_path.exists():\n    _published = json.loads(_muon_lab_manifest_path.read_text(encoding="utf-8"))\n    if not resuming:\n        raise RuntimeError(f"Refusing to reuse existing run manifest: {_muon_lab_manifest_path}")\n    if args.muon_lab_resume_mode == "exact" and fingerprint_payload(_published) != fingerprint_payload(_muon_lab_manifest):\n        raise RuntimeError("Exact resume requires an identical run manifest")\nelse:\n    write_immutable_manifest(_muon_lab_manifest_path, _muon_lab_manifest, rank=ddp_rank, barrier=(dist.barrier if ddp else None))\nif master_process:\n    _events = read_runtime_events(_muon_lab_events_path)\n    append_runtime_event(_muon_lab_events_path, {"sequence_number": (_events[-1]["sequence_number"] + 1 if _events else 0), "event": "resume" if resuming else "run_start", "step": args.resume_from_step if resuming else 0, "mode": args.muon_lab_resume_mode})\nif resuming and meta_data.get("muon_lab_provenance"):\n    _saved = meta_data["muon_lab_provenance"]\n    verify_fingerprint(_saved["state_compatibility"], expected_type="state_compatibility")\n    verify_fingerprint(_saved["trajectory"], expected_type="training_trajectory")\n    if args.muon_lab_resume_mode != "weights-only" and _saved["state_compatibility"]["sha256"] != _muon_lab_state_fp["sha256"]:\n        raise RuntimeError("Optimizer/model state compatibility fingerprint mismatch on resume")\n    if args.muon_lab_resume_mode == "exact" and _saved["trajectory"]["sha256"] != _muon_lab_trajectory_fp["sha256"]:\n        raise RuntimeError("Exact resume requires an identical training trajectory")\n'
_provenance_anchor = 'print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")\n'
if _provenance_anchor in text:
    text = replace_once(text, _provenance_anchor, _provenance_anchor + provenance_block, "M2 provenance")
_checkpoint_meta_anchor = '                "loop_state": { # all loop state (other than step) so that we can resume training\n'
if _checkpoint_meta_anchor in text:
    text = replace_once(text, _checkpoint_meta_anchor, '                "muon_lab_provenance": {\n                    "manifest_sha256": fingerprint_payload(_muon_lab_manifest),\n                    "state_compatibility": _muon_lab_state_fp,\n                    "trajectory": _muon_lab_trajectory_fp,\n                    "runtime_grouping_state": get_active_grouping_state(optimizer),\n                    "live_group_signature": build_live_group_signature(optimizer),\n                    "expected_state_schema": _expected_state_schema,\n                    "observed_state_signature": build_observed_state_signature(optimizer, rank=ddp_rank, world_size=ddp_world_size),\n                },\n' + _checkpoint_meta_anchor, "M2 checkpoint metadata")

# Only after all patch validation succeeds do we stage and commit our destination outputs.
# Existing Muon-lab outputs are backed up and restored if either replacement fails.
src_pkg = HERE / "nanochat_muon_lab"
dst_pkg = ROOT / "nanochat_muon_lab"
lab = ROOT / "scripts" / "base_train_muon_lab.py"
stage_root = Path(tempfile.mkdtemp(prefix=".muon_lab_stage_", dir=ROOT))
stage_pkg = stage_root / "nanochat_muon_lab"
stage_lab = stage_root / "base_train_muon_lab.py"
backup_pkg = stage_root / "previous_nanochat_muon_lab"
backup_lab = stage_root / "previous_base_train_muon_lab.py"
shutil.copytree(src_pkg, stage_pkg)
stage_lab.write_text(text)

try:
    if dst_pkg.exists():
        dst_pkg.rename(backup_pkg)
    if lab.exists():
        lab.rename(backup_lab)
    try:
        stage_pkg.rename(dst_pkg)
        stage_lab.rename(lab)
    except Exception:
        # Remove any partially committed new outputs before restoring previous outputs.
        if dst_pkg.exists():
            shutil.rmtree(dst_pkg)
        if lab.exists():
            lab.unlink()
        if backup_pkg.exists():
            backup_pkg.rename(dst_pkg)
        if backup_lab.exists():
            backup_lab.rename(lab)
        raise
finally:
    shutil.rmtree(stage_root, ignore_errors=True)

print(f"Installed package: {dst_pkg}")
print(f"Created benchmark trainer: {lab}")
print("Native scripts/base_train.py was not modified.")
