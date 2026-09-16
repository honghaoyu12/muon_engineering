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
            from nanochat_muon_lab.setup import set_attention_grouping
            q_heads = model_config.n_head
            kv_heads = model_config.n_kv_head
            set_attention_grouping(
                optimizer, heads_per_group_q=q_heads, heads_per_group_kv=kv_heads
            )'''
text = replace_once(text, old_sched, new_sched, "training-loop Muon schedule")

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
