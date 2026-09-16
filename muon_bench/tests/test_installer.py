from pathlib import Path
import subprocess
import sys


def test_installer_is_transactional_and_injects_resume_guard(tmp_path):
    # Minimal source fixture containing only the four audited NanoChat patch anchors. This tests our
    # patch mechanics; the real current-source compatibility gate still runs in the user's checkout.
    (tmp_path / "nanochat").mkdir()
    (tmp_path / "nanochat" / "optim.py").write_text("# fixture\n")
    (tmp_path / "scripts").mkdir()
    base = '''import argparse\nimport json\nparser = argparse.ArgumentParser()\nparser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")\noptimizer = model.setup_optimizer(\n    # AdamW hyperparameters\n    unembedding_lr=args.unembedding_lr * batch_lr_scale,\n    embedding_lr=args.embedding_lr * batch_lr_scale,\n    scalar_lr=args.scalar_lr * batch_lr_scale,\n    # Muon hyperparameters\n    matrix_lr=args.matrix_lr * batch_lr_scale,\n    weight_decay=weight_decay_scaled,\n)\nif resuming:\n    optimizer.load_state_dict(optimizer_data)\n    del optimizer_data\nfor group in optimizer.param_groups:\n        if group['kind'] == 'muon':\n            group["momentum"] = muon_momentum\n            group["weight_decay"] = muon_weight_decay\n'''
    (tmp_path / "scripts" / "base_train.py").write_text(base)
    installer = Path(__file__).parents[1] / "install_into_nanochat.py"
    subprocess.run([sys.executable, str(installer)], cwd=tmp_path, check=True, capture_output=True, text=True)
    generated = (tmp_path / "scripts" / "base_train_muon_lab.py").read_text()
    assert "role_split_dion_gns" in generated
    assert "muon_lab_ortho_fraction" in generated
    assert "Refusing optimizer resume across Muon-lab presets" in generated
    assert "muon_lab_effective_config_json" in generated
    assert "muon_lab_param_group_signature" in generated
    assert "sort_keys=True" in generated
    assert (tmp_path / "scripts" / "base_train.py").read_text() == base
    assert (tmp_path / "nanochat_muon_lab" / "research_optimizer.py").exists()


def test_installer_does_not_copy_package_when_patch_validation_fails(tmp_path):
    (tmp_path / "nanochat").mkdir()
    (tmp_path / "nanochat" / "optim.py").write_text("# fixture\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "base_train.py").write_text("# changed upstream; anchors absent\n")
    installer = Path(__file__).parents[1] / "install_into_nanochat.py"
    result = subprocess.run([sys.executable, str(installer)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "nanochat_muon_lab").exists()
    assert not (tmp_path / "scripts" / "base_train_muon_lab.py").exists()


def test_failed_reinstall_preserves_existing_lab_outputs(tmp_path):
    (tmp_path / "nanochat").mkdir()
    (tmp_path / "nanochat" / "optim.py").write_text("# fixture\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "base_train.py").write_text("# changed upstream; anchors absent\n")
    existing_pkg = tmp_path / "nanochat_muon_lab"
    existing_pkg.mkdir()
    (existing_pkg / "marker.txt").write_text("old package")
    existing_lab = tmp_path / "scripts" / "base_train_muon_lab.py"
    existing_lab.write_text("old trainer")

    installer = Path(__file__).parents[1] / "install_into_nanochat.py"
    result = subprocess.run([sys.executable, str(installer)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert (existing_pkg / "marker.txt").read_text() == "old package"
    assert existing_lab.read_text() == "old trainer"
