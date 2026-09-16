# Muon Engineering

Research tooling for benchmarking Muon-family optimizer variants in NanoChat.

## Repository Layout

- [`muon_bench/`](muon_bench/) contains the correctness-first benchmark package, experiment ladder, implementation notes, tests, and independent review.
- [`nanochat/`](nanochat/) is a Git submodule pinned to the NanoChat revision used for integration work.

## Clone

```bash
git clone --recurse-submodules https://github.com/honghaoyu12/muon_engineering.git
cd muon_engineering
```

For an existing clone:

```bash
git submodule update --init --recursive
```

## Install the Benchmark Harness

Run the installer from the NanoChat checkout root:

```bash
cd nanochat
python ../muon_bench/install_into_nanochat.py
```

This creates `nanochat_muon_lab/` and `scripts/base_train_muon_lab.py` inside the checkout without modifying NanoChat's native `scripts/base_train.py`.

Read [`muon_bench/README.md`](muon_bench/README.md) for presets and usage, [`muon_bench/EXPERIMENT_LADDER.md`](muon_bench/EXPERIMENT_LADDER.md) for the benchmark sequence, and [`muon_bench/REVIEW.md`](muon_bench/REVIEW.md) for the current engineering review.

## Validation Status

The package includes CPU correctness tests, but GPU, official-GNS, distributed, and checkpoint/resume gates must pass on the target environment before benchmark results are treated as production evidence.
