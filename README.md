# radtrain

**Radiation-resilient neural network training for space-based GPUs.**

The first systematic study of how space radiation (bit flips from Single Event Upsets) affects transformer pretraining, and software-only defenses that make training survivable in orbit.

## Key Finding

A single technique — **Fault-Aware Training (FAT)** — makes transformer training radiation-invariant across 4 orders of magnitude of radiation intensity, at a fixed 3.2% quality cost. No hardware modifications required.

```
                No Defense    FAT (1 flip/step)
Clean            0.980         1.011 (+3.2%)
10x LEO          CRASH         1.011
100x LEO         CRASH         1.012
1,000x LEO       CRASH         1.012
10,000x LEO      —             1.012
```

## What's in this repo

```
paper.md              — Full paper with 45 experiments and analysis
train.py              — Modified autoresearch training script with radiation simulation
inject.py             — Bit-flip injection framework (portable, works with any PyTorch model)
saved_models/         — 3 reference checkpoints (clean, FAT-trained, full space sim)
results.tsv           — Complete experimental results (45 experiments)
```

## The Problem

Space radiation causes random bit flips in GPU memory at measurable rates:
- **LEO (Low Earth Orbit)**: ~4.76x10^-7 errors/bit/day ([Wang et al. 2024](https://arxiv.org/abs/2407.11853))
- **94M param model in bf16**: ~1,200 bit flips per day
- **A single exponent bit flip** can crash training instantly

Prior work studied radiation effects on **inference** (RedNet, FAT for inference, RoR). Nobody studied **training** — where errors compound across millions of gradient steps.

## The Solution: Fault-Aware Training (FAT)

Inject 1 random bit flip per training step. The optimizer learns to find flat minima that are structurally resilient to weight corruption.

**Why it works**: Sparse, heavy-tailed perturbations (bit flips) push the optimizer into regions of the loss landscape where individual weight perturbations have minimal impact. This is distinct from SAM/dropout/Gaussian noise — the perturbation profile must be sparse and occasionally extreme to match the physics of radiation.

**Combined with NaN repair** (immediately zeroing NaN/Inf values after bit flips), FAT provides complete protection.

## Results Summary (45 experiments)

| Finding | Result |
|---------|--------|
| Training fragility | ~6 bit flips crash a 94M model |
| FAT radiation invariance | 1x-10,000x LEO, only +3.2% quality cost |
| FAT eliminates variance | Without FAT: val_bpb 1.009-1.102. With FAT: 1.009-1.011 |
| Only exponent bits matter | Mantissa (44%) and sign (6%) flips cause zero degradation |
| Muon = AdamW resilience | Optimizer choice doesn't affect radiation tolerance |
| Model size doesn't matter | Degradation scales linearly with flip-to-param ratio |
| Softcap is a defense | Logit bounding prevents radiation-induced overflow |
| Eclipse cycling is cheap | Power cycle every 200 steps costs only +0.002-0.004 |
| Full space sim | FAT + eclipse + progressive TID = 3.7% total cost |
| Noise type matters | Sparse bit flips work; dense Gaussian noise doesn't |

## Reproducing

Requires a single NVIDIA GPU (tested on H100 80GB). Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

```bash
# Install
uv sync

# Prepare data (one-time)
uv run prepare.py

# Clean baseline
SAVE_NAME=clean.pt uv run train.py

# FAT training (edit train.py: FAT_TRAINING_FLIPS=1)
SAVE_NAME=fat.pt uv run train.py

# Full space simulation (edit train.py: FAT + RADIATION_MULTIPLIER=100 + ECLIPSE_CYCLE_STEPS=200)
SAVE_NAME=space.pt uv run train.py
```

## Saved Models

| Model | val_bpb | Conditions | Size |
|-------|---------|------------|------|
| `clean_baseline.pt` | 0.9799 | No radiation, no FAT | 301 MB |
| `fat_trained.pt` | 1.0117 | FAT 1 flip/step | 301 MB |
| `full_space_sim.pt` | 1.0163 | FAT + 100x TID + 5 eclipses | 301 MB |

## Citation

```bibtex
@misc{radtrain2026,
  title={Fault-Aware Pretraining: Training Language Models Under Simulated Space Radiation},
  year={2026},
  url={https://github.com/tylergibbs/radtrain}
}
```

## References

- [RedNet: Space Radiation Tolerance](https://arxiv.org/abs/2407.11853) — Inference-focused radiation defense
- [Google TPU Constellation](https://arxiv.org/abs/2511.19468) — Space-based AI infrastructure proposal
- [Silent Data Corruption in LLM Training](https://arxiv.org/abs/2502.12340) — Terrestrial SDC effects
- [Fault-Aware Training for Inference](https://arxiv.org/abs/2502.09374) — FAT for inference (not training)
- [Simsekli et al. 2020](https://arxiv.org/abs/1901.06053) — Heavy-tailed noise and flat minima theory

## License

MIT
