"""
radtrain/inject.py — Portable bit-flip injection for radiation simulation.

Works with any PyTorch model. Simulates Single Event Upsets (SEUs) in bf16/fp16/fp32 weights
at rates calibrated to measured Low Earth Orbit radiation environments.

Usage:
    from inject import inject_bitflips, compute_flip_rate

    # Compute expected flip rate for your model
    rate = compute_flip_rate(model, multiplier=10)  # 10x LEO rate
    print(f"Expected flips per step: {rate['flips_per_step']:.4f}")

    # Inject flips during training
    for step in range(num_steps):
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Radiation simulation
        if random.random() < rate['flip_prob_per_step']:
            stats = inject_bitflips(model, num_flips=1)

Reference: Wang et al. 2024, "A Case for Application-Aware Space Radiation Tolerance
in Orbital Computing", arXiv:2407.11853. Measured LEO SEU rate: 4.76e-7 errors/bit/day
on Chaohu-1 satellite (NVIDIA Jetson Xavier NX).
"""

import torch
from typing import Optional


# Measured LEO Single Event Upset rate (Chaohu-1 satellite, Wang et al. 2024)
LEO_SEU_RATE_PER_BIT_PER_DAY = 4.76e-7


def compute_flip_rate(
    model: torch.nn.Module,
    multiplier: float = 1.0,
    training_seconds: float = 300.0,
    estimated_steps: int = 1100,
) -> dict:
    """Compute bit-flip rate for a model at a given radiation multiplier.

    Args:
        model: PyTorch model
        multiplier: Radiation rate multiplier (1.0 = realistic LEO)
        training_seconds: Total training time in seconds
        estimated_steps: Estimated number of training steps

    Returns:
        dict with total_bits, flips_per_day, flips_in_run, flip_prob_per_step
    """
    total_bits = sum(
        p.numel() * (16 if p.dtype in (torch.bfloat16, torch.float16) else 32)
        for p in model.parameters()
    )
    flips_per_day = LEO_SEU_RATE_PER_BIT_PER_DAY * total_bits * multiplier
    flips_in_run = flips_per_day * (training_seconds / 86400)
    flip_prob_per_step = flips_in_run / estimated_steps

    return {
        "total_bits": total_bits,
        "flips_per_day": flips_per_day,
        "flips_in_run": flips_in_run,
        "flip_prob_per_step": flip_prob_per_step,
        "multiplier": multiplier,
    }


def inject_bitflips(
    model: torch.nn.Module,
    num_flips: int = 1,
    mode: str = "all",
    repair_nan: bool = True,
) -> dict:
    """Inject random bit flips into model parameters.

    Simulates Single Event Upsets (SEUs) in weight memory by flipping
    random bits in the binary representation of parameters.

    Args:
        model: PyTorch model (parameters should be on GPU)
        num_flips: Number of bit flips to inject
        mode: "all" (uniform random), "exponent_only" (bits 7-14 of bf16),
              "mantissa_only" (bits 0-6), "sign_only" (bit 15)
        repair_nan: If True, immediately zero out NaN/Inf values after flipping

    Returns:
        dict with flip statistics
    """
    stats = {"total": 0, "nan_repaired": 0, "exponent": 0, "mantissa": 0, "sign": 0}

    # Collect parameters and compute total bits
    params = [(name, p) for name, p in model.named_parameters()]
    total_bits = sum(p.numel() * _bits_per_element(p) for _, p in params)

    for _ in range(num_flips):
        # Pick a random bit position across all parameters
        bit_idx = torch.randint(0, total_bits, (1,)).item()

        # Find which parameter and which element
        cumulative = 0
        for name, p in params:
            bpe = _bits_per_element(p)
            param_bits = p.numel() * bpe
            if cumulative + param_bits > bit_idx:
                local_bit = bit_idx - cumulative
                elem_idx = local_bit // bpe
                bit_pos = local_bit % bpe

                # Apply mode restriction
                if mode == "exponent_only":
                    bit_pos = 7 + (bit_pos % 8) if bpe == 16 else 23 + (bit_pos % 8)
                elif mode == "mantissa_only":
                    bit_pos = bit_pos % (7 if bpe == 16 else 23)
                elif mode == "sign_only":
                    bit_pos = bpe - 1

                # Flip the bit
                with torch.no_grad():
                    flat = p.data.view(-1)
                    if bpe == 16:
                        flat_i16 = flat.view(torch.int16)
                        old_bits = flat_i16[elem_idx].to(torch.int32)
                        new_bits = old_bits ^ (1 << bit_pos)
                        flat_i16[elem_idx] = new_bits.to(torch.int16)
                    else:  # fp32
                        flat_i32 = flat.view(torch.int32)
                        flat_i32[elem_idx] = flat_i32[elem_idx] ^ (1 << bit_pos)

                    # Track statistics
                    stats["total"] += 1
                    if bpe == 16:
                        if 7 <= bit_pos <= 14:
                            stats["exponent"] += 1
                        elif bit_pos == 15:
                            stats["sign"] += 1
                        else:
                            stats["mantissa"] += 1
                    else:
                        if 23 <= bit_pos <= 30:
                            stats["exponent"] += 1
                        elif bit_pos == 31:
                            stats["sign"] += 1
                        else:
                            stats["mantissa"] += 1

                    # NaN repair
                    if repair_nan and not torch.isfinite(flat[elem_idx]):
                        flat[elem_idx] = 0.0
                        stats["nan_repaired"] += 1

                break
            cumulative += param_bits

    return stats


def weight_clamp(model: torch.nn.Module, max_val: float = 1.0):
    """Clamp all bf16 weights to [-max_val, max_val].

    Prevents accumulation of extreme values from exponent bit flips.
    """
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype in (torch.bfloat16, torch.float16):
                p.data.clamp_(-max_val, max_val)


def repair_all_nan(model: torch.nn.Module) -> int:
    """Zero out all NaN/Inf values in model parameters. Returns count repaired."""
    count = 0
    with torch.no_grad():
        for p in model.parameters():
            bad = ~torch.isfinite(p.data)
            if bad.any():
                n = int(bad.sum().item())
                p.data[bad] = 0.0
                count += n
    return count


def _bits_per_element(p: torch.Tensor) -> int:
    if p.dtype in (torch.bfloat16, torch.float16):
        return 16
    elif p.dtype == torch.float32:
        return 32
    elif p.dtype == torch.float64:
        return 64
    else:
        return 32  # fallback
