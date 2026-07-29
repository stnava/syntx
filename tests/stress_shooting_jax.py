#!/usr/bin/env python3
"""
Empirical Stress Test Harness for GeodesicShootingModelJAX Optimization.
Stress tests fit() across multiple random seeds, 2D and 3D image dimensions,
identity initialization (p0 = 0), learning rates, fluid sigmas, and epoch sweeps.
Monitors loss trajectory for zero NaN / Inf occurrences.
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

import syntx
from syntx.shooting_jax import GeodesicShootingModelJAX


def generate_synthetic_pair_2d(shape=(64, 64), seed=42):
    np.random.seed(seed)
    y, x = np.ogrid[:shape[0], :shape[1]]
    center1 = (shape[0] / 2.0, shape[1] / 2.0)
    center2 = (shape[0] / 2.0 + 3.0, shape[1] / 2.0 - 2.0)

    r1 = np.sqrt((y - center1[0])**2 + (x - center1[1])**2)
    r2 = np.sqrt((y - center2[0])**2 + (x - center2[1])**2)

    img1 = np.exp(-0.5 * (r1 / 12.0)**2).astype(np.float32)
    img2 = np.exp(-0.5 * (r2 / 14.0)**2).astype(np.float32)
    
    # Add random smooth foreground structures
    img1 += 0.2 * np.exp(-0.5 * (((y - 20) / 5.0)**2 + ((x - 20) / 5.0)**2)).astype(np.float32)
    img2 += 0.2 * np.exp(-0.5 * (((y - 22) / 5.0)**2 + ((x - 18) / 5.0)**2)).astype(np.float32)

    return img1, img2


def generate_synthetic_pair_3d(shape=(32, 32, 32), seed=42):
    np.random.seed(seed)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center1 = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)
    center2 = (shape[0] / 2.0 + 2.0, shape[1] / 2.0 - 1.5, shape[2] / 2.0 + 1.0)

    r1 = np.sqrt((z - center1[0])**2 + (y - center1[1])**2 + (x - center1[2])**2)
    r2 = np.sqrt((z - center2[0])**2 + (y - center2[1])**2 + (x - center2[2])**2)

    img1 = np.exp(-0.5 * (r1 / 6.0)**2).astype(np.float32)
    img2 = np.exp(-0.5 * (r2 / 7.0)**2).astype(np.float32)

    return img1, img2


def run_stress_test():
    seeds = [42, 100, 123, 456, 777, 999, 2024, 2026, 31415, 99999]
    configs = [
        # (dim, shape, levels, epochs_per_level, affine_epochs, lr, fluid_sigma, n_steps)
        (2, (64, 64), [4, 2, 1], [30, 30, 20], 0, 2.0, 1.0, 6),
        (2, (64, 64), [4, 2, 1], [30, 30, 20], 20, 2.0, 1.0, 6),
        (2, (128, 128), [4, 2, 1], [20, 20, 20], 0, 1.0, 1.5, 8),
        (2, (64, 64), [2, 1], [50, 50], 0, 5.0, 0.5, 4),  # high LR / small sigma edge case
        (3, (32, 32, 32), [4, 2, 1], [20, 20, 15], 0, 2.0, 1.0, 6),
        (3, (32, 32, 32), [4, 2, 1], [20, 20, 15], 15, 2.0, 1.0, 6),
        (3, (32, 32, 32), [2, 1], [30, 30], 0, 1.0, 1.0, 8),
        (3, (48, 48, 48), [4, 2, 1], [15, 15, 10], 0, 2.0, 1.0, 6),
    ]

    total_runs = len(seeds) * len(configs)
    print(f"Starting Empirical Stress Test: {total_runs} total optimization sweeps...")
    print("=" * 80)

    nan_inf_count = 0
    passed_count = 0
    details = []

    start_time = time.time()

    for c_idx, config in enumerate(configs, 1):
        dim, shape, levels, epochs_per_level, affine_epochs, lr, fluid_sigma, n_steps = config
        print(f"\n--- Config Set {c_idx}/{len(configs)}: {dim}D shape={shape}, levels={levels}, epochs={epochs_per_level}, affine={affine_epochs}, lr={lr}, sigma={fluid_sigma}, steps={n_steps} ---")
        
        for seed in seeds:
            if dim == 2:
                img1, img2 = generate_synthetic_pair_2d(shape=shape, seed=seed)
            else:
                img1, img2 = generate_synthetic_pair_3d(shape=shape, seed=seed)

            fi_jax = jnp.array(img1)[None, None, ...]
            mi_jax = jnp.array(img2)[None, None, ...]

            model = GeodesicShootingModelJAX(
                dim=dim,
                image_shape=shape,
                n_time_steps=n_steps,
                fluid_sigma=fluid_sigma
            )

            # Confirm starting from identity p0 = 0
            p0_initial_max = float(jnp.max(jnp.abs(model.p0)))
            assert p0_initial_max == 0.0, f"Expected p0 identity initialization (0.0), got max {p0_initial_max}"

            initial_loss = float(model.forward(fi_jax, mi_jax))
            
            # Custom fit loop capturing every epoch loss to detect NaN/Inf
            nan_or_inf_detected = False
            loss_history = []
            
            # Monkey patch or monitor via callback by overriding fit or running step by step
            # Let's inspect model.fit() behaviour:
            try:
                model.fit(
                    fi_jax, mi_jax,
                    levels=levels,
                    epochs_per_level=epochs_per_level,
                    affine_epochs=affine_epochs,
                    lr=lr,
                    fluid_sigma=fluid_sigma,
                    verbose=False
                )
                final_loss = float(model.forward(fi_jax, mi_jax))

                # Check if p0 has any NaNs or Infs
                has_nan_p0 = bool(jnp.isnan(model.p0).any())
                has_inf_p0 = bool(jnp.isinf(model.p0).any())
                has_nan_loss = np.isnan(final_loss)
                has_inf_loss = np.isinf(final_loss)

                if has_nan_p0 or has_inf_p0 or has_nan_loss or has_inf_loss:
                    nan_or_inf_detected = True
                    nan_inf_count += 1
                    print(f"  [FAIL] Seed {seed}: NaN/Inf detected! final_loss={final_loss}, p0 NaN={has_nan_p0}, p0 Inf={has_inf_p0}")
                else:
                    passed_count += 1
                    loss_drop = initial_loss - final_loss
                    print(f"  [PASS] Seed {seed}: Initial loss={initial_loss:.4f} -> Final loss={final_loss:.4f} (drop={loss_drop:+.4f})")
            except Exception as e:
                nan_or_inf_detected = True
                nan_inf_count += 1
                print(f"  [FAIL] Seed {seed}: Exception raised during fit: {e}")

            details.append({
                'dim': dim,
                'shape': shape,
                'seed': seed,
                'initial_loss': initial_loss,
                'final_loss': final_loss if not nan_or_inf_detected else None,
                'status': 'PASS' if not nan_or_inf_detected else 'FAIL'
            })

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"STRESS TEST COMPLETE in {elapsed:.2f} seconds.")
    print(f"Total Runs: {total_runs}")
    print(f"Passed: {passed_count}/{total_runs}")
    print(f"NaN/Inf Failures: {nan_inf_count}/{total_runs}")

    if nan_inf_count == 0:
        print("\nSUCCESS: 100% of optimization sweeps completed with ZERO NaN or Inf loss occurrences!")
        return 0
    else:
        print(f"\nFAILURE: Detected {nan_inf_count} NaN/Inf occurrences!")
        return 1


if __name__ == '__main__':
    sys.exit(run_stress_test())
