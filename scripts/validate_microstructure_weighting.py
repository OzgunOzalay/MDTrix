#!/usr/bin/env python3
"""
Validation script for the microstructure-informed SIFT2 weighting modification.

Runs three comparisons:
  1. Original tcksift2 (no microstructure weighting)
  2. Modified tcksift2 with -microstructure_lambda 0 (should produce identical weights)
  3. Modified tcksift2 with default lambda (should show bias toward high-MicroAF streamlines)

Usage:
  python3 validate_microstructure_weighting.py <tracks.tck> <fod.mif> <micro_af.txt> [tcksift2_binary]

Arguments:
  tracks.tck     - input tractogram
  fod.mif        - FOD image
  micro_af.txt   - per-streamline MicroAF values (one per line)
  tcksift2_binary - path to tcksift2 (default: tcksift2)
"""

import sys
import os
import subprocess
import tempfile
import numpy as np


def run_tcksift2(binary, tracks, fod, output, extra_args=None):
    cmd = [binary, tracks, fod, output]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: tcksift2 failed with args {extra_args or []}")
        print(result.stderr)
        sys.exit(1)
    return result


def load_weights(path):
    return np.loadtxt(path)


def weight_stats(weights, label):
    print(f"\n--- {label} ---")
    print(f"  Count:  {len(weights)}")
    print(f"  Mean:   {np.mean(weights):.6f}")
    print(f"  Std:    {np.std(weights):.6f}")
    print(f"  Min:    {np.min(weights):.6f}")
    print(f"  Max:    {np.max(weights):.6f}")
    print(f"  Median: {np.median(weights):.6f}")
    pcts = np.percentile(weights, [5, 25, 75, 95])
    print(f"  Percentiles [5,25,75,95]: {pcts[0]:.6f}, {pcts[1]:.6f}, {pcts[2]:.6f}, {pcts[3]:.6f}")


def print_histogram(weights, label, bins=20):
    counts, edges = np.histogram(weights, bins=bins)
    print(f"\n  Histogram ({label}):")
    max_count = max(counts)
    bar_width = 40
    for i in range(len(counts)):
        bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
        print(f"    [{edges[i]:8.4f}, {edges[i+1]:8.4f}): {'#' * bar_len} ({counts[i]})")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    tracks = sys.argv[1]
    fod = sys.argv[2]
    micro_af_path = sys.argv[3]
    binary = sys.argv[4] if len(sys.argv) > 4 else "tcksift2"

    for f in [tracks, fod, micro_af_path]:
        if not os.path.isfile(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    micro_af = load_weights(micro_af_path)
    print(f"Loaded {len(micro_af)} MicroAF values (mean={np.mean(micro_af):.4f}, range=[{np.min(micro_af):.4f}, {np.max(micro_af):.4f}])")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_orig = os.path.join(tmpdir, "weights_original.txt")
        out_zero = os.path.join(tmpdir, "weights_lambda0.txt")
        out_mod = os.path.join(tmpdir, "weights_modified.txt")

        # --- Run 1: Original (no microstructure) ---
        print("\n========================================")
        print("Run 1: Original SIFT2 (no microstructure)")
        print("========================================")
        run_tcksift2(binary, tracks, fod, out_orig)
        w_orig = load_weights(out_orig)
        weight_stats(w_orig, "Original SIFT2")
        print_histogram(w_orig, "Original")

        # --- Run 2: Microstructure with lambda=0 (should be identical) ---
        print("\n========================================")
        print("Run 2: Microstructure with lambda=0")
        print("========================================")
        run_tcksift2(binary, tracks, fod, out_zero,
                     ["-microstructure_weighting", micro_af_path, "-microstructure_lambda", "0"])
        w_zero = load_weights(out_zero)
        weight_stats(w_zero, "Lambda=0")

        # --- Correctness check ---
        print("\n========================================")
        print("CORRECTNESS CHECK: lambda=0 vs original")
        print("========================================")
        max_diff = np.max(np.abs(w_orig - w_zero))
        mean_diff = np.mean(np.abs(w_orig - w_zero))
        rel_diff = np.max(np.abs(w_orig - w_zero) / (np.abs(w_orig) + 1e-30))
        print(f"  Max absolute difference:  {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        print(f"  Max relative difference:  {rel_diff:.2e}")
        if max_diff < 1e-10:
            print("  PASS: Weights are numerically identical")
        else:
            print("  FAIL: Weights differ (expected identical output with lambda=0)")

        # --- Run 3: Microstructure with default lambda ---
        print("\n========================================")
        print("Run 3: Microstructure with default lambda")
        print("========================================")
        run_tcksift2(binary, tracks, fod, out_mod,
                     ["-microstructure_weighting", micro_af_path])
        w_mod = load_weights(out_mod)
        weight_stats(w_mod, "Modified (default lambda)")
        print_histogram(w_mod, "Modified")

        # --- Bias check: low vs high MicroAF ---
        print("\n========================================")
        print("BIAS CHECK: effect of microstructure prior")
        print("========================================")
        median_af = np.median(micro_af)
        low_mask = micro_af < median_af
        high_mask = micro_af >= median_af

        ratio_orig_low = np.mean(w_orig[low_mask])
        ratio_orig_high = np.mean(w_orig[high_mask])
        ratio_mod_low = np.mean(w_mod[low_mask])
        ratio_mod_high = np.mean(w_mod[high_mask])

        print(f"  MicroAF median split threshold: {median_af:.4f}")
        print(f"  Low-MicroAF streamlines:  {np.sum(low_mask)}")
        print(f"  High-MicroAF streamlines: {np.sum(high_mask)}")
        print(f"\n  Original SIFT2:")
        print(f"    Mean weight (low AF):  {ratio_orig_low:.6f}")
        print(f"    Mean weight (high AF): {ratio_orig_high:.6f}")
        print(f"    Ratio high/low:        {ratio_orig_high / ratio_orig_low:.4f}")
        print(f"\n  Modified SIFT2:")
        print(f"    Mean weight (low AF):  {ratio_mod_low:.6f}")
        print(f"    Mean weight (high AF): {ratio_mod_high:.6f}")
        print(f"    Ratio high/low:        {ratio_mod_high / ratio_mod_low:.4f}")

        weight_change = w_mod - w_orig
        mean_change_low = np.mean(weight_change[low_mask])
        mean_change_high = np.mean(weight_change[high_mask])
        print(f"\n  Weight change (modified - original):")
        print(f"    Mean change for low-AF streamlines:  {mean_change_low:+.6f}")
        print(f"    Mean change for high-AF streamlines: {mean_change_high:+.6f}")

        if mean_change_low < mean_change_high:
            print("  PASS: Low-MicroAF streamlines received systematically lower weights")
        else:
            print("  FAIL: Expected low-MicroAF streamlines to receive lower weights")

        # --- Correlation check ---
        corr = np.corrcoef(weight_change, micro_af)[0, 1]
        print(f"\n  Correlation between weight change and MicroAF: {corr:.4f}")
        if corr > 0:
            print("  PASS: Positive correlation confirms prior is working as intended")
        else:
            print("  WARN: Expected positive correlation between weight change and MicroAF")

        print("\n========================================")
        print("SUMMARY")
        print("========================================")
        print(f"  Lambda=0 identity check:  {'PASS' if max_diff < 1e-10 else 'FAIL'}")
        print(f"  Bias direction check:     {'PASS' if mean_change_low < mean_change_high else 'FAIL'}")
        print(f"  Correlation check:        {'PASS' if corr > 0 else 'WARN'}")


if __name__ == "__main__":
    main()
