#!/usr/bin/env python3
"""
Validation script for the microstructure-informed SIFT2 weighting modification.

Runs two comparisons:
  1. Original tcksift2 (no microstructure map)
  2. Modified tcksift2 with -microstructure_map (should show bias toward high-MicroAF streamlines)

Then verifies that streamlines with high variance in microstructure sampling
receive systematically lower weights than those with low variance.

Usage:
  python3 validate_microstructure_weighting.py <tracks.tck> <fod.mif> <micro_map.mif> [tcksift2_binary]

Arguments:
  tracks.tck       - input tractogram
  fod.mif          - FOD image
  micro_map.mif    - 3D volumetric microstructure map (normalised axonal density)
  tcksift2_binary  - path to tcksift2 (default: tcksift2)
"""

import sys
import os
import subprocess
import tempfile
import struct

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


def read_tck_streamlines(tck_path):
    """Read streamline points from a .tck file."""
    streamlines = []
    with open(tck_path, 'rb') as f:
        # Read header
        header = {}
        while True:
            line = f.readline().decode('latin-1').strip()
            if line == 'END':
                break
            if ':' in line:
                key, val = line.split(':', 1)
                header[key.strip()] = val.strip()

        offset = int(header.get('file', '. 0').split()[-1])
        f.seek(offset)

        current_streamline = []
        while True:
            data = f.read(12)
            if len(data) < 12:
                break
            x, y, z = struct.unpack('<fff', data)
            if np.isinf(x) and np.isinf(y) and np.isinf(z):
                break
            if np.isnan(x) and np.isnan(y) and np.isnan(z):
                if current_streamline:
                    streamlines.append(np.array(current_streamline))
                    current_streamline = []
            else:
                current_streamline.append([x, y, z])

        if current_streamline:
            streamlines.append(np.array(current_streamline))

    return streamlines


def sample_microstructure_along_streamline(streamline, interp_func, sampling_interval):
    """Sample microstructure map along a streamline at regular intervals."""
    values = []
    if len(streamline) < 2:
        return np.array(values)

    dist_along_segment = 0.0
    for i in range(len(streamline) - 1):
        p0 = streamline[i]
        p1 = streamline[i + 1]
        segment = p1 - p0
        seg_len = np.linalg.norm(segment)
        if seg_len < 1e-12:
            dist_along_segment = 0.0
            continue

        while dist_along_segment <= seg_len:
            frac = dist_along_segment / seg_len
            point = p0 + frac * segment
            val = interp_func(point)
            if val is not None and np.isfinite(val) and val > 0:
                values.append(val)
            dist_along_segment += sampling_interval

        dist_along_segment -= seg_len

    return np.array(values)


def compute_per_streamline_stats(streamlines, micro_map_path, sampling_interval=None):
    """Compute mean and variance of microstructure samples per streamline.

    This is a simplified Python re-implementation for validation purposes.
    Uses scipy for image loading and interpolation if available, otherwise
    reports that detailed per-streamline analysis is not available.
    """
    try:
        import nibabel as nib
        from scipy.ndimage import map_coordinates
    except ImportError:
        return None, None, None

    # Try loading the image
    try:
        img = nib.load(micro_map_path)
    except Exception:
        return None, None, None

    data = img.get_fdata()
    affine = img.affine
    inv_affine = np.linalg.inv(affine)
    voxel_sizes = np.abs(np.diag(affine)[:3])

    if sampling_interval is None:
        sampling_interval = 0.5 * np.min(voxel_sizes)

    def interp_func(point_scanner):
        # Convert scanner coordinates to voxel coordinates
        vox = inv_affine @ np.append(point_scanner, 1.0)
        vox = vox[:3]
        # Check bounds
        for d in range(3):
            if vox[d] < -0.5 or vox[d] > data.shape[d] - 0.5:
                return None
        val = map_coordinates(data, vox.reshape(3, 1), order=1, mode='constant', cval=np.nan)[0]
        return val if np.isfinite(val) else None

    means = np.zeros(len(streamlines))
    variances = np.zeros(len(streamlines))
    effectives = np.zeros(len(streamlines))

    for i, sl in enumerate(streamlines):
        samples = sample_microstructure_along_streamline(sl, interp_func, sampling_interval)
        if len(samples) > 0:
            m = np.mean(samples)
            v = np.var(samples) if len(samples) > 1 else 0.0
            means[i] = m
            variances[i] = v
            eff = m / (1.0 + max(0.0, v))
            effectives[i] = max(eff, 1e-6)
        else:
            means[i] = 1.0
            variances[i] = 0.0
            effectives[i] = 1.0

    return means, variances, effectives


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    tracks = sys.argv[1]
    fod = sys.argv[2]
    micro_map_path = sys.argv[3]
    binary = sys.argv[4] if len(sys.argv) > 4 else "tcksift2"

    for f in [tracks, fod, micro_map_path]:
        if not os.path.isfile(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_orig = os.path.join(tmpdir, "weights_original.txt")
        out_mod = os.path.join(tmpdir, "weights_modified.txt")

        # --- Run 1: Original (no microstructure) ---
        print("\n========================================")
        print("Run 1: Original SIFT2 (no microstructure)")
        print("========================================")
        run_tcksift2(binary, tracks, fod, out_orig)
        w_orig = load_weights(out_orig)
        weight_stats(w_orig, "Original SIFT2")
        print_histogram(w_orig, "Original")

        # --- Run 2: Microstructure with default lambda ---
        print("\n========================================")
        print("Run 2: Microstructure map with default lambda")
        print("========================================")
        run_tcksift2(binary, tracks, fod, out_mod,
                     ["-microstructure_map", micro_map_path])
        w_mod = load_weights(out_mod)
        weight_stats(w_mod, "Modified (default lambda)")
        print_histogram(w_mod, "Modified")

        # --- Weight comparison ---
        print("\n========================================")
        print("WEIGHT COMPARISON: original vs modified")
        print("========================================")
        weight_change = w_mod - w_orig
        print(f"  Mean weight change:     {np.mean(weight_change):+.6f}")
        print(f"  Std of weight change:   {np.std(weight_change):.6f}")
        print(f"  Min weight change:      {np.min(weight_change):+.6f}")
        print(f"  Max weight change:      {np.max(weight_change):+.6f}")

        # --- Per-streamline variance analysis ---
        print("\n========================================")
        print("VARIANCE-BASED ANALYSIS")
        print("========================================")
        print("Computing per-streamline microstructure statistics...")

        streamlines = read_tck_streamlines(tracks)
        print(f"  Read {len(streamlines)} streamlines from track file")

        means, variances, effectives = compute_per_streamline_stats(streamlines, micro_map_path)

        if effectives is not None and len(effectives) == len(w_orig):
            # Key correctness check: streamlines with high variance should get lower weights
            median_var = np.median(variances)
            low_var_mask = variances <= median_var
            high_var_mask = variances > median_var

            print(f"\n  Variance median split threshold: {median_var:.6f}")
            print(f"  Low-variance streamlines:  {np.sum(low_var_mask)}")
            print(f"  High-variance streamlines: {np.sum(high_var_mask)}")

            mean_change_low_var = np.mean(weight_change[low_var_mask])
            mean_change_high_var = np.mean(weight_change[high_var_mask])

            print(f"\n  Mean weight change (low variance):  {mean_change_low_var:+.6f}")
            print(f"  Mean weight change (high variance): {mean_change_high_var:+.6f}")

            if mean_change_high_var < mean_change_low_var:
                print("  PASS: High-variance streamlines received systematically lower weights")
            else:
                print("  FAIL: Expected high-variance streamlines to receive lower weights")

            # MicroAF_effective analysis
            median_eff = np.median(effectives)
            low_eff_mask = effectives < median_eff
            high_eff_mask = effectives >= median_eff

            mean_change_low_eff = np.mean(weight_change[low_eff_mask])
            mean_change_high_eff = np.mean(weight_change[high_eff_mask])

            print(f"\n  MicroAF_effective median: {median_eff:.6f}")
            print(f"  Mean weight change (low effective):  {mean_change_low_eff:+.6f}")
            print(f"  Mean weight change (high effective): {mean_change_high_eff:+.6f}")

            if mean_change_low_eff < mean_change_high_eff:
                print("  PASS: Low MicroAF_effective streamlines received lower weights")
            else:
                print("  FAIL: Expected low MicroAF_effective to receive lower weights")

            # Correlation check
            corr_eff = np.corrcoef(weight_change, effectives)[0, 1]
            corr_var = np.corrcoef(weight_change, variances)[0, 1]
            print(f"\n  Correlation(weight_change, MicroAF_effective): {corr_eff:+.4f}")
            print(f"  Correlation(weight_change, variance):          {corr_var:+.4f}")

            if corr_eff > 0:
                print("  PASS: Positive correlation with effective confirms prior works")
            else:
                print("  WARN: Expected positive correlation with MicroAF_effective")

            if corr_var < 0:
                print("  PASS: Negative correlation with variance confirms suppression")
            else:
                print("  WARN: Expected negative correlation with variance")

            # Summary
            print("\n========================================")
            print("SUMMARY")
            print("========================================")
            variance_check = mean_change_high_var < mean_change_low_var
            effective_check = mean_change_low_eff < mean_change_high_eff
            corr_eff_check = corr_eff > 0
            corr_var_check = corr_var < 0

            print(f"  Variance bias check:       {'PASS' if variance_check else 'FAIL'}")
            print(f"  Effective value bias check: {'PASS' if effective_check else 'FAIL'}")
            print(f"  Effective correlation:      {'PASS' if corr_eff_check else 'WARN'}")
            print(f"  Variance correlation:       {'PASS' if corr_var_check else 'WARN'}")

        else:
            print("  Could not compute per-streamline statistics.")
            print("  Install nibabel and scipy for detailed analysis:")
            print("    pip install nibabel scipy")
            print("\n  Falling back to simple weight distribution comparison...")

            # Simple check: modified weights should differ from original
            max_diff = np.max(np.abs(weight_change))
            print(f"  Max absolute difference: {max_diff:.6f}")
            if max_diff > 1e-6:
                print("  PASS: Modified weights differ from original (microstructure prior active)")
            else:
                print("  FAIL: Weights are identical (prior may not be active)")


if __name__ == "__main__":
    main()
