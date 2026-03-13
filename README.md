# MDTrix

**MDTrix** is a modified version of [MRtrix3](https://github.com/MRtrix3/mrtrix3) that adds microstructure-informed streamline weighting to `tcksift2`.

## What this fork changes

Standard SIFT2 optimises per-streamline weights to match a whole-brain tractogram to fixel-wise fibre densities. MDTrix extends this by adding an optional microstructure-informed prior term to the SIFT2 objective function, allowing anatomical plausibility information from a microstructure map to influence the final streamline weights.

### Modified objective function

The standard SIFT2 cost function has two terms — a data fidelity term and a regularisation term:

```
Minimise:  Σ_f (FOD_f − Σ_i w_i · l_if)²  +  λ₁ · Σ_i (w_i · log(w_i))
```

MDTrix adds a third, microstructure-informed prior term:

```
Minimise:  Σ_f (FOD_f − Σ_i w_i · l_if)²  +  λ₁ · Σ_i (w_i · log(w_i))  +  λ₂ · Σ_i (w_i / MicroAF_effective_i)
```

where `MicroAF_effective_i` is a data-driven per-streamline quantity computed internally from a user-supplied volumetric microstructure map.

### How `MicroAF_effective` is computed

Given a 3D volumetric microstructure map (normalised axonal density, mean WM value = 1.0), MDTrix:

1. **Samples** the map along each streamline at half-voxel intervals (derived from the image header at runtime)
2. **Computes** the mean and variance of the sampled values per streamline
3. **Derives** the effective microstructure value:

```
MicroAF_effective_i = mean_i / (1 + variance_i)
```

This formula has no free parameters. The variance itself acts as the weighting mechanism:

| Tissue type | Variance | Effect |
|---|---|---|
| Coherent WM (e.g. compact subcortical bundles) | Low | Denominator ≈ 1, full prior strength |
| Heterogeneous WM (e.g. long tracts through crossings) | High | Denominator >> 1, prior automatically suppressed |

### New CLI flags

| Flag | Description |
|---|---|
| `-microstructure_map <image>` | Path to a 3D volumetric microstructure map (`.mif`, `.mif.gz`, `.nii.gz`). Optional — if omitted, behaviour is identical to standard SIFT2 with zero overhead. |
| `-microstructure_lambda <float>` | Strength of the microstructure prior. Default: `0.05`. |

### Usage example

```bash
# Standard SIFT2 (unchanged behaviour)
tcksift2 tracks.tck wmfod.mif weights.txt

# With microstructure-informed prior
tcksift2 tracks.tck wmfod.mif weights.txt \
    -microstructure_map MicroAF.mif.gz \
    -microstructure_lambda 0.05
```

### Safeguards

- If `MicroAF_effective` falls below `1e-6` for any streamline, it is clamped to `1e-6` and a warning reports how many streamlines were affected.
- Streamlines with no valid samples (e.g. all points outside the image FOV) receive a neutral value of `1.0` (no effect on their weight).

### Validation

A Python validation script is included at `scripts/validate_microstructure_weighting.py`:

```bash
python3 scripts/validate_microstructure_weighting.py \
    tracks.tck wmfod.mif MicroAF.mif.gz [/path/to/tcksift2]
```

It runs standard and modified SIFT2 on the same data and verifies that:
- Streamlines with high microstructure variance receive systematically lower weights
- Weight changes correlate positively with `MicroAF_effective` and negatively with variance

## Building

MDTrix builds exactly like MRtrix3:

```bash
./configure
./build
```

Dependencies: Python (>=2.6), C++11 compiler, Eigen (>=3.2.8), zlib, OpenGL (>=3.3), Qt (>=4.8).

## Source code

The microstructure-informed SIFT2 implementation lives in:

| File | Role |
|---|---|
| `cmd/tcksift2.cpp` | Command entry point and CLI flag definitions |
| `src/dwi/tractography/SIFT2/tckfactor.h` | `TckFactor` class declaration |
| `src/dwi/tractography/SIFT2/tckfactor.cpp` | Microstructure map loading, streamline sampling, `MicroAF_effective` computation |
| `src/dwi/tractography/SIFT2/line_search.h` | Line search functor with microstructure cost term |
| `src/dwi/tractography/SIFT2/line_search.cpp` | Cost function evaluation including microstructure gradient |

## Acknowledgements

MDTrix is built on [MRtrix3](https://www.mrtrix.org/). All original MRtrix3 functionality is preserved. See the MRtrix3 [documentation](https://mrtrix.readthedocs.io/) and [community forum](http://community.mrtrix.org/) for general usage.
