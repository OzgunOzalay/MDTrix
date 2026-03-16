# MDTrix

**MDTrix** is a modified version of [MRtrix3](https://github.com/MRtrix3/mrtrix3) that adds microstructure-informed streamline weighting to `tcksift2`.

## What this fork changes

Standard SIFT2 optimises per-streamline weights to match a whole-brain tractogram to fixel-wise fibre densities. MDTrix extends this by adding an optional microstructure-informed prior term to the SIFT2 objective function, allowing anatomical plausibility information from a microstructure map to influence the final streamline weights.

### Modified objective function

The standard SIFT2 cost function has two terms — a data fidelity term and regularisation terms (Tikhonov and Total Variation):

```
Minimise:  Σ_f PM_f · (FOD_f − μ · TD_f)²  +  λ_tik · A · Σ_i coeff_i²  +  λ_tv · A · Σ_i TV(coeff_i)
```

MDTrix adds a microstructure-informed prior term:

```
+  λ_micro · A · Σ_i blend_i · (coeff_i − log(MicroAF_i))²
```

where:
- `coeff_i` is the log-domain weighting coefficient for streamline *i* (the output weight is `exp(coeff_i)`)
- `MicroAF_i` is the mean microstructure value sampled along streamline *i*
- `blend_i` is a per-streamline factor in {0.0, 0.5, 1.0} determined by endpoint anatomy (see parcellation below)
- `A = Σ_f(PM_f · FOD_f²) / N_tracks` is the same scaling constant used by Tikhonov and TV regularisation

This penalty pulls each streamline's coefficient toward `log(MicroAF_i)` — the log of the mean microstructure value sampled along that streamline. The strength of this pull is controlled by `λ_micro`, `A`, and the blend factor.

### How `MicroAF` is computed

Given a 3D volumetric microstructure map (normalised axonal density, mean WM value ≈ 1.0), MDTrix:

1. **Samples** the map along each streamline at half-voxel intervals using trilinear interpolation
2. **Computes** the arithmetic mean of the sampled values per streamline
3. **Clamps** values below 0.1 to 0.1 (with a warning reporting how many streamlines were affected)
4. **Sets** streamlines with no valid samples to 1.0 (neutral — no influence on the coefficient)

### Parcellation-based blending

When a parcellation atlas and class CSV are provided, MDTrix classifies each streamline's endpoints and modulates the microstructure prior strength accordingly:

| Endpoint pair | `blend_i` | Effect |
|---|---|---|
| Both subcortical | 1.0 | Full microstructure prior |
| One subcortical, one cortical | 0.5 | Half-strength prior |
| Both cortical | 0.0 | Pure SIFT2 (no microstructure influence) |
| Unknown/unclassified endpoints | 0.0 | Pure SIFT2 (no microstructure influence) |

This reflects the assumption that microstructure maps are most informative for compact subcortical bundles and least informative for cortical terminations.

### New CLI flags

| Flag | Description |
|---|---|
| `-microstructure_map <image>` | Path to a 3D volumetric microstructure map (`.mif`, `.mif.gz`, `.nii.gz`). Optional — if omitted, behaviour is identical to standard SIFT2 with zero overhead. |
| `-microstructure_lambda <float>` | Strength of the microstructure prior. Default: `0.05`. Scaled internally by the constant A. |
| `-parcellation <image>` | Atlas/parcellation image (integer-labelled) for endpoint-based classification. Must be used with `-parcellation_classes`. |
| `-parcellation_classes <path>` | CSV file mapping parcellation region intensity values to class (`Subcortical` or `Cortical`). Format: `intensity,class`. Must be used with `-parcellation`. |

### Usage example

```bash
# Standard SIFT2 (unchanged behaviour)
tcksift2 tracks.tck wmfod.mif weights.txt

# With microstructure-informed prior
tcksift2 tracks.tck wmfod.mif weights.txt \
    -microstructure_map MicroAF.mif.gz \
    -microstructure_lambda 0.05

# With microstructure prior and parcellation-based blending
tcksift2 tracks.tck wmfod.mif weights.txt \
    -microstructure_map MicroAF.mif.gz \
    -microstructure_lambda 0.05 \
    -parcellation atlas.mif \
    -parcellation_classes labels.csv
```

### Safeguards

- If `MicroAF` falls below `0.1` for any streamline, it is clamped to `0.1` and a warning reports how many streamlines were affected. This prevents extreme penalties from `log(MicroAF)` dominating the cost function.
- Streamlines with no valid samples (e.g. all points outside the image FOV) receive a neutral value of `1.0` (no effect on their weight).
- Streamlines whose endpoints land in regions not listed in the parcellation CSV receive `blend = 0.0` (no microstructure influence), to avoid applying the prior in uncharacterised anatomy.

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
| `src/dwi/tractography/SIFT2/tckfactor.cpp` | Microstructure map loading, streamline sampling, parcellation classification |
| `src/dwi/tractography/SIFT2/line_search.h` | Line search functor with microstructure cost term |
| `src/dwi/tractography/SIFT2/line_search.cpp` | Cost function evaluation including microstructure gradient |

## Acknowledgements

MDTrix is built on [MRtrix3](https://www.mrtrix.org/). All original MRtrix3 functionality is preserved. See the MRtrix3 [documentation](https://mrtrix.readthedocs.io/) and [community forum](http://community.mrtrix.org/) for general usage.
