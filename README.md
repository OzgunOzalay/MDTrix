# MDTrix

**MDTrix** is a modified version of [MRtrix3](https://github.com/MRtrix3/mrtrix3) that adds microstructure-informed streamline weighting to `tcksift2`.

## What this fork changes

Standard SIFT2 optimises per-streamline weights to match a whole-brain tractogram to fixel-wise fibre densities. MDTrix extends this with a post-optimisation step that replaces the SIFT2 weights of subcortical–subcortical connections with values derived from a microstructure map.

The SIFT2 optimiser itself is unchanged — no extra term is added to the objective function. Microstructure influence is applied after convergence via `-micro_strength`.

### Post-optimisation blend (`-micro_strength`)

After SIFT2 converges, each Sub-Sub streamline coefficient is blended in log-weight space:

```
coeff_final = (1 - s) * coeff_sift2 + s * log(MicroAF_normalised)
```

In linear weight space this is equivalent to a geometric blend:

```
weight_final = sift2_weight^(1-s) * MicroAF_normalised^s
```

where:
- `s` is the blend strength (0.0–1.0), set with `-micro_strength`
- `MicroAF_normalised` is the per-streamline mean microstructure value, rescaled so its mean matches the mean SIFT2 weight of Sub-Sub streamlines (preserving contrast while keeping magnitudes consistent)

Setting `s = 1.0` fully replaces the SIFT2 weight with the (normalised) microstructure value. Setting `s = 0.0` leaves weights unchanged (pure SIFT2). Intermediate values blend the two.

### How `MicroAF` is computed

Given a 3D volumetric microstructure map (centred at 1.0, range approximately 0.5–1.5):

1. **Samples** the map along each streamline at half-voxel intervals using trilinear interpolation
2. **Computes** the arithmetic mean of the sampled values per streamline
3. **Clamps** values below 0.1 to 0.1 (warning printed if any are affected)
4. **Sets** streamlines with no valid samples to 1.0 (neutral — no influence on weight)

### MicroAF normalisation

The microstructure map uses absolute tissue-fraction values (e.g. MicroWF centred at ~1.0) while SIFT2 weights for densely tracked tractograms are typically two orders of magnitude smaller. To avoid upscaling affected streamlines, MDTrix computes a scale factor from Sub-Sub streamlines only:

```
scale = mean(SIFT2 weight, Sub-Sub) / mean(MicroAF, Sub-Sub)
```

`MicroAF_normalised = MicroAF * scale` preserves the microstructural rank ordering (contrast between streamlines) while anchoring the mean injected weight to the SIFT2 weight scale. The scale factor is printed to stderr on every run so it can be recorded and compared across experiments.

Using only Sub-Sub streamlines for this calculation ensures the scale factor is invariant to changes in how Sub-Cor or Cor-Cor connections are treated — Sub-Sub weights will be identical across two runs that differ only in the Sub-Cor policy.

### Parcellation-based connection classification

When a parcellation atlas and class CSV are provided, each streamline is classified by its endpoint anatomy:

| Endpoint pair | `blend` | Weight source |
|---|---|---|
| Both subcortical (Sub-Sub) | 1.0 | `s * 100%` MicroAF, rest SIFT2 |
| One subcortical, one cortical (Sub-Cor) | 0.0 | Pure SIFT2 |
| Both cortical (Cor-Cor) | 0.0 | Pure SIFT2 |
| Unknown/unclassified | 0.0 | Pure SIFT2 |

Only Sub-Sub connections receive microstructure weighting. All other connections are left exactly as SIFT2 assigned them.

### New CLI flags

| Flag | Description |
|---|---|
| `-microstructure_map <image>` | 3D volumetric microstructure map (`.mif`, `.mif.gz`, `.nii.gz`). Centred at 1.0. Optional — if omitted, behaviour is identical to standard SIFT2 with zero overhead. |
| `-parcellation <image>` | Atlas/parcellation image (integer-labelled) for endpoint-based classification. Must be used with `-parcellation_classes`. |
| `-parcellation_classes <path>` | CSV file mapping parcellation region intensity values to class (`Subcortical` or `Cortical`). Format: `intensity,class`. Must be used with `-parcellation`. |
| `-micro_strength <float>` | Post-optimisation blend strength in range 0.0–1.0. At 1.0, Sub-Sub weights are fully replaced by normalised MicroAF values. Requires `-microstructure_map`. |

### Usage example

```bash
# Standard SIFT2 (unchanged behaviour)
tcksift2 tracks.tck wmfod.mif weights.txt

# With microstructure-informed weights for Sub-Sub connections (full replacement)
tcksift2 tracks.tck wmfod.mif weights.txt \
    -microstructure_map MicroWF.mif.gz \
    -parcellation atlas.mif \
    -parcellation_classes labels.csv \
    -micro_strength 1.0

# With partial blend (80% microstructure, 20% SIFT2 for Sub-Sub)
tcksift2 tracks.tck wmfod.mif weights.txt \
    -microstructure_map MicroWF.mif.gz \
    -parcellation atlas.mif \
    -parcellation_classes labels.csv \
    -micro_strength 0.8
```

The parcellation CSV should have the format:

```
intensity,class
1,Subcortical
2,Subcortical
3,Cortical
...
```

### Safeguards and diagnostics

- **Scale mismatch prevention:** MicroAF is automatically rescaled to the SIFT2 weight magnitude before blending. The scale factor is printed unconditionally to stderr.
- **Floor clamping:** MicroAF values below 0.1 are clamped to 0.1 with a warning reporting the count.
- **Missing samples:** Streamlines with no valid microstructure samples (all points outside the image FOV) receive MicroAF = 1.0 (neutral weight after normalisation).
- **Unknown anatomy:** Streamlines with endpoints outside the parcellation FOV or in regions not listed in the CSV receive blend = 0.0 (pure SIFT2).

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
| `src/dwi/tractography/SIFT2/tckfactor.cpp` | Microstructure map loading, streamline sampling, parcellation classification, post-optimisation blend (`apply_micro_strength`) |

## Acknowledgements

MDTrix is built on [MRtrix3](https://www.mrtrix.org/). All original MRtrix3 functionality is preserved. See the MRtrix3 [documentation](https://mrtrix.readthedocs.io/) and [community forum](http://community.mrtrix.org/) for general usage.
