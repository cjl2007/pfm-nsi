# PFM-NSI User Guide

## 1) What This Tool Does

PFM-NSI evaluates FC map quality for precision functional mapping.

Primary metric:

- `NSI` from ridge regression (`R^2`) between subject FC maps and canonical priors

Optional metrics:

- Moran's I
- Spectral slope
- Usability projection (`P(PFM-usable | NSI)`)
- Reliability projection (`P(R >= threshold | NSI, time)`)

Advanced options:

- Binary ROI sparse-target override (no subsampling)
- Per-network NSI histograms (network assignment from ridge betas)
- Per-structure NSI histograms (LH/RH collapsed, grayscale stacked rows)

## 2) Python CLI

Basic run:

```bash
pfm-nsi run --cifti /path/to/Data.dtseries.nii
```

With usability + reliability:

```bash
pfm-nsi run \
  --cifti /path/to/Data.dtseries.nii \
  --usability \
  --reliability \
  --nsi-t 10 \
  --query-t 30,60
```

Advanced ROI override:

```bash
pfm-nsi run \
  --cifti /path/to/Data.dtseries.nii \
  --roi-binary /path/to/roi_mask.dscalar.nii
```

Advanced per-network histograms:

```bash
pfm-nsi run \
  --cifti /path/to/Data.dtseries.nii \
  --network-hists \
  --network-assignment-lambda 10
```

Advanced per-structure histograms:

```bash
pfm-nsi run \
  --cifti /path/to/Data.dtseries.nii \
  --structure-hists \
  --structure-assignment-lambda 10
```

Notes:

- `--roi-binary` overrides structure-based sparse selection.
- When `--roi-binary` is set, additional sparse subsampling is disabled.
- Network assignment uses the ridge beta vector for each sparse target and picks `argmax(beta)`.

## 3) Python API

```python
from pfm_nsi import pfm_nsi
from pfm_nsi.plots import pfm_nsi_plots

structures = [
    "CORTEX_LEFT", "CEREBELLUM_LEFT", "ACCUMBENS_LEFT", "CAUDATE_LEFT",
    "PALLIDUM_LEFT", "PUTAMEN_LEFT",
        "THALAMUS_LEFT",
        "HIPPOCAMPUS_LEFT", "AMYGDALA_LEFT",
    "CORTEX_RIGHT", "CEREBELLUM_RIGHT", "ACCUMBENS_RIGHT", "CAUDATE_RIGHT",
    "PALLIDUM_RIGHT", "PUTAMEN_RIGHT",
        "THALAMUS_RIGHT",
        "HIPPOCAMPUS_RIGHT", "AMYGDALA_RIGHT",
]

opts = {
    "ridge_lambdas": [10],
    "compute_morans": True,
    "compute_slope": True,
    "compute_network_histograms": True,
    "network_assignment_lambda": 10,
}

qc, maps = pfm_nsi("/path/to/Data.dtseries.nii", structures, "pfm_nsi/models/priors.npz", opts)
pfm_nsi_plots(qc, show_plots=True, network_histograms=True, network_assignment_lambda=10)
```

ROI override in API:

```python
opts["BinaryROI"] = "/path/to/roi_mask.dscalar.nii"
opts["SparseIdxOverrideBypassStructures"] = True
```

## 4) MATLAB Fallback

```matlab
addpath('/path/to/pfm-nsi/matlab/scripts');
addpath(genpath('/path/to/MSCcodebase/Utilities/read_write_cifti'));

load('/path/to/pfm-nsi/matlab/models/priors.mat');
C = ft_read_cifti_mod('/path/to/Data.dtseries.nii');

opts = struct;
opts.ridge_lambdas = 10;
opts.compute_morans = true;
opts.compute_slope = true;
opts.compute_network_histograms = true;
opts.network_assignment_lambda = 10;

Structures = {'CORTEX_LEFT','CORTEX_RIGHT'};
OUT = pfm_nsi(C, ...
    'Priors', Priors, ...
    'Structures', Structures, ...
    'Opts', opts);
```

MATLAB ROI override:

```matlab
opts.BinaryROI = '/path/to/roi_mask.dscalar.nii';
opts.BinaryROIThreshold = 0.5;
```

## 5) Python vs MATLAB Alignment

Options aligned:

- `ridge_lambdas`
- `compute_morans`
- `compute_slope`
- `SparseFrac`
- `SparseIdxOverride`
- `BinaryROI` / binary ROI override behavior
- `compute_network_histograms`
- `network_assignment_lambda`
- `compute_structure_histograms`
- `structure_assignment_lambda`

Core output alignment:

- `NSI.Univariate.MaxRho`
- `NSI.Ridge.LambdaXX.R2`
- `NSI.Ridge.LambdaXX.Betas`
- `NSI.MedianScore`
- `NSI.NetworkAssignment.LambdaXX.NetworkIndex`
- `NSI.NetworkAssignment.LambdaXX.NetworkLabels`
- `NSI.StructureAssignment.LambdaXX.StructureLabelsByTarget`
- `NSI.StructureAssignment.LambdaXX.StructureLabelsUnique`
- `MoransI.mI`
- `SpectralSlope.slope`

## 6) Release Checklist

- Verify CLI help: `pfm-nsi run -h`
- Run one Python + one MATLAB sample and compare `NSI.MedianScore`
- Confirm advanced outputs with ROI and network hist flags
- Tag release and publish README/docs/tutorial
