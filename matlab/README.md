# MATLAB Implementation (PFM-NSI Fallback)

This folder provides the MATLAB fallback aligned to the Python CLI/API.

## Contents

- `scripts/pfm_qc.m`
- `scripts/pfm_qc_plots.m`
- `scripts/conditional_reliability_from_nsi.m`
- `scripts/local_predict_binom_logit.m`
- `models/Priors.mat`
- `models/NSI_usability_model.mat`
- `models/NSI_reliability_model.mat`
- `models/Cifti_surf_neighbors_LR_normalwall.mat`
- `examples/ExampleUse.m`

## Requirements

- MATLAB
- MSC CIFTI I/O utilities (`ft_read_cifti_mod`)
  - https://github.com/MidnightScanClub/MSCcodebase

## Setup

```matlab
repo_root = '/path/to/pfm-nsi';
addpath(fullfile(repo_root, 'matlab', 'scripts'));
addpath(genpath('/path/to/MSCcodebase/Utilities/read_write_cifti'));
```

## Basic Usage

```matlab
load(fullfile(repo_root, 'matlab', 'models', 'Priors.mat'));
C = ft_read_cifti_mod('/path/to/Data.dtseries.nii');

opts = struct;
opts.ridge_lambdas = 10;
opts.compute_morans = false;
opts.compute_slope = false;

Structures = {'CORTEX_LEFT','CORTEX_RIGHT'};
[QcPfm, Maps] = pfm_qc(C, Structures, Priors, opts);
OUT = pfm_qc_plots(QcPfm);
```

## Advanced Options

Binary ROI sparse-target override (advanced, non-default):

```matlab
opts.BinaryROI = '/path/to/roi_mask.dscalar.nii';
opts.BinaryROIThreshold = 0.5;
```

Per-network NSI histograms (advanced):

```matlab
opts.compute_network_histograms = true;
opts.network_assignment_lambda = 10;
opts.compute_structure_histograms = true;
opts.structure_assignment_lambda = 10;
```

When enabled, `QcPfm.NSI.NetworkAssignment.Lambda10` includes:

- `NetworkIndex` (assigned network for each sparse target)
- `NetworkLabels`
- `Summary` (`n_targets`, `median_nsi`, `mean_nsi` per network)

Structure histograms use `QcPfm.NSI.StructureAssignment.Lambda10` with:

- `StructureLabelsByTarget`
- `StructureLabelsUnique`
- `Summary` (`n_targets`, `median_nsi`, `mean_nsi` per structure)

## Alignment Notes (Python vs MATLAB)

Core fields are aligned:

- `NSI.Univariate.MaxRho`
- `NSI.Ridge.LambdaXX.R2`
- `NSI.Ridge.LambdaXX.Betas`
- `NSI.MedianScore`
- `NSI.NetworkAssignment.LambdaXX.*` (advanced)
- `MoransI.mI`
- `SpectralSlope.slope`

File formats differ (`.mat` vs `.npz/.json`), but metric definitions are matched.
