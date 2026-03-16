# PFM-NSI

Precision Functional Mapping - Network Similarity Index (PFM-NSI).

This repository ships two aligned implementations:

- Python package + CLI (`pfm_nsi`, `pfm-nsi` command)
- MATLAB fallback (`matlab/scripts/*.m`)

Both implementations compute the same core outputs:

- NSI (univariate + ridge-based)
- Optional usability projection
- Optional reliability projection
- Optional contextual metrics (Moran's I, spectral slope)
- Advanced binary ROI sparse-target override
- Advanced per-network NSI histograms (beta-based network assignment)
- Advanced per-structure NSI histograms (LH/RH-collapsed, grayscale)

## Install (Python)

```bash
python -m pip install pfm-nsi
```

From source:

```bash
git clone https://github.com/<ORG_OR_USER>/pfm-nsi.git
cd pfm-nsi
python -m pip install -e .
```

## Quick Start

Single dataset (NSI):

```bash
pfm-nsi run --cifti /path/to/Data.dtseries.nii
```

Add usability + reliability projections:

```bash
pfm-nsi run \
  --cifti /path/to/Data.dtseries.nii \
  --usability \
  --reliability \
  --nsi-t 10 \
  --query-t 60
```

Advanced ROI-targeted NSI (no subsampling, structures overridden):

```bash
pfm-nsi run \
  --cifti /path/to/Data.dtseries.nii \
  --roi-binary /path/to/roi_mask.dscalar.nii \
  --network-hists \
  --structure-hists
```

## MATLAB Fallback

See [matlab/README.md](matlab/README.md) for setup and examples.

Core entry points:

- `matlab/scripts/pfm_nsi.m`
- `matlab/scripts/pfm_nsi_core.m`
- `matlab/scripts/pfm_nsi_plots.m`
- `matlab/scripts/conditional_reliability_from_nsi.m`

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [Tutorial](docs/TUTORIAL.md)
- [MATLAB README](matlab/README.md)

## Outputs

Python `run` writes:

- `<prefix>_nsi.npz`
- `<prefix>_hist_nsi.png`
- `<prefix>_hist_nsi_by_network.png` (advanced, `--network-hists`)
- `<prefix>_network_hist_summary.csv/.json` (advanced)
- `<prefix>_hist_nsi_by_structure.png` (advanced, `--structure-hists`)
- `<prefix>_structure_hist_summary.csv/.json` (advanced)
- `<prefix>_reliability.json` (when `--reliability`)

MATLAB writes `.mat` structures with equivalent fields (see user guide mapping table).

## Citation

Preprint: https://www.biorxiv.org/content/10.64898/2026.02.10.704857v1

## License

MIT
