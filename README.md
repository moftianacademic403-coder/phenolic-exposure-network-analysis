# Phenolic Exposure–Effect Network Analysis

Reproducible Python workflow supporting the manuscript:

> **Integrated Biomonitoring of BPA, Triclosan, and Triclocarban in Municipal Sanitation Workers: A Stability-Validated Network Analysis of Multi-System Biomarker Associations**

## Overview

This repository implements the complete exposure–effect network analysis for serum bisphenol A (BPA), triclosan (TCS), and triclocarban (TCC) in the exposed cohort. The workflow evaluates 93 exposure–effect pairs (3 phenolic compounds × 31 biological effect biomarkers) and reproduces the statistical validation steps reported in the manuscript.

The analysis includes:

- Spearman exposure–effect correlations;
- Benjamini–Hochberg false discovery rate correction within each chemical family;
- directed-by-design bipartite network construction;
- unweighted and weighted degree metrics;
- 1,000-resample non-parametric bootstrap stability analysis;
- covariate-adjusted partial Spearman correlations;
- cross-validated graphical LASSO sensitivity analysis;
- publication-ready network, stability, and robustness figures;
- supplementary Tables S4–S8.

## Repository structure

```text
.
├── data/
│   └── README.md
├── notebooks/
│   └── phenolic_network_analysis.ipynb
├── outputs/
│   └── README.md
├── src/
│   └── phenolic_network_analysis.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Data availability

Participant-level data are not included because they may be subject to ethical and confidentiality restrictions. The workflow expects an Excel workbook named `total.xlsx` in `data/`, with a worksheet named `total` and the variables documented in `data/README.md`.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Running the analysis

Place the input workbook at:

```text
data/total.xlsx
```

Then run:

```bash
python src/phenolic_network_analysis.py
```

Custom paths can be supplied as command-line arguments:

```bash
python src/phenolic_network_analysis.py \
  --input /path/to/total.xlsx \
  --output /path/to/outputs
```

The same workflow can be run interactively from `notebooks/phenolic_network_analysis.ipynb`.

## Reproducibility settings

The manuscript analysis uses:

- exposed-group code: `1`;
- bootstrap resamples: `1,000`;
- random seed: `42`;
- FDR threshold: `0.05`;
- robust-core threshold: `0.70`;
- covariates: age, body mass index, cigarette smoking, hookah use, and job duration.

## Generated outputs

The workflow writes the following files to `outputs/`:

- `BPA_extracted_from_total.csv`
- `Phenolic_network_validation_tables.xlsx`
- `Fig 2. Network visualization of phenolic compounds exposure and effect biomarkers.png`
- `Fig 3. Edge-Weight stability of the exposure effect network.png`
- `Fig 4. robustness matrix of each edge across validation criteria.png`

The Excel workbook contains Tables S4–S8 covering connectivity, FDR correction, bootstrap stability, robustness criteria, and conditional associations retained in the regularised Gaussian graphical model.

## Interpretation

Edge direction is imposed only to distinguish exposure and effect node classes. It is not inferred from the symmetric Spearman coefficients and must not be interpreted as evidence of biological directionality or causality. The cross-sectional network is therefore an exploratory, hypothesis-generating association map.

## Citation

When citing the code, use this repository URL together with the release or commit used for the analysis:

`https://github.com/moftianacademic403-coder/phenolic-exposure-network-analysis`
