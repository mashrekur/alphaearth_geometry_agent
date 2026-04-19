# Manifold Geometry of AlphaEarth Satellite Foundation Model Embeddings
[![DOI](https://zenodo.org/badge/1179076410.svg)](https://doi.org/10.5281/zenodo.19652729)
Analysis code for:

> **Geometric Characterization and Compositional Reasoning over Satellite Foundation Model Embeddings**  
> Mashrekur Rahman, Samuel Barrett, Christina Last  
> *In preparation*

## Overview

This repository contains the manifold characterization and compositional arithmetic analysis for Google AlphaEarth satellite foundation model embeddings. We analyze the geometric structure of the 64-dimensional embedding space across 12.1 million co-located samples in the Continental United States (2017–2023) at 0.025° grid spacing (~2.75 km).

The analysis proceeds in two phases:

**Phase 1 — Manifold Characterization.** Global covariance structure, intrinsic dimensionality estimation via Levina–Bickel MLE, local PCA and tangent space mapping, and multi-scale geometric analysis across neighborhood sizes.

**Phase 2 — Compositional Arithmetic.** Targeted property shifts using local vs. global directions, property transfer, vector analogies with FAISS retrieval, and retrieval coherence analysis linking geometric structure to downstream performance.

This work builds on the dimension-level interpretability established in [Rahman (2026)](https://doi.org/10.5281/zenodo.18566431), extending from *what individual dimensions encode* to *how the embedding space is structured geometrically*.

## Repository Structure

```
├── phase1_1_covariance.py        # Global covariance, eigendecomposition, dimension clustering
├── phase1_2_intrinsic_dim.py     # Levina–Bickel MLE intrinsic dimensionality
├── phase1_3_local_pca.py         # Local PCA, tangent space mapping, geometric dictionary
├── phase1_4_multiscale.py        # Scale dependence of local geometry (k = 20–2000)
├── phase2_arithmetic.py          # Compositional arithmetic experiments (shift, transfer, analogy)
├── phase2b_retrieval_coherence.py# Retrieval coherence and enhanced geometric dictionary
├── phase2_approach_figure.py     # Conceptual figure: global vs. local arithmetic
├── paper2_final_figures.py       # Publication-quality figure generation
├── requirements.txt              # Python dependencies
└── README.md
```

## Data

This analysis uses the same co-located AlphaEarth embeddings and environmental variables extracted in [Paper 1](https://doi.org/10.5281/zenodo.18566431). No additional data extraction is required. The expected input directory is `../../data/unified_conus/` containing yearly parquet files (`conus_unified_YYYY.parquet` for 2017–2023).

## Pipeline

### Phase 1.1: Global Covariance Structure (`phase1_1_covariance.py`)

Loads a balanced 1M subsample across all seven years and computes the 64×64 covariance and correlation matrices. Eigendecomposes to identify principal directions, effective dimensionality (participation ratio), and variance concentration. Runs hierarchical clustering to identify co-varying dimension groups and cross-references clusters with Paper 1's dimension dictionary. Repeats per-year to test temporal stability of the geometric structure itself, complementing Paper 1's stability of dimension *values*.

```bash
python phase1_1_covariance.py
```

**Key outputs:** `covariance_matrix_64x64.csv`, `eigenvalues.csv`, `eigenvectors.csv`, `dimension_clusters.json`, `per_year_eigenvalues.csv`

### Phase 1.2: Intrinsic Dimensionality (`phase1_2_intrinsic_dim.py`)

Estimates the local intrinsic dimensionality at each sample point using the Levina & Bickel (2004) maximum likelihood estimator across multiple neighborhood sizes (k = 5–100). Computes combined and per-year estimates on 200K and 100K subsamples respectively. Maps spatial variation in intrinsic dimensionality across CONUS.

```bash
python phase1_2_intrinsic_dim.py
```

**Key outputs:** `intrinsic_dim_combined.csv`, `intrinsic_dim_per_year.csv`, figures showing spatial distribution and scale dependence

### Phase 1.3: Local PCA and Tangent Space Mapping (`phase1_3_local_pca.py`)

Computes local PCA at 10K probe locations (k=100 neighborhoods) to characterize tangent space structure. Measures alignment between local and global principal components, tangent space angles between neighboring locations, and location-dependent dimension importance. Constructs the geometric dictionary mapping each probe location to its local geometry summary.

```bash
python phase1_3_local_pca.py
```

**Key outputs:** `local_pca_results.csv`, `geometric_dictionary.json`, figures for alignment maps, tangent stability, and local vs. global dimensionality

### Phase 1.4: Multi-Scale Geometry (`phase1_4_multiscale.py`)

Repeats local PCA at the same 10K probe locations across neighborhood sizes k = 20, 100, 500, 2000. Tracks how local–global alignment, tangent space stability, dominant environmental category, and local dimensionality change with scale. Identifies the crossover scale where local geometry recovers alignment with continental-scale gradients.

```bash
python phase1_4_multiscale.py
```

**Key outputs:** `multiscale_results.csv`, figures showing alignment and stability as functions of neighborhood size

### Phase 2: Compositional Arithmetic (`phase2_arithmetic.py`)

Tests whether compositional operations analogous to word embedding analogies (e.g., "make wetter," "transfer temperature profile," "A is to B as C is to ?") work in the AlphaEarth embedding space. Compares four strategies: global direction shift, local PCA-derived direction shift, random direction, and geographic baseline. Uses FAISS to retrieve nearest on-manifold neighbors after each operation. Evaluates with target accuracy, non-target preservation, on-manifold distance, and retrieval precision@k.

```bash
python phase2_arithmetic.py
```

**Key outputs:** `arithmetic_results.csv`, `analogy_results.csv`, figures comparing strategy performance

### Phase 2B: Retrieval Coherence (`phase2b_retrieval_coherence.py`)

Evaluates whether the geometric structure characterized in Phase 1 predicts the physical coherence of FAISS-retrieved neighborhoods. Computes environmental property variance within retrieved sets at 5K probe locations, correlates coherence with local intrinsic dimensionality and tangent space metrics, and builds regional dimension importance profiles across six CONUS subregions. Produces the enhanced geometric dictionary used downstream.

```bash
python phase2b_retrieval_coherence.py
```

**Key outputs:** `retrieval_coherence.csv`, `enhanced_geo_dictionary.json`, `regional_dimension_importance.csv`

### Figures

`phase2_approach_figure.py` generates a conceptual two-panel figure illustrating why global arithmetic fails (lands off-manifold) while location-aware arithmetic succeeds (follows local curvature). `paper2_final_figures.py` generates all publication figures from the CSV outputs of Phases 1–2.

```bash
python phase2_approach_figure.py
python paper2_final_figures.py
```

## Hardware Requirements

- **Phase 1:** 32 GB RAM recommended; all scripts subsample to configurable sizes
- **Phase 2:** FAISS index required (built from full 12.1M vectors); CUDA-capable GPU recommended for large-scale nearest neighbor search
- **Figures:** Standard workstation

Tested on NVIDIA RTX 5090 (32 GB VRAM), 64 GB system RAM.

## Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
pyarrow>=12.0
faiss-cpu>=1.7  # or faiss-gpu
```

Optional: `cartopy` for geographic map projections in figures.

## Related Work

The dimension-level interpretability analysis (Paper 1) is available at:
- **Code:** [https://doi.org/10.5281/zenodo.18566431](https://doi.org/10.5281/zenodo.18566431)
- **Preprint:** [arXiv:2602.10354](https://arxiv.org/abs/2602.10354)

## Citation

```bibtex
@article{rahman2026manifold,
  title={Geometric Characterization and Compositional Reasoning over Satellite Foundation Model Embeddings},
  author={Rahman, Mashrekur and Barrett, Samuel and Last, Christina},
  year={2026}
}
```

## License

This project is released under the MIT License.
