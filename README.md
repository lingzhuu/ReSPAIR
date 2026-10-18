# ReSPAIR

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![License: MIT][badge-license]](LICENSE)
[![Python][badge-python]](pyproject.toml)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lingzhuu/ReSPAIR/test.yaml?branch=main&label=tests
[badge-docs]: https://img.shields.io/readthedocs/respair
[badge-license]: https://img.shields.io/badge/license-MIT-blue.svg
[badge-python]: https://img.shields.io/badge/python-3.10%2B-blue.svg
[tests]: https://github.com/lingzhuu/ReSPAIR/actions/workflows/test.yaml
[documentation]: https://respair.readthedocs.io

**ReSPAIR** finds *pair-specific* gene–metabolite links
in spatial transcriptomics + spatial metabolomics data measured on the same tissue section, **after removing known
tissue structure and every hidden spatial pattern** that moves many molecules together.

<p align="center"><img src="benchmarks/results/figures/fig1_overview.png" width="720" alt="ReSPAIR overview"></p>

## Why

Plain correlation between a gene and a metabolite map is dominated by tissue regions and unlabeled gradients
(hypoxia, inflammation, section artefacts). ReSPAIR models both modalities as

```
G = Z B_gᵀ + F Λ_gᵀ + E_g
M = Z B_mᵀ + F Λ_mᵀ + E_g Wᵀ + E_m
```

where `Z` is known structure (e.g. cell-type proportions), `F` are hidden spatial factors treated as confounding,
and the sparse matrix `W` holds the links to be tested: does metabolite *k* follow the *local residual* of gene *j*?

`respair.fit` estimates `F`, the loadings and `W` jointly:

1. Genes follow a Poisson–log-normal count model; the gene residual `E_g` is the posterior mean of its latent
   log-scale deviation. Metabolites are log-transformed.
2. The hidden factors `F` are smooth spatial functions (inducing-point Gaussian processes, one length scale per
   factor), so genes and metabolites can be measured at different locations.
3. `F`, the loadings and a sparse `W` are updated in turn, and the link term is removed from the metabolites before
   `F` is re-estimated, so `F` does not absorb the links.
4. Each pair is tested on a spatial frequency basis with a null variance from the power spectra of both sides
   (accounts for spatial autocorrelation), conditionally on the other genes selected for the same metabolite;
   the uncertainty in `F` is carried into the test, and z-scores can be calibrated with negative-control genes.
5. An optional reaction-network prior gives a posterior inclusion probability per pair.

## Installation

ReSPAIR needs Python ≥ 3.10.

```bash
pip install git+https://github.com/lingzhuu/ReSPAIR.git
```

For development (tests, benchmark scripts):

```bash
git clone https://github.com/lingzhuu/ReSPAIR.git
cd ReSPAIR
pip install -e ".[dev]"
pre-commit install
```

## Quick start

```python
import respair
import sys
sys.path.insert(0, "benchmarks/scripts")      # simulated example data; run from the repository root
from simulate import simulate

d = simulate(seed=0)                        # or your own data, see below
res = respair.fit(
    d["G"],                                 # n_spots × n_genes counts
    d["M"],                                 # n_pixels × n_metabolites intensities
    d["coords"],                            # n_spots × 2 gene coordinates
    coords_m=None,                          # n_pixels × 2 metabolite coordinates (None: same as coords)
    Z=d["Z"],                               # known structure, e.g. cell-type proportions (optional)
    neg_genes=d["neg_genes"],               # negative-control gene indices (optional)
    prior_pairs=None,                       # n_genes × n_metabolites boolean reaction-network prior (optional)
)
res["z"], res["p"]                          # n_genes × n_metabolites
res["W"]                                    # sparse link estimate (n_metabolites × n_genes)
res["pip"]                                  # posterior inclusion probability
hits = respair.bh(res["p"], alpha=0.1)      # Benjamini–Hochberg
```

See [docs/model.md](docs/model.md) for the model and the estimator.

## Simulation benchmark

50 × 50 grid, 2,000 genes, 200 metabolites, 3 cell types as known structure, one unlabeled smooth confounder
affecting 300 genes and 40 metabolites, private hidden factors on each side, 150 true links (strength 0.5),
negative-binomial counts. Mean over 5 seeds:

| Method | AUC (all null pairs) | Power at 5% FPR | AUC (confounded pairs only) | Power at 5% FPR |
|---|---|---|---|---|
| Pearson correlation | 0.888 | 0.544 | 0.763 | 0.275 |
| MANTIS-SCI | 0.539 | 0.069 | 0.389 | 0.029 |
| MANTIS-SPC-CT | 0.762 | 0.203 | 0.194 | 0.000 |
| **ReSPAIR** | **1.000** | **1.000** | **1.000** | **1.000** |

This simulation follows ReSPAIR's own model, so it favours ReSPAIR. Scenarios that break the model and a
semi-synthetic benchmark on a real SMA section are in [`benchmarks/`](benchmarks/README.md); MANTIS is our
re-implementation of the bioRxiv preprint.

## Documentation

- Model and design rationale: [docs/model.md](docs/model.md)
- API reference: built with Sphinx from `docs/` (`pip install -e ".[doc]" && make -C docs html`)

## Roadmap

- AnnData / SpatialData input.
- Sparse eigensolvers and memory limits for > 10k spots or pixels.
- Application to the striatum SMA dataset of Vicari et al.

## Citation

A manuscript is in preparation. Until then, please cite the software via [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE)
