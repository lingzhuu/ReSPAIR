# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- `respair.fit` is now the joint estimator: Poisson–log-normal gene likelihood, continuous GP hidden factors
  re-estimated jointly with the loadings and a sparse W, conditional (isozyme-aware) pair test, propagation of the
  uncertainty in F, optional reaction-network prior, and separate gene / metabolite grids.
- The package contains only the method. The simulator (`respair.simulate`) and the baseline methods
  (`respair.benchmarks`) moved to `benchmarks/scripts/simulate.py` and `benchmarks/scripts/baselines.py`.

### Removed
- The step-wise (simplified) estimator and `respair.naive_z`, `respair.spectral_z`, `respair.lambda_gc` from the
  top-level API.

### Other
- Repository reorganised into a `src/` layout with `hatchling` packaging, tests, CI and Sphinx docs.
- Simulation and benchmark scripts moved to `benchmarks/`; all text, result keys and figures are in English.
- `benchmarks/results/sim_5seeds.json` regenerated with the current spectrum-based choice of K (was K = 4).

## [0.1.0]

### Added
- Baseline wrappers in `respair.benchmarks`: LIANA+/SpatialDM, Lee's L, Dutilleul modified t-test.
- Simplified ReSPAIR estimator (`respair.fit`): hidden-factor removal, spectral pair test, negative-control calibration.
- Simulator (`respair.simulate`) and MANTIS re-implementation (`respair.benchmarks`).
