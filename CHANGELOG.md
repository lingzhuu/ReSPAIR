# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Repository reorganised into a `src/` layout with `hatchling` packaging, tests, CI and Sphinx docs.
- Simulation and benchmark scripts moved to `benchmarks/`.

## [0.1.0]

### Added
- Simplified ReSPAIR estimator (`respair.fit`): hidden-factor removal, spectral pair test, negative-control calibration.
- Simulator (`respair.simulate`) and MANTIS re-implementation (`respair.benchmarks`).
