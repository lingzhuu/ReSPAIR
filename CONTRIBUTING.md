# Contributing

```bash
git clone https://github.com/lingzhuu/ReSPAIR.git && cd ReSPAIR
pip install -e ".[dev]"
pre-commit install
```

- Tests: `pytest` (fast); `pytest --runslow` also runs a full-size simulation.
- Lint: `pre-commit run --all-files` (ruff).
- Docs: `pip install -e ".[doc]" && make -C docs html`.
- Work on a branch and open a pull request against `main`; CI must pass.
- Add a line to `CHANGELOG.md` under *Unreleased* for user-visible changes.

Releases: bump `version` in `pyproject.toml` and `CITATION.cff`, move the *Unreleased* entries in the changelog
under the new version, then tag `vX.Y.Z` on `main`.
