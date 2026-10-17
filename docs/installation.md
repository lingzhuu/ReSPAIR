# Installation

ReSPAIR needs Python ≥ 3.10 and depends only on NumPy and SciPy.

```bash
pip install git+https://github.com/lingzhuu/ReSPAIR.git
```

Optional extras:

| Extra | Installs | For |
|---|---|---|
| `benchmarks` | scikit-learn, matplotlib | simulation benchmark and MANTIS baseline |
| `test` | pytest | running the test suite |
| `doc` | Sphinx, MyST, furo | building this documentation |
| `dev` | all of the above + pre-commit, ruff | development |

```bash
git clone https://github.com/lingzhuu/ReSPAIR.git && cd ReSPAIR
pip install -e ".[dev]"
```
