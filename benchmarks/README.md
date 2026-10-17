# Benchmarks

Scripts that reproduce the simulation results and figures. They are not part of the installed package.

```bash
pip install -e ".[benchmarks]"           # from the repository root
python benchmarks/scripts/run_sim.py 5   # all methods, 5 seeds   -> results/sim_5seeds.json
python benchmarks/scripts/roc.py 5 0.5   # ROC / AUC, link 0.5    -> results/roc.json, figures/fig_ROC.png
python benchmarks/scripts/sweep_sim.py 2 # confounder / link sweep -> results/sweep.json
python benchmarks/scripts/plot_sim.py    # figures                -> results/figures/
```

| Path | Content |
|---|---|
| `scripts/run_sim.py` | false-positive rate, power and BH FDR for every method and ablation |
| `scripts/roc.py` | ROC, AUC and power at 5% FPR (all null pairs, and confounded pairs only) |
| `scripts/sweep_sim.py` | sweep over confounder strength and link strength |
| `scripts/plot_sim.py` | figures 1–4 |
| `results/` | committed outputs used in the README and manuscript |

The simulator is `respair.simulate.simulate`. Baselines live in `respair.benchmarks`: the MANTIS re-implementation
(needs scikit-learn) and wrappers for LIANA+/SpatialDM, Lee's L, the Dutilleul modified t-test, MEFISTO, MISTy and
SpaMV. Each wrapper imports its own package only when called, so install the ones you need (`liana`, `mofapy2`;
SpaMV runs in its own environment via `_spamv_runner.py`).
