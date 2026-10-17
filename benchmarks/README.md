# Benchmarks

Scripts that reproduce the simulation results and figures. They are not part of the installed package.

```bash
pip install -e ".[benchmarks]"           # from the repository root
python benchmarks/scripts/compare.py --seeds 5 --link 0.5  # main benchmark: published methods vs ReSPAIR
python benchmarks/scripts/compare.py --seeds 5 --link 0.2  # harder setting
python benchmarks/scripts/run_sim.py 5   # all methods, 5 seeds   -> results/sim_5seeds.json
python benchmarks/scripts/roc.py 5 0.5   # ablation ROC (supplement) -> results/ablation_roc.json
python benchmarks/scripts/sweep_sim.py 2 # confounder / link sweep -> results/sweep.json
python benchmarks/scripts/plot_sim.py    # figures                -> results/figures/
```

| Path | Content |
|---|---|
| `scripts/compare.py` | main benchmark: AUC and power at 5% FPR of published methods vs ReSPAIR |
| `scripts/run_sim.py` | false-positive rate, power and BH FDR for every method and ablation |
| `scripts/roc.py` | supplementary: ROC of ReSPAIR's ablations (each step on its own) |
| `scripts/sweep_sim.py` | sweep over confounder strength and link strength |
| `scripts/plot_sim.py` | figures 1–4 |
| `results/` | committed outputs used in the README and manuscript |

The simulator is `respair.simulate.simulate`. Baselines live in `respair.benchmarks`: the MANTIS re-implementation
(needs scikit-learn) and wrappers for LIANA+/SpatialDM, Lee's L, and the Dutilleul modified t-test. The LIANA+ wrapper imports `liana` only when called.

Only methods that test each gene–metabolite pair are benchmarked. Integration and factor models without a pairwise test
(SpaMV, MEFISTO, MISTy) answer a different question and are not included.
