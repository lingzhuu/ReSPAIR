# Benchmarks

Scripts that reproduce the simulation results and figures. They are not part of the installed package.

```bash
pip install -e ".[benchmarks]"           # from the repository root
python benchmarks/scripts/compare.py --seeds 5 --link 0.5  # main benchmark: published methods vs ReSPAIR
python benchmarks/scripts/compare.py --seeds 5 --link 0.2  # harder setting
python benchmarks/scripts/misspec.py 3 0.5                 # scenarios that break ReSPAIR's assumptions
python benchmarks/scripts/prepare_sma.py <SMA dir> aligned.npz  # align the real SMA section (Visium + MALDI)
python benchmarks/scripts/semisim.py aligned.npz 5         # semi-synthetic benchmark on the real section
python benchmarks/scripts/run_sim.py 5   # all methods, 5 seeds   -> results/sim_5seeds.json
python benchmarks/scripts/roc.py 5 0.5   # ablation ROC (supplement) -> results/ablation_roc.json
python benchmarks/scripts/sweep_sim.py 2 # confounder / link sweep -> results/sweep.json
python benchmarks/scripts/plot_sim.py    # figures                -> results/figures/
```

| Path | Content |
|---|---|
| `scripts/compare.py` | main benchmark: AUC and power at 5% FPR of published methods vs ReSPAIR |
| `scripts/misspec.py` | same benchmark when the simulation breaks ReSPAIR's assumptions (7 scenarios) |
| `scripts/prepare_sma.py` | aligns SMA sample V11L12-038_D1 (Vicari et al. 2024; copy in the MAGPIE tutorial repository) to Visium spots |
| `scripts/semisim.py` | semi-synthetic benchmark: real genes, metabolites built from real maps, planted links |
| `scripts/run_sim.py` | false-positive rate, power and BH FDR for every method and ablation |
| `scripts/roc.py` | supplementary: ROC of ReSPAIR's ablations (each step on its own) |
| `scripts/sweep_sim.py` | sweep over confounder strength and link strength |
| `scripts/plot_sim.py` | figures 1–4 |
| `scripts/simulate.py`, `scripts/baselines.py` | simulator and baseline methods (not part of the `respair` package) |
| `results/` | committed outputs used in the README and manuscript |

The simulator is `scripts/simulate.py` (`simulate`, `semi_synthetic`). Baselines live in `scripts/baselines.py`: the MANTIS re-implementation
(needs scikit-learn) and wrappers for LIANA+/SpatialDM, Lee's L, and the Dutilleul modified t-test. The LIANA+ wrapper imports `liana` only when called.

Only methods that test each gene–metabolite pair are benchmarked. Integration and factor models without a pairwise test
(SpaMV, MEFISTO, MISTy) answer a different question and are not included.
