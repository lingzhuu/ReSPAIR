# Quick start

```python
import respair
import sys
sys.path.insert(0, "benchmarks/scripts")      # simulated example data; run from the repository root
from simulate import simulate

d = simulate(seed=0)
res = respair.fit(d["G"], d["M"], d["coords"], Z=d["Z"], neg_genes=d["neg_genes"])

res["z"].shape          # (n_genes, n_metabolites)
res["K"]                # hidden factors removed
res["W"]                # sparse link estimate
hits = respair.bh(res["p"], alpha=0.1)
```

## Inputs

| Argument | Shape | Meaning |
|---|---|---|
| `G_counts` | spots × genes | raw gene counts |
| `M_int` | pixels × metabolites | metabolite intensities |
| `coords` | spots × 2 | gene coordinates |
| `coords_m` | pixels × 2 | metabolite coordinates (optional; default: the gene spots) |
| `Z` | spots × c | known structure such as cell-type proportions (optional) |
| `neg_genes` | indices | negative-control genes for calibration (optional) |
| `prior_pairs` | genes × metabolites | boolean reaction-network prior (optional) |

## Outputs

`z` and `p` are gene × metabolite matrices; `pip` is the posterior inclusion probability of each pair and `W` the
sparse link estimate. `c` is the calibration factor (≈ 1 when the null is well specified).
