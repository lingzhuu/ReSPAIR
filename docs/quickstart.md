# Quick start

```python
import respair
from respair.simulate import simulate

d = simulate(seed=0)
res = respair.fit(d["G"], d["M"], d["coords"], Z=d["Z"], neg_genes=d["neg_genes"])

res["z"].shape          # (n_genes, n_metabolites)
res["K"]                # hidden factors removed
hits = respair.bh(res["p"], alpha=0.1)
```

## Inputs

| Argument | Shape | Meaning |
|---|---|---|
| `G_counts` | spots × genes | raw gene counts |
| `M_int` | spots × metabolites | metabolite intensities on the **same** spots |
| `coords` | spots × 2 | spatial coordinates |
| `Z` | spots × c | known structure such as cell-type proportions (optional) |
| `neg_genes` | indices | negative-control genes for calibration (optional) |

## Outputs

`z` and `p` are gene × metabolite matrices. `z_scale` holds the same test restricted to large, medium and small
spatial scales, which helps tell local links from broad co-variation. `c` is the calibration factor (≈ 1 when the
null is well specified).
