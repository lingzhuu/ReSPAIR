# Full estimator

`respair.full.fit_full` fits the model of {doc}`model` jointly instead of step by step:

$$
\begin{aligned}
\log \text{rate}(G) &= o + a + Z B_g^{\top} + F \Lambda_g^{\top} + E_g, \qquad G \mid \text{rate} \sim \text{Poisson} \\
M &= c + Z B_m^{\top} + F \Lambda_m^{\top} + (A E_g) W^{\top} + E_m
\end{aligned}
$$

## What differs from the simplified estimator

| Part | Simplified (`respair.fit`) | Full (`respair.full.fit_full`) |
|---|---|---|
| Gene likelihood | Analytic Pearson residuals | Poisson–log-normal; $E_g$ is the Laplace posterior mean of the latent log-scale residual |
| Hidden factors $F$ | PCA of both blocks, fixed | Continuous GP functions $F_r(s) = \Phi_r(s)\alpha_r$ (inducing points, one length scale per factor), re-estimated jointly with the loadings and $W$ |
| Link term during factor estimation | Not modelled | $(A E_g)W^{\top}$ is removed from the metabolites before $F$ is updated |
| $W$ | Not estimated | Sparse: lasso with a per-pair penalty equal to a spectral z-threshold, then least-squares refit on the support |
| Test | Marginal spectral test | Spectral test of each pair conditional on the other selected genes of the same metabolite (isozymes) |
| Uncertainty in $F$ | Ignored | Test repeated over posterior draws of $F$, combined with Rubin's rules |
| Reaction-network prior | — | Lower selection threshold for network pairs in $W$; two-groups empirical Bayes gives a posterior inclusion probability with separate prior rates for network and other pairs |
| Grids | Metabolites must be aggregated to the gene spots | Genes and metabolites keep their own coordinates; $F$ is evaluated at each, $A$ averages gene spots around each metabolite pixel |

**Why Poisson–log-normal and not negative binomial.** The NB is a gamma mixture on the natural scale; its latent
variable cannot enter the metabolite equation linearly. The Poisson–log-normal has the same mean–variance relation
(overdispersion $1/\theta \approx e^{\sigma^2} - 1$) and a Gaussian latent log residual, which is exactly the $E_g$
the model links to the metabolites.

## Estimation

1. Initialise $F$ with the simplified estimator's factors (same choice of $K$) and pick each factor's GP length
   scale by marginal likelihood on a grid.
2. Alternate, `n_iter` times:
   - genes: Laplace E-step for the latent log-rates, least-squares M-step for $a, B_g, \Lambda_g$ and the per-gene
     variance;
   - links: screen pairs with the conditional spectral test, then the thresholded lasso and refit for $W$;
   - metabolites: least squares for $c, B_m, \Lambda_m$ after removing $(A E_g)W^{\top}$;
   - factors: Gaussian update of $\alpha$ given all loadings (closed form; its precision is kept for step 3).
3. Test every pair at the posterior mean of $F$ and at `n_draws` posterior draws; calibrate with negative-control
   genes as in the simplified estimator.

Run time on the benchmark simulation (2,500 spots, 2,000 genes, 200 metabolites, 4 CPU threads): about 100 s, against
about 12 s for the simplified estimator.

## Results: full vs simplified on the same simulations

Mean over 5 seeds (benchmark simulation and its variants) or 3 seeds (scenarios from `misspec.py` and the
semi-synthetic data from `semisim.py`). Power is the detection rate of true links at a 5% false-positive rate among
all unlinked pairs. Both estimators have a realised false-positive rate of 5.0% at p < 0.05 in every scenario.
Full numbers: `benchmarks/results/full_vs_simple*.json`; figure: `benchmarks/results/figures/fig_full_vs_simple.png`.

| Scenario | Power, simplified | Power, full | AUC, simplified | AUC, full |
|---|---|---|---|---|
| *Benchmark simulation and variants (5 seeds)* | | | | |
| base (link 0.5) | 100.0% | 100.0% | 1.000 | 1.000 |
| base (link 0.2) | 96.3% | 96.9% | 0.989 | 0.991 |
| low depth (link 0.2) | 63.7% | 64.5% | 0.889 | 0.889 |
| isozyme decoys (link 0.2) | 96.0% | 96.1% | 0.991 | 0.991 |
| isozyme decoys: true links vs decoy pairs | 24.0% | 28.9% | 0.682 | 0.754 |
| shifted grid (link 0.2) | 86.9% | 92.4% | 0.963 | 0.979 |
| network prior (link 0.2) | 95.6% | 96.5% | 0.990 | 0.992 |
| network prior, ranked by posterior inclusion probability | 98.5% | 98.8% | 0.997 | 0.998 |
| link on total signal (link 0.2) | 85.1% | 90.8% | 0.956 | 0.975 |
| nonlinear link (link 0.2) | 93.1% | 94.5% | 0.984 | 0.986 |
| many weak programs (link 0.2) | 91.1% | 95.6% | 0.977 | 0.988 |
| zero-inflated metabolites (link 0.2) | 91.1% | 93.6% | 0.976 | 0.982 |
| *Misspecified scenarios from `misspec.py` (link 0.5, 3 seeds)* | | | | |
| ReSPAIR's model (default) | 100.0% | 100.0% | 1.000 | 1.000 |
| links on total expression | 98.9% | 99.6% | 0.998 | 0.999 |
| threshold links | 100.0% | 100.0% | 1.000 | 1.000 |
| many confounders | 99.8% | 100.0% | 1.000 | 1.000 |
| cell-type drift | 100.0% | 100.0% | 1.000 | 1.000 |
| misregistration (1 spot) | 88.2% | 91.3% | 0.966 | 0.974 |
| low MSI resolution (2×2) | 99.3% | 99.6% | 0.999 | 0.999 |
| MSI dropout (30%) | 95.6% | 96.2% | 0.988 | 0.992 |
| *Semi-synthetic on the SMA section, `semisim.py` (3 seeds)* | | | | |
| links on whole gene expression, strength 0.5 | 87.1% | 91.8% | 0.956 | 0.979 |
| links on whole gene expression, strength 0.3 | 69.5% | 71.3% | 0.898 | 0.916 |
| links on whole gene expression, strength 0.15 | 33.6% | 33.2% | 0.734 | 0.730 |
| links on fine-scale gene signal, strength 0.5 | 98.2% | 99.3% | 0.994 | 0.998 |

What this shows, and what it does not:

- On the benchmark simulation (ReSPAIR's own model) the two estimators are almost identical; at link strength 0.5
  both are at 100%. The full estimator is not needed there.
- The largest gains are where the data deviate from the simplified estimator's assumptions: links on the gene's
  whole expression, many weak confounding programs, metabolites on a different grid, zero-inflated metabolites.
  The gains are a few percentage points of power, not a change in kind.
- Isozymes are handled only partly. When a decoy gene's residual is correlated (ρ = 0.8) with a linked gene, the
  conditional test lowers the decoys' false-positive rate at p < 0.05 from 90% to 81%, not to 5%. Both genes are
  noisy count measurements of the same latent signal, and conditioning on one noisy proxy does not remove the
  information carried by the other (errors in variables).
- The reaction-network prior raises power from 96% to 99% in a scenario where the network contains 70% of the
  true links and 3 wrong pairs per true link. The same empirical-Bayes step gives the simplified estimator nearly the
  same gain (98.5%), so this is a property of using a prior, not of the joint fit, and it depends entirely on how
  good the network is.
- Propagating the uncertainty in $F$ changes little: with thousands of genes, $F$ is well determined. The
  between-draw share of the test variance is under 1% at the default depth and about 4% at low depth.
- The 5.0% false-positive rate is partly by construction: both estimators rescale z so that negative-control genes
  are at the nominal level.
- On the semi-synthetic data the full estimator gains about 5 points of power at link strength 0.5, 2 points at 0.3,
  and nothing at 0.15, where both detect about a third of the links.
