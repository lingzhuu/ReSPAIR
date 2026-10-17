# Model and design rationale

ReSPAIR models low-rank confounding plus sparse pairwise links. This page explains the model, how it differs from the original joint model it replaced, and the design choices behind the estimator.

## The core change (in one sentence)

Original model: shared latent patterns = links.
New model: **every latent pattern (shared or private) is treated as tissue structure / confounding; a link is the sparse pairwise relationship that remains between gene and metabolite residuals after all of these have been removed.**

By analogy: first remove every area effect that shapes the distribution of many amenities at once — "wealthy district", "school district", "old town" — whether or not it is labelled. Then, within the same area, ask whether the excess at **this particular** tennis court corresponds to the excess at **this particular** supermarket.

---

## 1. Model

$$
\begin{aligned}
G &= Z B_g^{\top} + F \Lambda_g^{\top} + E_g \\
M &= Z B_m^{\top} + F \Lambda_m^{\top} + E_g W^{\top} + E_m
\end{aligned}
$$

| Symbol | Meaning | Difference from the original model |
|---|---|---|
| $Z$, $B_g$, $B_m$ | Known tissue structure and its effects | Unchanged |
| $F$ ($n \times K$) | **All** latent spatial patterns; $K$ is typically 10–50, each column a Gaussian process with its own length scale $\ell_r$ | No longer split into shared/private; all treated as confounding |
| $\Lambda_g$, $\Lambda_m$ | Loadings of each molecule on each latent pattern | Used only for removal, no longer to define links |
| $E_g$ | The gene's "local residual": after removing known structure and all latent patterns, how much higher or lower than expected the gene is at this spot | New key quantity |
| $W$ (metabolites $\times$ genes) | **Sparse pairwise link matrix.** $W_{kj} \neq 0$ means that wherever gene $j$ is higher than expected, metabolite $k$ is also higher (or lower) than expected there | New; this is the object being tested |
| $E_m$ | Residual noise of the metabolites | Unchanged |

The model implies the following covariance between gene $j$ and metabolite $k$ at the same location:

$$
\mathrm{Cov}(g_j, m_k) = \underbrace{\textstyle\sum_r \lambda_{g,jr}\lambda_{m,kr}}_{\text{low rank: confounding / broad programs}} \;+\; \underbrace{\sigma^2_{g,j}\, W_{kj}}_{\text{sparse: pairwise link}}
$$

The full link table is therefore $C = \Lambda_g \Lambda_m^{\top} + W D_g$, where $D_g$ is the diagonal matrix of gene residual variances. The low-rank part is removed, and only individual entries of $W$ are tested.

**Why the metabolites depend on the gene residuals, rather than using a symmetric form:** biologically, enzymes set reaction rates and thereby affect metabolites, so this direction is natural. Moreover, in this form $W$ appears as a strictly sparse block of the covariance, which makes it easy both to estimate and to interpret. A symmetric version can be added later if needed.

## 2. Identifiability: what separates confounding from links

This is the point the new model must explain clearly, and the one reviewers will certainly ask about. There is only one basis for the distinction:

> **Confounding is "pervasive"; links are "sparse".** A latent factor (hypoxia, vasculature, an unannotated region) simultaneously affects hundreds to thousands of genes and many metabolites, so it can be reliably estimated from 30,000 genes. A single enzyme–metabolite link involves one gene and a handful of metabolites, which is not enough to form a latent pattern.

Statistically, this corresponds to the identifiability conditions for "low-rank + sparse" decompositions (Chandrasekaran et al. 2011/2012) and to the theory of confounder adjustment in genomics (CATE: Wang, Zhao, Hastie, Owen, Ann. Stat. 2017; FarmTest: Fan et al., JASA 2019). A large number of genes ($p \approx$ 20,000–30,000, far greater than $K$) is exactly the regime most favourable to such methods.

**The cost that must be stated honestly in the proposal:** if a genuine biochemical relationship is itself "pervasive" (for example, HIF driving the whole set of glycolytic genes up together with lactate), it will be assigned to the low-rank part and removed as a program. What the new model finds are links that are "**pair-specific, beyond the overall programs**". This framing is narrower than before, but more defensible, and closer to the original intent of "local chemical reactions".

## 3. Estimation

There are two versions, corresponding to point 5 of the proposal ("simple version first, then the full version").

### Simplified version (do first; can be running within days)

1. **Fit the confounding model**: concatenate $G$ and $M$ and fit `[G, M] = Z B + F Λ + residual`. Counts (genes) can initially be handled with GLM-PCA or a negative binomial factor model, with metabolites log-transformed; spatial smoothing can be omitted at first, or an off-the-shelf MEFISTO / NSF can be used to obtain a GP-based $F$.
2. **Extract residuals**: $\hat E_g$ ($n \times$ 30,000) and $\hat E_m$ ($n \times$ 1,000), using Pearson residuals.
3. **Obtain the statistics for all pairs with a single matrix multiplication**: $T = \hat E_g^{\top} \hat E_m$ (30,000 $\times$ 1,000).
4. **Apply spatial calibration to each pair** (see Section 4) to obtain p-values.

Because the gene residuals are essentially uncorrelated after confounding has been removed ($D_g$ approximately diagonal), the "marginal" residual correlation here is equivalent to testing whether the corresponding entry of $W$ is zero; there is no need to run a 30,000-dimensional regression for each metabolite. The original proposal's advantage of "fit once, compute all combinations together" is retained.

### Full version (later)

Estimate $F$, $\Lambda$ and $W$ jointly in a single model: a sparse / inducing-point GP for $F$, an L1 or spike-and-slab prior on $W$, and a negative binomial likelihood for the genes. The benefits are that the estimation uncertainty in $F$ propagates to $W$, and that reaction-network priors can be used (Section 5).

## 4. Significance testing: accounting for spatial autocorrelation on both sides

Even after the latent patterns are removed, the residuals still contain small-scale spatial autocorrelation. Two residuals that are both "patchy" will show inflated sample correlation even when entirely unrelated. The approach is a decomposition over spatial frequency:

1. Choose a spatial basis $U$ (a 2D DCT/Fourier basis for regular grids, graph Laplacian eigenvectors for irregular spots), ordered from low to high spatial frequency.
2. Project the residuals onto it: $\hat g_j = U^{\top} \hat e_{g,j}$, $\hat m_k = U^{\top} \hat e_{m,k}$.
3. The statistic is $T_{jk} = \sum_l \hat g_{jl} \hat m_{kl}$. Under the null hypothesis that the two sides are independent, the frequency components are approximately independent, so

$$
\mathrm{Var}_0(T_{jk}) \approx \sum_l s_{g,jl}\, s_{m,kl}
$$

   where $s$ is each side's power near frequency $l$ (estimated by smoothing within frequency bands).
4. Compute $z_{jk} = T_{jk} / \sqrt{\mathrm{Var}_0}$ and convert to a p-value.

This variance is also a single matrix multiplication ($S_g S_m^{\top}$), so 30 million pairs remain fast. This is what point 4 of the proposal means by "accounting for the spatial patterns of both genes and metabolites"; in essence it is a multivariate, batched version of Dutilleul's (1993) modified t-test.

**I ran a small simulation to check calibration** (40 $\times$ 40 grid, two mutually independent spatial fields with length scales 3 and 5, 500 replicates, nominal level 5%):

| Method | Actual false positive rate |
|---|---|
| Plain correlation | 69% |
| Frequency correction, coarse bands (about 80 components per band) | 15% |
| Frequency correction, fine bands (about 10 components per band) | 4.8% |

So the method works, but **the bands must be narrow**, especially at low frequencies; otherwise the test is anti-conservative. In the final method this should be treated as a setting to be determined by simulation.

**The multi-scale question is solved along the way:** decomposing $T_{jk}$ into contributions from different frequency bands directly answers "at which spatial scale does this pair's link occur". The smallest identifiable scale is set by the platform's resolution (about 100 µm spot spacing for Visium).

## 5. Choosing the number of latent patterns K: negative controls, not a test of K_s = 0

The global test $H_0: K_s = 0$ in the original proposal is a non-regular test on the boundary of the parameter space, and is no longer needed. $K$ is instead chosen with negative controls (in the spirit of RUV):

- **Negative-control pairs**: gene–metabolite pairs for which a direct biochemical relationship is very unlikely, for example non-metabolic genes (ribosomal proteins, splicing factors, etc.) paired with any metabolite, or reaction-network pairs after shuffling the gene–metabolite correspondence.
- **Criterion for choosing K**: increase $K$ step by step until the p-values of the negative-control pairs are approximately uniform. If $K$ is too small, confounding is not fully removed and the negative controls show many false positives; if, as $K$ keeps increasing, the signal of positive controls (known enzyme–metabolite pairs) starts to drop, the model has begun absorbing genuine links. The balance point between the two is $K$.
- **The global test** becomes: among candidate pairs, is the proportion of significant pairs significantly higher than among negative controls?

**Candidate pairs** (the part of $W$ that receives focused testing) come from reaction networks: for each metabolite, the genes of its catalysing enzymes and transporters in Recon3D / Mouse1 / Rhea. Genome-wide pairs can also be tested, but applying multiple-testing correction to the candidate pairs separately gives much higher power.

## 6. Multiple testing

After confounding is removed, the correlation among pair statistics comes mainly from residual correlation among gene residuals, which is much weaker than in the original proposal (where the whole table was determined by $K_s$ patterns), so the BH procedure is broadly applicable. A single "null-model parametric bootstrap" check is still recommended: simulate data from the fitted confounding model with $W = 0$, run the full pipeline, and confirm that the FDR is at the nominal level.

## 7. What the revision fixes, relative to the original assessment

| Issue raised in the original assessment | In the new model |
|---|---|
| 1. Unannotated shared confounding is counted as links | All latent patterns are removed; links are sought only in the residuals ✔ |
| 2. A low-rank table can only express "belonging to the same program" | Links live in the sparse $W$ and are represented pair by pair ✔ |
| 3. Little difference from MEFISTO | MEFISTO-type methods are only the first step; the core contribution becomes the pairwise test and calibration on the residuals ✔ |
| 4. Non-regular test on a product of parameters | The test is on an individual $W_{kj}$, which is regular ✔ |
| 5. Boundary test of $K_s = 0$ | Replaced by choosing $K$ with negative controls ✔ |
| 6. 30 million highly correlated tests | Correlation is greatly reduced, and checked with a parametric bootstrap ✔ |
| 7. Two-sided spatial test not designed | Frequency-decomposition correction, validated in a small simulation ✔ |
| 9. Orthogonality constraint only blocks linear relationships | $Z^{\top} F_s = 0$ is no longer needed; nonlinear tissue effects are absorbed by the latent patterns ✔ |
| 10. Uncertainty in / circularity of $Z$ | Errors in estimating $Z$ are absorbed by the latent patterns; $Z$ is used mainly for interpretation (partly ✔) |
| 11. Gaussian noise | Use negative binomial / Pearson residuals (to be implemented) |
| 12. Different grids on the two sides | MSI pixels still need to be aggregated to Visium spots first (unresolved; in the full version a GP can be observed at each modality's own coordinates) |
| 13. Multi-scale analysis limited by resolution | Decomposition by frequency band, with the lower scale limit stated according to resolution ✔ |
| 8. Circular simulation design | Still to be designed; see below |

## 8. How the simulation study should change

Core scenario: generate an **unannotated factor that simultaneously affects a set of genes and a set of metabolites** (e.g. a hypoxia gradient), then plant a small number of genuine pairwise links. Compare:

- plain correlation, MANTIS, **the original approach (shared patterns = links)**, and the new approach;
- metrics: detection rate for the true pairs, and the false positive rate for pairs affected by the hypoxia factor but with no pairwise link.

The original approach and MANTIS are expected to produce many false positives driven by the hypoxia factor, while the new approach should not. Including the original approach in the comparison also lets the paper show "why shared patterns cannot be treated as links", which is itself a contribution.

Also add: count noise, nonlinear cell-type effects, ion suppression (a region-dependent multiplicative factor affecting only the metabolites), and registration error in spot positions.

## 9. Revised risk table for the proposal

| Risk | Mitigation |
|---|---|
| Genuine but "pervasive" biochemical programs are removed as confounding | State the scope explicitly as "pair-specific links beyond programs"; monitor the choice of $K$ with positive controls |
| Confounding that affects only a few molecules cannot be distinguished from genuine links | This is the method's fundamental assumption; use simulation to assess how low the "pervasiveness" of confounding can be before the method fails |
| Poorly chosen frequency bands in the spatial correction make the test anti-conservative | Calibrate band width with a null-model bootstrap |
