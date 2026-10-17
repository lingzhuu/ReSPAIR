"""Baseline methods for the benchmark.

MANTIS follows its public code (github.com/yuhaotuo/MANTIS, v0.55, tools/_spc.py and _sampler.py), with two changes
for speed; the statistic and the construction of the null are unchanged:
- Autocorrelation-preserving permutation: the original uses simulated annealing with pairwise swaps on a networkx
  graph; here the same objective (keep x' W x equal to its original value) is vectorised on a dense weight matrix and
  swaps that reduce the gap are accepted greedily.
- One permutation per metabolite (the original default); the null is pooled over 10 bins of metabolite
  spatial autocorrelation.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Lasso

from .core import standardize


def _zscore(A):
    return standardize(np.asarray(A, dtype=float))


def gaussian_W(coords, l2, row_norm=True, zero_diag=True):
    D2 = cdist(coords, coords, "sqeuclidean")
    W = np.exp(-D2 / (2 * l2))
    if zero_diag:
        np.fill_diagonal(W, 0.0)
    if row_norm:
        W /= W.sum(1, keepdims=True)
    return W


def sampler_W(coords, l):
    """MANTIS sampler weights: 1 within distance l, Gaussian decay beyond."""
    D = cdist(coords, coords)
    W = np.where(D <= l, 1.0, np.exp(-D ** 2 / (2 * l * l)))
    np.fill_diagonal(W, 0.0)
    return W


def autocorr_preserving_shuffle(x, W, rng, max_steps=60000, rel_tol=1e-3):
    """Return a permutation of x whose x' W x is close to the original (the MANTIS null)."""
    xc = x - x.mean()
    target = xc @ W @ xc
    y = xc[rng.permutation(len(xc))]
    S = W @ y
    cur = y @ S
    tol = rel_tol * abs(target) + 1e-12
    n = len(y)
    for _ in range(max_steps):
        if abs(cur - target) < tol:
            break
        u, v = rng.integers(n, size=2)
        if u == v:
            continue
        a, b = y[u], y[v]
        # change in y'Wy after swapping u and v
        delta = 2 * (b - a) * (S[u] - S[v] - W[u, v] * (b - a))
        if abs(cur + delta - target) < abs(cur - target):
            y[u], y[v] = b, a
            S += (b - a) * W[:, u] + (a - b) * W[:, v]
            cur += delta
    return y + x.mean()


def _lasso_resid(X, P, alpha):
    R = np.empty_like(X)
    mdl = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)
    for j in range(X.shape[1]):
        y = X[:, j]
        if P is None:
            R[:, j] = y
            continue
        mdl.fit(P, y)
        R[:, j] = y - P @ mdl.coef_
    return R


def _pooled_empirical_p(real, null):
    z = np.sort(null.ravel())
    N0 = z.size
    left = np.searchsorted(z, real, side="right")
    right = N0 - np.searchsorted(z, real, side="left")
    p = 2 * np.minimum((left + 1) / (N0 + 1), (right + 1) / (N0 + 1))
    return np.minimum(p, 1.0)


def mantis(Yg, Ym, coords, CT=None, seed=0, n_bins=10, sampler_l=4 * np.sqrt(2), spc_l2=10.0):
    """MANTIS SCI (CT=None) or SPC-CT (with cell-type proportions CT). Returns a gene × metabolite p-value matrix."""
    rng = np.random.default_rng(seed)
    X0 = _zscore(Yg)
    Y0 = _zscore(Ym)
    P = None
    if CT is not None:
        P = _zscore(CT)
    Ws = sampler_W(coords, sampler_l)
    W = gaussian_W(coords, spc_l2)
    n = len(coords)

    Xs = _zscore(_lasso_resid(X0, P, 0.1))
    Ys = _zscore(_lasso_resid(Y0, P, 0.1))
    real = Xs.T @ (W @ Ys) / n

    Ynull = np.column_stack([autocorr_preserving_shuffle(Ym[:, k], Ws, rng) for k in range(Ym.shape[1])])
    Yn = _zscore(_lasso_resid(_zscore(Ynull), P, 0.06))
    null = Xs.T @ (W @ Yn) / n

    # bin metabolites by spatial autocorrelation and pool the null within bins
    Yc = Y0 - Y0.mean(0)
    M0 = np.einsum("ik,ij,jk->k", Yc, Ws, Yc)
    edges = np.unique(np.quantile(M0, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, M0, side="right") - 1, 0, len(edges) - 2)
    p = np.ones_like(real)
    for b in np.unique(bins):
        cols = bins == b
        p[:, cols] = _pooled_empirical_p(real[:, cols], null[:, cols])
    return p


def _logcpm(G):
    G = np.asarray(G, dtype=float)
    return np.log1p(G / G.sum(1, keepdims=True) * 1e4)


def liana_global(G_counts, M_int, coords, bandwidth=3.0, nz_prop=0.05):
    """Global bivariate metrics of LIANA+ (Dimitrov et al., Nat Cell Biol 2024).

    Returns {"morans": (score, p), "lee": (score, None)}, each gene × metabolite.
    The analytic p-value of Moran's R is the global test of SpatialDM (Li et al., Nat Commun 2023), built into LIANA+.
    Spatial weights come from `li.ut.spatial_neighbors`; the statistics follow `li.mt.bivariate(global_name=...)` line by
    line but in matrix form, since the official call loops over 400k pairs (identical output, about 50× faster).
    Genes expressed in fewer than nz_prop of spots are filtered as in LIANA+; their pairs get score 0 and p 1.
    """
    import anndata as ad
    import liana as li
    from scipy.stats import norm
    from .core import log_standardize
    Xg = _logcpm(G_counts)
    Xm = log_standardize(M_int)  # log with a small offset, so MSI zeros stay finite
    Xm = Xm - Xm.min() + 1e-3
    adata = ad.AnnData(np.zeros((len(coords), 1), np.float32))
    adata.obsm["spatial"] = np.asarray(coords, float)
    W = li.ut.spatial_neighbors(adata, bandwidth=bandwidth, cutoff=0.1, inplace=False)
    W = np.asarray(W.todense() if hasattr(W, "todense") else W, float)
    n = W.shape[0]
    keep = (Xg > 0).mean(0) >= nz_prop

    def zscore(X, spot_n):
        X = X - X.mean(0)
        return np.clip(X / np.sqrt((X ** 2).sum(0) / spot_n), -10, 10)

    # Moran's R (SpatialDM): global z-scoring, weights rescaled to sum to n, analytic z-test
    Wn = W * (n / W.sum())
    R = (Wn @ zscore(Xg, 1)).T @ zscore(Xm, 1)
    var = (n ** 2 * (Wn * Wn).sum() - 2 * n * (Wn @ Wn).sum() + Wn.sum() ** 2) / (n ** 2 * (n - 1) ** 2)
    P = 2 * norm.sf(np.abs(R / np.sqrt(var)))
    # Lee's L: per-spot z-scoring, spatial lag of both variables, x'W'Wy / Σ_i (Σ_j w_ij)²
    W2 = W.T @ W
    L = (W2 @ zscore(Xg, n)).T @ zscore(Xm, n) / W2.sum()
    R[~keep] = 0; P[~keep] = 1; L[~keep] = 0
    return {"morans": (R, P), "lee": (L, None)}


def modified_ttest(Xg, Xm, coords, n_classes=None):
    """Dutilleul (1993) modified t-test, vectorised after R SpatialPack::modified.ttest.

    Moran correlograms over distance classes estimate each variable's autocorrelation, giving the effective sample size
    n_eff = 1 + n² / (n + Σ_k n_k I_x(k) I_y(k)), and the correlation is tested against F(1, n_eff − 2).
    Returns (r, p), each gene × metabolite.
    """
    from scipy.stats import f as fdist
    X = _zscore(Xg); Y = _zscore(Xm)
    n = len(coords)
    D = cdist(coords, coords)
    if n_classes is None:
        n_classes = int(np.ceil(np.log2(n * (n - 1) / 2) + 1))  # Sturges, as in SpatialPack
    off = D[np.triu_indices(n, 1)]
    edges = np.linspace(0, off.max() + 1e-9, n_classes + 1)
    cls = np.digitize(D, edges[1:-1])
    np.fill_diagonal(cls, -1)
    Ig, Im, nk = [], [], []
    for k in range(n_classes):
        A = (cls == k).astype(float)
        cnt = A.sum()
        if cnt == 0:
            continue
        nk.append(cnt)
        Ig.append((X * (A @ X)).sum(0) / cnt)   # after standardising Σx² = n, so I(k) = (n/W_k) Σ x_i x_j / n
        Im.append((Y * (A @ Y)).sum(0) / cnt)
    Ig = np.array(Ig); Im = np.array(Im); nk = np.array(nk)
    trace = n + (Ig * nk[:, None]).T @ Im          # p × q: tr(Σx Σy)
    n_eff = np.clip(1 + n ** 2 / trace, 3, n)
    r = X.T @ Y / n
    F = r ** 2 * (n_eff - 2) / np.clip(1 - r ** 2, 1e-12, None)
    return r, fdist.sf(F, 1, n_eff - 2)
