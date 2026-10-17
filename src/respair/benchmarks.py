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


def liana_global(G_counts, M_int, coords, bandwidth=3.0, nz_prop=0.05, chunk=10):
    """Global bivariate metrics of LIANA+ (Dimitrov et al., Nat Cell Biol 2024), via its official API.

    Returns {"morans": (score, p), "lee": (score, None)}, each gene × metabolite.
    The analytic p-value of Moran's R is the global test of SpatialDM (Li et al., Nat Commun 2023), built into LIANA+.
    Genes expressed in fewer than nz_prop of spots are filtered by LIANA+; their pairs get score 0 and p 1.
    """
    import anndata as ad
    import pandas as pd
    import liana as li
    Xg = _logcpm(G_counts)
    Xm = np.log(np.asarray(M_int, float))
    p, q = Xg.shape[1], Xm.shape[1]
    gn = [f"g{j}" for j in range(p)]
    mn = [f"m{k}" for k in range(q)]
    adata = ad.AnnData(np.hstack([Xg, Xm - Xm.min() + 1e-3]).astype(np.float32),
                       var=pd.DataFrame(index=gn + mn))
    adata.obsm["spatial"] = np.asarray(coords, float)
    li.ut.spatial_neighbors(adata, bandwidth=bandwidth, cutoff=0.1)
    out = {"morans": (np.zeros((p, q)), np.ones((p, q))), "lee": (np.zeros((p, q)), None)}
    # all p×q pairs at once need tens of GB of memory, so call in chunks of metabolites
    for start in range(0, q, chunk):
        mk = mn[start:start + chunk]
        resource = pd.DataFrame({"ligand": np.repeat(gn, len(mk)), "receptor": np.tile(mk, p)})
        for name in ["morans", "lee"]:
            res = li.mt.bivariate(adata, resource=resource, local_name=None, global_name=name,
                                  n_perms=0 if name == "morans" else None, nz_prop=nz_prop,
                                  use_raw=False)
            gi = res["ligand"].str[1:].astype(int).to_numpy()
            mi = res["receptor"].str[1:].astype(int).to_numpy()
            out[name][0][gi, mi] = res[name].to_numpy()
            if name == "morans":
                out[name][1][gi, mi] = res[f"{name}_pvals"].to_numpy()
    return out


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


def mefisto(G_counts, M_int, coords, K=10, seed=0, frac_inducing=0.2, iters=1000):
    """MEFISTO (Velten et al., Nat Methods 2022; mofapy2) with two views (genes, metabolites) and spatial coordinates as covariates.

    MEFISTO has no pairwise test; following the usual reading of multi-omics factor models, the score for gene j and metabolite k is
    Σ_r w_g,jr w_m,kr (product of shared-factor loadings). Returns a gene × metabolite score matrix.
    """
    from mofapy2.run.entry_point import entry_point
    Xg = _zscore(_logcpm(G_counts))
    Xm = _zscore(np.log(np.asarray(M_int, float)))
    ent = entry_point()
    ent.set_data_options(scale_views=True)
    ent.set_data_matrix([[Xg], [Xm]], views_names=["gene", "metabolite"])
    ent.set_model_options(factors=K)
    ent.set_train_options(iter=iters, seed=seed, quiet=True, verbose=False)
    ent.set_covariates([np.asarray(coords, float)], covariates_names=["x", "y"])
    ent.set_smooth_options(sparseGP=True, frac_inducing=frac_inducing, scale_cov=False)
    ent.build()
    ent.run()
    W = ent.model.nodes["W"].getExpectation()
    Wg, Wm = W[0], W[1]
    return Wg @ Wm.T


def misty(G_counts, M_int, coords, bandwidth=3.0):
    """MISTy (Tanevski et al., Genome Biol 2022), Python implementation from LIANA+.

    Each metabolite is a target and genes are predictors, with juxta (nearest neighbours) and para (kernel-weighted) views;
    set_diag=True includes the spot itself. Linear model. The pair score is the larger absolute importance across views.
    Returns a gene × metabolite score matrix.
    """
    import anndata as ad
    import pandas as pd
    from liana.method import genericMistyData
    from liana.method.sp import LinearModel
    Xg = _logcpm(G_counts)
    Xm = np.log(np.asarray(M_int, float))
    p, q = Xg.shape[1], Xm.shape[1]
    obs = pd.DataFrame(index=[f"s{i}" for i in range(len(coords))])
    rna = ad.AnnData(Xg.astype(np.float32), obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(p)]))
    msi = ad.AnnData(Xm.astype(np.float32), obs=obs, var=pd.DataFrame(index=[f"m{k}" for k in range(q)]))
    for a in (rna, msi):
        a.obsm["spatial"] = np.asarray(coords, float)
    mdata = genericMistyData(intra=msi, extra=rna, bandwidth=bandwidth, cutoff=0.1, add_juxta=True,
                             add_para=True, set_diag=True, nz_threshold=0)
    mdata(model=LinearModel, bypass_intra=True, verbose=False)
    inter = mdata.uns["interactions"]
    S = np.zeros((p, q))
    gi = inter["predictor"].str[1:].astype(int).to_numpy()
    mi = inter["target"].str[1:].astype(int).to_numpy()
    imp = np.abs(inter["importances"].fillna(0).to_numpy())
    np.maximum.at(S, (gi, mi), imp)
    return S


def spamv(G_counts, M_int, coords, python, epochs=30, zs_dim=10, seed=0, workdir=None):
    """SpaMV (Nat Commun 2025). The pair score is the product of shared-topic loadings Σ_k β_RNA[g,k] β_MSI[m,k].

    SpaMV needs torch and lives in a separate environment; it is called in a subprocess (python = that interpreter).
    The default 400+400 epochs take hours on CPU, so the benchmark trains for `epochs` in each of the two stages.
    """
    import subprocess, tempfile, os
    workdir = workdir or tempfile.mkdtemp()
    inp, out = os.path.join(workdir, "in.npz"), os.path.join(workdir, "out.npy")
    np.savez(inp, G=np.asarray(G_counts, np.float32), M=np.log(np.asarray(M_int, float)).astype(np.float32),
             coords=np.asarray(coords, float))
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_spamv_runner.py")
    subprocess.run([python, runner, inp, out, str(epochs), str(zs_dim), str(seed)], check=True)
    return np.load(out)
