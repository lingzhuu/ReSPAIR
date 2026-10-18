"""Building blocks of the ReSPAIR estimator (``respair.fit``): normalisation, spatial frequency basis, initial hidden
factors, band power spectra and calibration.

K (number of hidden factors) comes from the singular-value spectrum: count singular values more than 10% above the
"plateau" (median of singular values 15–25) and add 2. Removing a few extra factors costs little (sparse pairwise
links are not absorbed by broad patterns), while removing too few leaves confounding, so err on the high side.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import laplacian
from scipy.spatial import cKDTree
from scipy.stats import norm

# ---------- normalisation ----------

def pearson_residuals(counts, theta=100.0):
    """Analytic Pearson residuals (Lause et al. 2021). counts: n × p."""
    counts = np.asarray(counts, dtype=float)
    n = counts.shape[0]
    total = counts.sum()
    mu = np.outer(counts.sum(1), counts.sum(0)) / total
    r = (counts - mu) / np.sqrt(mu + mu ** 2 / theta)
    clip = np.sqrt(n)
    return np.clip(r, -clip, clip)


def log_standardize(x, eps=None):
    x = np.asarray(x, dtype=float)
    if eps is None:
        pos = x[x > 0]
        eps = np.percentile(pos, 1) if pos.size else 1.0
    return np.log(x + eps)


def standardize(X):
    X = X - X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return X / sd


# ---------- spatial frequency basis ----------

def spatial_basis(coords, k=8):
    """Eigenvectors of the kNN-graph Laplacian, ordered from low to high frequency, without the constant one."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    A = sparse.coo_matrix((np.ones_like(rows, dtype=float), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    L = laplacian(A, normed=True).toarray()
    evals, U = np.linalg.eigh(L)
    return evals[1:], U[:, 1:]


# ---------- confounder removal ----------

def _residualize(X, C):
    if C is None or C.shape[1] == 0:
        return X - X.mean(0)
    C1 = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(C1, X, rcond=None)
    return X - C1 @ beta


def select_K(Rg, Rm, buffer=2, ratio=1.1):
    """Choose K from the singular-value spectrum of both blocks, weighted and concatenated."""
    Xg = standardize(Rg) / np.sqrt(Rg.shape[1])
    Xm = standardize(Rm) / np.sqrt(Rm.shape[1])
    s = np.linalg.svd(np.hstack([Xg, Xm]), compute_uv=False)
    plateau = np.median(s[15:25])
    return int((s > ratio * plateau).sum()) + buffer


def hidden_factors(Rg, Rm, K):
    """Weighted PCA of the concatenated blocks, each with equal total variance. Returns n × K factor scores."""
    if K == 0:
        return np.zeros((Rg.shape[0], 0))
    Xg = standardize(Rg) / np.sqrt(Rg.shape[1])
    Xm = standardize(Rm) / np.sqrt(Rm.shape[1])
    X = np.hstack([Xg, Xm])
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :K] * s[:K]


# ---------- test ----------

def band_power(Y_hat, band_size):
    """Y_hat: L × p (frequency components × features). Returns the local mean power of each component, shaped like Y_hat."""
    L = Y_hat.shape[0]
    P = np.empty_like(Y_hat)
    for start in range(0, L, band_size):
        sl = slice(start, min(start + band_size, L))
        P[sl] = (Y_hat[sl] ** 2).mean(0, keepdims=True)
    return P


def spectral_z(Eg, Em, U, band_size=10, scales=None):
    """Return z (p × q). scales: list of (start, stop) frequency ranges; if given, z per scale is returned too."""
    G_hat = U.T @ standardize(Eg)
    M_hat = U.T @ standardize(Em)
    T = G_hat.T @ M_hat
    Pg = band_power(G_hat, band_size)
    Pm = band_power(M_hat, band_size)
    V = Pg.T @ Pm
    z = T / np.sqrt(V)
    out = {"z": z}
    if scales is not None:
        out["z_scale"] = []
        for a, b in scales:
            Ts = G_hat[a:b].T @ M_hat[a:b]
            Vs = Pg[a:b].T @ Pm[a:b]
            out["z_scale"].append(Ts / np.sqrt(Vs))
    return out


def naive_z(Eg, Em):
    """z of plain Pearson correlation (Fisher transform), no spatial correction."""
    n = Eg.shape[0]
    r = standardize(Eg).T @ standardize(Em) / n
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r) * np.sqrt(n - 3)


def pvals(z):
    return 2 * norm.sf(np.abs(z))


def calibration_factor(z_neg, q=0.95):
    """Ratio of the q quantile of negative-control ``|z|`` to the matching standard-normal quantile."""
    return np.quantile(np.abs(z_neg), q) / norm.isf((1 - q) / 2)


def lambda_gc(z):
    """Genomic inflation factor: median z² / median of χ²₁."""
    return np.median(z ** 2) / 0.4549364


def bh(p, alpha=0.1):
    """Benjamini–Hochberg; returns a boolean array."""
    flat = p.ravel()
    m = flat.size
    order = np.argsort(flat)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = flat[order] <= thresh
    k = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej.reshape(p.shape)
