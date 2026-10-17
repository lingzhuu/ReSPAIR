"""Simulated data: known tissue structure + one unlabeled confounder affecting genes and metabolites
(hypoxia-like gradient) + private hidden factors on each side + a few true pairwise links.

True links follow the model: metabolite k tracks the local residual of gene j (what remains after tissue structure
and all hidden factors are removed).
"""
import numpy as np


def gp_samples(coords, length, n_draws, rng, jitter=1e-6):
    D = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    K = np.exp(-D / (2 * length ** 2))
    L = np.linalg.cholesky(K + jitter * np.eye(len(coords)))
    return L @ rng.standard_normal((len(coords), n_draws))


def simulate(seed=0, side=50, p=2000, q=200, n_conf_genes=300, n_conf_mets=40,
             conf_strength=0.6, n_links=150, link_strength=0.5, n_neg=500,
             depth=4000, theta=10.0, link_on="local", link_fn="linear", n_extra_conf=0, ct_drift=0.0,
             misalign=0, msi_block=1, msi_dropout=0.0):
    """Simulate one data set. The defaults follow ReSPAIR's model; the remaining options break its assumptions.

    link_on       "local": links act on the gene's local residual (the model); "total": on the gene's whole
                  log-expression, including tissue structure and hidden factors
    link_fn       "linear" or "threshold": the metabolite responds only where the gene signal is above its median
    n_extra_conf  extra unlabeled confounders (each its own GP and gene/metabolite set), so confounding is not low-rank
                  in the sense of a few strong factors
    ct_drift      cell-type effects drift along one tissue axis (Z × space interaction, not removed by regressing on Z)
    misalign      metabolite map shifted by this many spots relative to genes (registration error)
    msi_block     metabolites measured at block × block lower resolution
    msi_dropout   fraction of metabolite values replaced by the detection floor (below limit of detection)
    """
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 10_000)  # misspecification draws, so the defaults reproduce earlier data
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    n = len(coords)

    # known structure: proportions of 3 cell types (softmax of smooth fields)
    logits = 2.0 * gp_samples(coords, 12, 3, rng)
    Z = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    Bg = rng.normal(0, 0.8, (3, p))
    Bm = rng.normal(0, 0.8, (3, q))

    # unlabeled shared confounder
    H = gp_samples(coords, 6, 1, rng)[:, 0]
    conf_g = rng.choice(p, n_conf_genes, replace=False)
    conf_m = rng.choice(q, n_conf_mets, replace=False)
    lam_g = np.zeros(p); lam_g[conf_g] = rng.choice([-1, 1], n_conf_genes) * rng.uniform(0.5, 1, n_conf_genes) * conf_strength
    lam_m = np.zeros(q); lam_m[conf_m] = rng.choice([-1, 1], n_conf_mets) * rng.uniform(0.5, 1, n_conf_mets) * conf_strength

    # private hidden factors: 3 for genes, 2 for metabolites
    Fg = gp_samples(coords, 8, 3, rng)
    Fm = gp_samples(coords, 8, 2, rng)
    Phi_g = rng.normal(0, 0.4, (3, p)) * (rng.random((3, p)) < 0.2)
    Phi_m = rng.normal(0, 0.4, (2, q)) * (rng.random((2, q)) < 0.3)

    # local gene residual: short-scale GP + independent noise
    local_g = 0.5 * gp_samples(coords, 1.5, p, rng) + 0.3 * rng.standard_normal((n, p))

    base = rng.normal(0, 1, p)
    # optional misspecification of the confounding (drawn from rng2)
    drift = (coords[:, 0] - coords[:, 0].mean()) / np.ptp(coords[:, 0]) * 2  # -1 … 1 along x
    extra_g, extra_m, extra_pairs = np.zeros((n, p)), np.zeros((n, q)), []
    for _ in range(n_extra_conf):
        Hk = gp_samples(coords, rng2.uniform(3, 10), 1, rng2)[:, 0]
        cg = rng2.choice(p, 150, replace=False); cm = rng2.choice(q, 20, replace=False)
        extra_g[:, cg] += np.outer(Hk, rng2.choice([-1, 1], 150) * rng2.uniform(0.5, 1, 150) * conf_strength)
        extra_m[:, cm] += np.outer(Hk, rng2.choice([-1, 1], 20) * rng2.uniform(0.5, 1, 20) * conf_strength)
        extra_pairs.append((cg, cm))
    if ct_drift:
        Dg = rng2.normal(0, 0.8, (3, p)); Dm = rng2.normal(0, 0.8, (3, q))
        extra_g += ct_drift * drift[:, None] * ((Z - Z.mean(0)) @ Dg)
        extra_m += ct_drift * drift[:, None] * ((Z - Z.mean(0)) @ Dm)

    eta = base + (Z - Z.mean(0)) @ Bg + np.outer(H, lam_g) + Fg @ Phi_g + local_g + extra_g
    mu = np.exp(eta)
    mu = mu / mu.sum(1, keepdims=True) * depth
    counts = rng.poisson(rng.gamma(theta, mu / theta))

    # true links, avoiding negative-control genes
    neg_genes = rng.choice(p, n_neg, replace=False)
    candidates = np.setdiff1d(np.arange(p), neg_genes)
    link_g = rng.choice(candidates, n_links, replace=True)
    link_m = rng.choice(q, n_links, replace=True)
    W = np.zeros((q, p))
    W[link_m, link_g] = rng.choice([-1, 1], n_links) * link_strength
    if link_on == "local":
        local_std = local_g / local_g.std(0)
    else:
        local_std = (eta - base - (eta - base).mean(0)) / (eta - base).std(0)
    if link_fn == "threshold":
        local_std = np.maximum(local_std - np.median(local_std, 0), 0)
        local_std = (local_std - local_std.mean(0)) / local_std.std(0)

    local_m = 0.5 * gp_samples(coords, 1.5, q, rng) + 0.5 * rng.standard_normal((n, q))
    logm = (Z - Z.mean(0)) @ Bm + np.outer(H, lam_m) + Fm @ Phi_m + local_std @ W.T + local_m + extra_m
    if misalign:  # shift the metabolite image, repeating the edge
        img = logm.reshape(side, side, q)
        idx = np.clip(np.arange(side) + misalign, 0, side - 1)
        logm = img[:, idx][idx].reshape(n, q)
    if msi_block > 1:  # average over blocks, then assign the block value to every spot in it
        b = msi_block
        blk = (coords[:, 1] // b) * (side // b + 1) + coords[:, 0] // b
        _, inv = np.unique(blk, return_inverse=True)
        sums = np.zeros((inv.max() + 1, q)); np.add.at(sums, inv, logm)
        logm = (sums / np.bincount(inv)[:, None])[inv]
    M = np.exp(logm + 3.0)
    if msi_dropout:
        floor = np.quantile(M, 0.01, axis=0)
        M = np.where(rng2.random(M.shape) < msi_dropout, floor, M)

    truth = np.zeros((p, q), bool)
    truth[link_g, link_m] = True
    conf_pair = np.zeros((p, q), bool)
    conf_pair[np.ix_(conf_g, conf_m)] = True
    for cg, cm in extra_pairs:
        conf_pair[np.ix_(cg, cm)] = True
    return dict(G=counts, M=M, coords=coords, Z=Z, truth=truth, conf_pair=conf_pair,
                neg_genes=neg_genes, H=H)


def _kernel_smoother(coords, bandwidth):
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_pairs(4 * bandwidth, output_type="ndarray")
    i = np.r_[pairs[:, 0], pairs[:, 1], np.arange(len(coords))]
    j = np.r_[pairs[:, 1], pairs[:, 0], np.arange(len(coords))]
    w = np.exp(-((coords[i] - coords[j]) ** 2).sum(1) / (2 * bandwidth ** 2))
    from scipy.sparse import csr_matrix
    S = csr_matrix((w, (i, j)), shape=(len(coords), len(coords)))
    return csr_matrix(S.multiply(1 / np.asarray(S.sum(1)))).tocsr()


def semi_synthetic(G_real, M_real, coords, seed=0, p=2000, q=200, n_links=150, link_strength=0.5,
                   link_on="total", n_neg=500, n_regions=8, smooth_bw=2.0, noise_bw=0.6, signal_bw=0.8,
                   conf_quantile=0.9):
    """Semi-synthetic data on a real tissue section: real genes, metabolites built from real metabolite maps.

    Genes are the real counts (top `p` by Pearson-residual variance), so their tissue structure, hidden programs,
    noise and spatial autocorrelation are all real. Each of the `q` metabolites keeps the large-scale map of a real
    metabolite (Gaussian-kernel smoothing, `smooth_bw` spots, not a low-rank fit), plus noise with the real residual
    variance and short-range autocorrelation, plus planted links to real genes. Real zeros are kept as zeros.

    link_on        "total": the metabolite follows the gene's whole log-expression (not ReSPAIR's model);
                   "local": only the gene's fine-scale part (expression minus its smoothed map; ReSPAIR's model)
    link_strength  link effect in units of the metabolite's standard deviation
    Annotated structure Z is k-means regions (`n_regions`) on gene PCs, as an analyst would have. Confounded null pairs
    are unlinked pairs whose smoothed gene and metabolite maps are most correlated (top 1 − `conf_quantile`).
    """
    from sklearn.cluster import KMeans

    from .core import pearson_residuals
    rng = np.random.default_rng(seed)
    G_real = np.asarray(G_real); M_real = np.asarray(M_real, float); coords = np.asarray(coords, float)
    n = len(coords)
    # genes: detected in ≥10% of spots, most variable Pearson residuals; fixed across seeds
    det = (G_real > 0).mean(0) >= 0.1
    R = pearson_residuals(G_real[:, det])
    gsel = np.flatnonzero(det)[np.argsort(-R.var(0))[:p]]
    G = G_real[:, gsel]
    # metabolites: detected in ≥50% of spots, most variable on the log scale
    mdet = np.flatnonzero((M_real > 0).mean(0) >= 0.5)
    pos = M_real[M_real > 0]
    Lall = np.log(M_real[:, mdet] + np.percentile(pos, 1))
    msel = mdet[np.argsort(-Lall.var(0))[:q]]
    L = np.log(M_real[:, msel] + np.percentile(pos, 1))
    S = _kernel_smoother(coords, smooth_bw)
    B = S @ L                                      # real large-scale metabolite maps
    sd_res = (L - B).std(0)
    E = _kernel_smoother(coords, noise_bw) @ rng.standard_normal((n, q))
    E = E / E.std(0) * sd_res
    # gene signal that drives the links
    # the metabolite must not copy the gene's sequencing noise, so links use a denoised expression (kernel-smoothed over
    # the spot and its immediate neighbours, `signal_bw` spots) rather than the observed counts
    lg = _kernel_smoother(coords, signal_bw) @ np.log1p(G / G.sum(1, keepdims=True) * 1e4)
    if link_on == "local":
        lg = lg - S @ lg
    sig = (lg - lg.mean(0)) / np.where(lg.std(0) > 0, lg.std(0), 1)
    neg_genes = rng.choice(p, n_neg, replace=False)
    cand = np.setdiff1d(np.arange(p), neg_genes)
    link_g = rng.choice(cand, n_links, replace=True)
    link_m = rng.choice(q, n_links, replace=True)
    W = np.zeros((q, p))
    W[link_m, link_g] = rng.choice([-1, 1], n_links) * link_strength
    logm = B + E + (sig @ W.T) * L.std(0)
    M = np.where(M_real[:, msel] > 0, np.exp(logm), 0.0)
    truth = np.zeros((p, q), bool); truth[link_g, link_m] = True
    # annotated regions
    U, s, _ = np.linalg.svd(R[:, np.argsort(-R.var(0))[:p]] - R[:, np.argsort(-R.var(0))[:p]].mean(0),
                            full_matrices=False)
    lab = KMeans(n_regions, n_init=10, random_state=0).fit_predict(U[:, :20] * s[:20])
    Z = np.eye(n_regions)[lab]
    # confounded null pairs: strongest large-scale co-variation between real gene and metabolite maps
    Sg = S @ lg if link_on == "total" else S @ np.log1p(G / G.sum(1, keepdims=True) * 1e4)
    Sg = (Sg - Sg.mean(0)) / np.where(Sg.std(0) > 0, Sg.std(0), 1)
    Bz = (B - B.mean(0)) / B.std(0)
    r = np.abs(Sg.T @ Bz / n)
    conf_pair = (r >= np.quantile(r[~truth], conf_quantile)) & ~truth
    return dict(G=G, M=M, coords=coords, Z=Z, truth=truth, conf_pair=conf_pair, neg_genes=neg_genes,
                genes_idx=gsel, mets_idx=msel)
