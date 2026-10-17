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
             depth=4000, theta=10.0):
    rng = np.random.default_rng(seed)
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
    eta = base + (Z - Z.mean(0)) @ Bg + np.outer(H, lam_g) + Fg @ Phi_g + local_g
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
    local_std = local_g / local_g.std(0)

    local_m = 0.5 * gp_samples(coords, 1.5, q, rng) + 0.5 * rng.standard_normal((n, q))
    logm = (Z - Z.mean(0)) @ Bm + np.outer(H, lam_m) + Fm @ Phi_m + local_std @ W.T + local_m
    M = np.exp(logm + 3.0)

    truth = np.zeros((p, q), bool)
    truth[link_g, link_m] = True
    conf_pair = np.zeros((p, q), bool)
    conf_pair[np.ix_(conf_g, conf_m)] = True
    return dict(G=counts, M=M, coords=coords, Z=Z, truth=truth, conf_pair=conf_pair,
                neg_genes=neg_genes, H=H)
