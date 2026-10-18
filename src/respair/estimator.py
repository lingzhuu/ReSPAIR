"""ReSPAIR estimator: joint fit of the low-rank + sparse model.

    log-rate(G) = o + a + Z B_g + F Λ_g + E_g,     G | log-rate ~ Poisson       (Poisson–log-normal)
    M           = c + Z B_m + F Λ_m + (A E_g) Wᵀ + E_m

Estimation and testing:

- Genes use a count likelihood. The Poisson–log-normal model is the log-normal counterpart of the negative binomial
  (same mean–variance relation, overdispersion 1/θ ≈ exp(σ²) − 1). It is used instead of the gamma-mixed NB because
  its latent log-scale residual E_g is Gaussian and can enter the metabolite equation linearly; E_g is the Laplace
  posterior mean.
- F is a continuous spatial function, F_r(s) = Φ_r(s) α_r with an inducing-point GP basis and a length scale per
  factor, so it can be evaluated at the gene spots and at the metabolite pixels separately (different grids).
  A maps gene spots to metabolite pixels for the link term.
- F, the loadings and W are estimated together by alternating updates; the link term is removed from the
  metabolites before F is re-estimated, so F does not absorb link signal.
- W has a sparse prior: a weighted lasso whose per-pair penalty equals a z-threshold on the spectral null variance
  (a coefficient is non-zero only if its conditional spectral z exceeds the threshold), followed by a least-squares
  refit on the support. A reaction-network prior lowers the threshold for network pairs.
- Each pair is tested conditionally on the other selected genes of the same metabolite, which separates a true
  enzyme from a correlated isozyme.
- The posterior uncertainty of F is propagated by repeating the test over posterior draws of F and combining
  them with Rubin's rules.
- With a reaction-network prior, a two-groups empirical-Bayes model on z returns a posterior inclusion probability
  per pair with a separate prior inclusion rate for network and non-network pairs.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm

from . import core

# ---------- GP basis for continuous factors ----------


def _rbf(a, b, length):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * length ** 2))


class GPBasis:
    """Whitened inducing-point basis Φ(s) = k(s, U) K_UU^{-1/2}, so that F = Φ α with α ~ N(0, I) is a GP draw."""

    def __init__(self, coords, length, max_inducing=300):
        lo, hi = coords.min(0), coords.max(0)
        step = 0.8 * length
        while True:
            axes = [np.arange(lo[d], hi[d] + step / 2, step) for d in range(coords.shape[1])]
            if np.prod([len(a) for a in axes]) <= max_inducing:
                break
            step *= 1.1
        self.U = np.column_stack([g.ravel() for g in np.meshgrid(*axes)])
        self.length = length
        s, V = np.linalg.eigh(_rbf(self.U, self.U, length))
        keep = s > 1e-6 * s.max()
        self.T = V[:, keep] / np.sqrt(s[keep])

    def __call__(self, coords):
        return _rbf(coords, self.U, self.length) @ self.T


def _log_evidence(f, Phi, noise):
    """log N(f | 0, (1 − noise) ΦΦᵀ + noise I), via Woodbury."""
    n, m = Phi.shape
    a, b = 1 - noise, noise
    A = np.eye(m) * (b / a) + Phi.T @ Phi
    L = np.linalg.cholesky(A)
    u = np.linalg.solve(L, Phi.T @ f)
    quad = (f @ f - u @ u) / b
    logdet = (n - m) * np.log(b) + m * np.log(a) + 2 * np.log(np.diag(L)).sum()
    return -0.5 * (quad + logdet)


def choose_length(f, coords, grid=(2, 3, 4, 6, 8, 12, 16), noise_grid=(0.02, 0.1, 0.3)):
    """Length scale for one factor by maximum marginal likelihood over a grid."""
    f = (f - f.mean()) / f.std()
    best, best_l = -np.inf, grid[-1]
    for l in grid:
        Phi = GPBasis(coords, l)(coords)
        for nz in noise_grid:
            ev = _log_evidence(f, Phi, nz)
            if ev > best:
                best, best_l = ev, l
    return best_l


# ---------- gene count model ----------


def _pln_estep(Y, prior_mean, s2, x0, n_iter=30):
    """Laplace posterior of the latent log-rate x: y ~ Poisson(e^x), x ~ N(prior_mean, s2). Returns mode, variance."""
    x = x0.copy()
    for _ in range(n_iter):
        ex = np.exp(np.minimum(x, 30))
        step = (Y - ex - (x - prior_mean) / s2) / (ex + 1 / s2)
        x += np.clip(step, -3, 3)
        if np.abs(step).max() < 1e-5:
            break
    return x, 1 / (np.exp(np.minimum(x, 30)) + 1 / s2)


def _ols(D, X):
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)
    return beta


def _design(Z, F):
    cols = [np.ones((len(F), 1))]
    if Z is not None:
        cols.append(Z - Z.mean(0))
    cols.append(F)
    return np.hstack(cols)


# ---------- spatial maps between grids ----------


def kernel_map(src, dst, bandwidth=None):
    """Row-normalised Gaussian-kernel weights (n_dst × n_src) that average source locations around each target.

    Returns ``None`` when both sets of coordinates are identical (same grid).
    """
    if src.shape == dst.shape and np.allclose(src, dst):
        return None
    tree = cKDTree(src)
    if bandwidth is None:
        d, _ = tree.query(src, 2)
        bandwidth = np.median(d[:, 1])
    k = min(9, len(src))
    d, idx = tree.query(dst, k)
    w = np.exp(-(d ** 2) / (2 * (0.6 * bandwidth) ** 2))
    w /= w.sum(1, keepdims=True)
    A = np.zeros((len(dst), len(src)))
    np.put_along_axis(A, idx, w, axis=1)
    return A


def _apply(A, X):
    return X if A is None else A @ X


# ---------- sparse W and conditional test ----------


def _sparse_W(Xh, Rh, Pg, z_thr, screen, n_sweeps=30):
    """Weighted lasso per metabolite with penalty z_thr·sqrt(Var0), then least-squares refit on the support.

    Xh, Rh: spectral coefficients (L × p, L × q) of the link regressors and metabolite residuals.
    Pg: band power of Xh. z_thr: p × q per-pair thresholds. screen: p × q candidate mask.
    """
    p, q = Xh.shape[1], Rh.shape[1]
    W = np.zeros((q, p))
    xx = (Xh ** 2).sum(0)
    band = core.band_power
    for k in range(q):
        C = np.nonzero(screen[:, k])[0]
        if C.size == 0:
            continue
        r = Rh[:, k].copy()
        Pr = band(r[:, None], 10)[:, 0]
        lam = z_thr[C, k] * np.sqrt(Pg[:, C].T @ Pr)
        XC = Xh[:, C]
        b = np.zeros(C.size)
        for _ in range(n_sweeps):
            moved = 0.0
            for i in range(C.size):
                rho = XC[:, i] @ r + xx[C[i]] * b[i]
                new = np.sign(rho) * max(abs(rho) - lam[i], 0) / xx[C[i]]
                if new != b[i]:
                    r -= XC[:, i] * (new - b[i])
                    moved = max(moved, abs(new - b[i]))
                    b[i] = new
            if moved < 1e-6:
                break
        S = C[b != 0]
        if S.size:
            W[k, S] = np.linalg.lstsq(Xh[:, S], Rh[:, k], rcond=None)[0]
    return W


def conditional_test(Xh, Rh, W, band_size=10, corr_tol=0.1):
    """Spectral score test of every pair, conditional on the other selected genes of the same metabolite.

    T_jk = x̃_jᵀ r̃_k with x̃_j the link regressor of gene j residualised on the selected genes S_k \\ {j} and
    r̃_k the metabolite residual after removing those genes. Var0 = Σ_l P(x̃_j)_l P(r_k)_l. Returns T and V (p × q).
    """
    Pg = core.band_power(Xh, band_size)
    R_full = Rh - Xh @ W.T                          # all selected links removed
    Pr = core.band_power(R_full, band_size)
    T = Xh.T @ R_full                               # = conditional score for genes outside the support
    V = Pg.T @ Pr
    norms = np.sqrt((Xh ** 2).sum(0))
    for k in np.nonzero((W != 0).any(1))[0]:
        S = np.nonzero(W[k])[0]
        XS = Xh[:, S]
        # genes outside S that are correlated with a selected gene: residualise them on S
        corr = np.abs(Xh.T @ XS) / np.outer(norms, norms[S])
        corr[S] = 0
        J = np.nonzero(corr.max(1) > corr_tol)[0]
        if J.size:
            Xt = Xh[:, J] - XS @ np.linalg.lstsq(XS, Xh[:, J], rcond=None)[0]
            T[J, k] = Xt.T @ R_full[:, k]
            V[J, k] = core.band_power(Xt, band_size).T @ Pr[:, k]
        # selected genes: test each against the others in S
        for i, j in enumerate(S):
            O = np.delete(S, i)
            xj, rk = Xh[:, j], Rh[:, k]
            if O.size:
                XO = Xh[:, O]
                xj = xj - XO @ np.linalg.lstsq(XO, xj, rcond=None)[0]
                rk = rk - XO @ np.linalg.lstsq(XO, rk, rcond=None)[0]
            T[j, k] = xj @ rk
            V[j, k] = core.band_power(xj[:, None], band_size)[:, 0] @ Pr[:, k]
    return T, V


def two_groups_pip(z, prior=None, n_iter=200):
    """Empirical-Bayes spike-and-slab on z: z ~ (1 − π) N(0, 1) + π N(0, 1 + τ²), π set separately for prior pairs."""
    groups = np.zeros(z.shape, int) if prior is None else prior.astype(int)
    pi = np.array([0.01, 0.1])
    tau2 = 9.0
    for _ in range(n_iter):
        f0 = norm.pdf(z)
        f1 = norm.pdf(z, scale=np.sqrt(1 + tau2))
        num = pi[groups] * f1
        gam = num / (num + (1 - pi[groups]) * f0)
        new_pi = np.array([gam[groups == g].mean() if (groups == g).any() else pi[g] for g in (0, 1)])
        new_pi = np.clip(new_pi, 1e-6, 1 - 1e-6)
        tau2 = max((gam * z ** 2).sum() / gam.sum() - 1, 0.5)
        if np.abs(new_pi - pi).max() < 1e-7:
            pi = new_pi
            break
        pi = new_pi
    return gam, {"pi": pi.tolist(), "tau2": float(tau2)}


# ---------- main entry point ----------


def fit(G_counts, M_int, coords, coords_m=None, Z=None, Z_m=None, K=None, neg_genes=None,
             prior_pairs=None, z_select=4.0, z_select_prior=2.5, n_iter=8, n_draws=4,
             band_size=10, knn=8, length_grid=(2, 3, 4, 6, 8, 12, 16), calibrate=True, seed=0, verbose=False):
    """Test every gene–metabolite pair for a pair-specific spatial link (joint low-rank + sparse fit).

    Parameters
    ----------
    G_counts
        ``n_g × p`` gene counts at the gene spots ``coords``.
    M_int
        ``n_m × q`` metabolite intensities at ``coords_m`` (defaults to ``coords``: same grid).
    Z, Z_m
        Known tissue structure at the gene spots and at the metabolite pixels. ``Z_m`` defaults to ``Z`` mapped
        to the metabolite grid.
    K
        Number of hidden factors; ``None`` uses the singular-value spectrum rule of ``respair.core.select_K``.
    neg_genes
        Negative-control genes for the empirical calibration of z.
    prior_pairs
        Optional ``p × q`` boolean reaction-network prior. Network pairs get the lower selection threshold
        ``z_select_prior`` in W and their own prior inclusion rate in the posterior inclusion probability.
    z_select, z_select_prior
        Spectral z-thresholds that define the lasso penalty on W.
    n_iter
        Number of alternating updates.
    n_draws
        Posterior draws of F used to propagate its uncertainty into the test (0 uses the posterior mean only).

    Returns
    -------
    dict
        ``z``, ``p`` (``p × q``), ``pip`` (posterior inclusion probability), ``W`` (sparse estimate), ``F_g``,
        ``F_m``, ``K``, ``lengths``, calibration factor ``c``, ``Eg`` (posterior-mean gene residuals),
        ``z_uncalibrated`` and ``F_draw_var_share`` (share of the test variance due to uncertainty in F).
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(G_counts, dtype=float)
    coords = np.asarray(coords, dtype=float)
    coords_m = coords if coords_m is None else np.asarray(coords_m, dtype=float)
    A = kernel_map(coords, coords_m)
    if Z is not None and Z_m is None:
        Z_m = _apply(A, Z)
    Ym = core.standardize(core.log_standardize(M_int))
    n, p = Y.shape
    q = Ym.shape[1]

    # --- initialisation: weighted PCA factors of Pearson residuals (on the gene grid)
    Yg0 = core.pearson_residuals(Y)
    Rg0 = core._residualize(Yg0, Z)
    Rm0 = core._residualize(_apply(kernel_map(coords_m, coords), Ym), Z)
    if K is None:
        K = core.select_K(Rg0, Rm0)
    F0 = core.hidden_factors(Rg0, Rm0, K)
    lengths = [choose_length(F0[:, r], coords, length_grid) for r in range(K)]
    bases = {l: GPBasis(np.vstack([coords, coords_m]), l) for l in set(lengths)}
    Phi_g = [bases[l](coords) for l in lengths]
    Phi_m = [bases[l](coords_m) for l in lengths]
    alpha = []
    for r in range(K):
        f = F0[:, r] / F0[:, r].std()
        alpha.append(np.linalg.solve(Phi_g[r].T @ Phi_g[r] + 0.1 * np.eye(Phi_g[r].shape[1]), Phi_g[r].T @ f))

    def factors(al):
        Fg = np.column_stack([Phi_g[r] @ al[r] for r in range(K)]) if K else np.zeros((n, 0))
        Fm = np.column_stack([Phi_m[r] @ al[r] for r in range(K)]) if K else np.zeros((len(coords_m), 0))
        return Fg, Fm

    o = np.log(Y.sum(1) / Y.sum(1).mean())[:, None]
    x = np.log(Y + 0.5) - o
    Fg, Fm = factors(alpha)
    Dg = _design(Z, Fg)
    beta_g = _ols(Dg, x)
    s2 = np.maximum(((x - Dg @ beta_g) ** 2).mean(0), 1e-2)
    x = x + o
    W = np.zeros((q, p))
    evals, U = core.spatial_basis(coords_m, k=knn)
    zthr = np.full((p, q), z_select)
    if prior_pairs is not None:
        zthr[prior_pairs] = z_select_prior

    def link_parts(Fg, Fm, beta_g, x):
        Dg = _design(Z, Fg)
        Eg = x - o - Dg @ beta_g
        Dm = _design(Z_m, Fm)
        X = core._residualize(_apply(A, core.standardize(Eg)), Dm[:, 1:])
        R = core._residualize(Ym, Dm[:, 1:])
        return Eg, X, R, Dm

    P_alpha = None
    for it in range(n_iter):
        # genes: E-step for the latent log-rates, M-step for loadings and variances
        Dg = _design(Z, Fg)
        for _ in range(3):
            prior = o + Dg @ beta_g
            x, v = _pln_estep(Y, prior, s2, x)
            beta_g = _ols(Dg, x - o)
            s2 = np.maximum(((x - o - Dg @ beta_g) ** 2 + v).mean(0), 1e-3)
        Eg, X, R, Dm = link_parts(Fg, Fm, beta_g, x)

        # sparse links
        Xh, Rh = U.T @ X, U.T @ R
        Pg = core.band_power(Xh, band_size)
        T0, V0 = conditional_test(Xh, Rh, W, band_size)
        screen = (np.abs(T0 / np.sqrt(V0)) > zthr - 1.0) | (W.T != 0)
        W = _sparse_W(Xh, Rh, Pg, zthr, screen)

        # metabolite loadings given the link term
        Ym_adj = Ym - _apply(A, core.standardize(Eg)) @ W.T
        beta_m = _ols(Dm, Ym_adj)
        s2m = np.maximum(((Ym_adj - Dm @ beta_m) ** 2).mean(0), 1e-3)

        # factors: Gaussian update of α given loadings (posterior mean and precision)
        nz = Dg.shape[1] - K
        Lg, Lm = beta_g[nz:].T, beta_m[nz:].T          # p × K, q × K
        Xg_t = x - o - Dg[:, :nz] @ beta_g[:nz]         # gene data minus intercept and Z
        Xm_t = Ym_adj - Dm[:, :nz] @ beta_m[:nz]
        wg, wm = 1 / s2, 1 / s2m
        Gw = (Lg * wg[:, None]).T @ Lg
        Mw = (Lm * wm[:, None]).T @ Lm
        sizes = [P.shape[1] for P in Phi_g]
        offs = np.r_[0, np.cumsum(sizes)].astype(int)
        P_alpha = np.eye(offs[-1])
        rhs = np.zeros(offs[-1])
        bg = Xg_t @ (Lg * wg[:, None])                  # n × K
        bm = Xm_t @ (Lm * wm[:, None])
        for r in range(K):
            sr = slice(offs[r], offs[r + 1])
            rhs[sr] = Phi_g[r].T @ bg[:, r] + Phi_m[r].T @ bm[:, r]
            for s in range(r, K):
                ss = slice(offs[s], offs[s + 1])
                blk = Gw[r, s] * (Phi_g[r].T @ Phi_g[s]) + Mw[r, s] * (Phi_m[r].T @ Phi_m[s])
                P_alpha[sr, ss] += blk
                if s != r:
                    P_alpha[ss, sr] += blk.T
        a_flat = np.linalg.solve(P_alpha, rhs)
        alpha = [a_flat[offs[r]:offs[r + 1]] for r in range(K)]
        Fg, Fm = factors(alpha)
        sc = Fg.std(0)
        sc[sc == 0] = 1
        alpha = [alpha[r] / sc[r] for r in range(K)]
        P_alpha = P_alpha * np.repeat(sc, sizes)[:, None] * np.repeat(sc, sizes)[None, :]
        Fg, Fm = factors(alpha)
        beta_g = _ols(_design(Z, Fg), x - o)
        if verbose:
            print(f"iter {it}: |W|_0 = {(W != 0).sum()}", flush=True)

    # --- test, averaged over posterior draws of F (Rubin's rules)
    def test_at(Fg_d, Fm_d):
        Dg_d = _design(Z, Fg_d)
        b = _ols(Dg_d, x - o)
        _, X, R, _ = link_parts(Fg_d, Fm_d, b, x)
        Xh, Rh = U.T @ X, U.T @ R
        return conditional_test(Xh, Rh, W, band_size)

    T_mean, V_mean = test_at(Fg, Fm)
    share = 0.0
    if n_draws > 0 and K > 0:
        Lc = np.linalg.cholesky(P_alpha)
        a_mean = np.concatenate(alpha)
        Ts, Vs = [], []
        for _ in range(n_draws):
            a_d = a_mean + np.linalg.solve(Lc.T, rng.standard_normal(a_mean.size))
            al = [a_d[offs[r]:offs[r + 1]] for r in range(K)]
            T_d, V_d = test_at(*factors(al))
            Ts.append(T_d)
            Vs.append(V_d)
        Ts, Vs = np.array(Ts), np.array(Vs)
        between = Ts.var(0, ddof=1) * (1 + 1 / n_draws)
        T_stat = Ts.mean(0)
        V_tot = Vs.mean(0) + between
        share = float(np.median(between / V_tot))
    else:
        T_stat, V_tot = T_mean, V_mean

    z_raw = T_stat / np.sqrt(V_tot)
    c = core.calibration_factor(z_raw[neg_genes]) if (calibrate and neg_genes is not None) else 1.0
    z = z_raw / c
    pip, eb = two_groups_pip(z, prior_pairs)
    return {"z": z, "p": core.pvals(z), "pip": pip, "eb": eb, "W": W, "F_g": Fg, "F_m": Fm, "K": K,
            "lengths": lengths, "c": float(c), "Eg": Eg, "z_uncalibrated": z_raw, "F_draw_var_share": share}
