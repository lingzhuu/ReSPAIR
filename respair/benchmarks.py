"""对比方法。

MANTIS 按其公开代码（github.com/yuhaotuo/MANTIS，v0.55，tools/_spc.py 与 _sampler.py）的流程复现，
做了两处为速度的改动，统计量和零分布的构造不变：
- 保持空间自相关的置换：原代码用 networkx 图上的模拟退火逐对交换；这里用稠密权重矩阵向量化实现同一个目标
  （让置换后 x' W x 等于原值），贪心接受使目标差变小的交换。
- 每个代谢物生成 1 个置换（与原代码默认一致），零分布按代谢物空间自相关分 10 箱汇总。
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
    """MANTIS 采样器的权重：距离 <= l 权重为 1，之外为高斯衰减。"""
    D = cdist(coords, coords)
    W = np.where(D <= l, 1.0, np.exp(-D ** 2 / (2 * l * l)))
    np.fill_diagonal(W, 0.0)
    return W


def autocorr_preserving_shuffle(x, W, rng, max_steps=60000, rel_tol=1e-3):
    """返回 x 的一个置换，使 x' W x 接近原值（MANTIS 零分布的思路）。"""
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
        # 交换 u, v 后 y'Wy 的变化
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
    """MANTIS 的 SCI（CT=None）或 SPC-CT（给出细胞类型比例 CT）。返回 p 值矩阵（基因 × 代谢物）。"""
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

    # 按代谢物的空间自相关分箱，箱内汇总零分布
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
    """LIANA+（Dimitrov et al., Nat Cell Biol 2024）的全局双变量指标，按其官方 API 调用。

    返回 {"morans": (score, p), "lee": (score, None)}，形状 基因 × 代谢物。
    Moran's R 的解析 p 值即 SpatialDM（Li et al., Nat Commun 2023）的全局检验，LIANA+ 内置了同一实现。
    表达比例低于 nz_prop 的基因被 LIANA+ 过滤，这些配对记为 score 0、p 1。
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
    # 一次放入全部 p×q 配对会占用几十 GB 内存，按代谢物分块调用
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
    """Dutilleul (1993) 修正 t 检验，按 R 包 SpatialPack::modified.ttest 的思路实现并向量化。

    用分距离等级的 Moran 相关图估计两变量各自的空间自相关，得到每对的有效样本量
    n_eff = 1 + n² / (n + Σ_k n_k I_x(k) I_y(k))，再用 F(1, n_eff − 2) 检验相关系数。
    返回 (r, p)，形状 基因 × 代谢物。
    """
    from scipy.stats import f as fdist
    X = _zscore(Xg); Y = _zscore(Xm)
    n = len(coords)
    D = cdist(coords, coords)
    if n_classes is None:
        n_classes = int(np.ceil(np.log2(n * (n - 1) / 2) + 1))  # Sturges，与 SpatialPack 一致
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
        Ig.append((X * (A @ X)).sum(0) / cnt)   # 标准化后 Σx² = n，所以 I(k) = (n/W_k) Σ x_i x_j / n
        Im.append((Y * (A @ Y)).sum(0) / cnt)
    Ig = np.array(Ig); Im = np.array(Im); nk = np.array(nk)
    trace = n + (Ig * nk[:, None]).T @ Im          # p × q：tr(Σx Σy)
    n_eff = np.clip(1 + n ** 2 / trace, 3, n)
    r = X.T @ Y / n
    F = r ** 2 * (n_eff - 2) / np.clip(1 - r ** 2, 1e-12, None)
    return r, fdist.sf(F, 1, n_eff - 2)


def mefisto(G_counts, M_int, coords, K=10, seed=0, n_inducing=500, iters=1000):
    """MEFISTO（Velten et al., Nat Methods 2022，mofapy2）。两个视图（基因、代谢物），空间坐标作为协变量。

    MEFISTO 不直接给配对检验；按多组学因子模型的常规读法，基因 j 与代谢物 k 的关联分数取
    Σ_r w_g,jr w_m,kr（共享因子载荷乘积，即我们原方案的读法）。返回 分数矩阵 基因 × 代谢物。
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
    ent.set_smooth_options(sparseGP=True, n_inducing=n_inducing, scale_cov=False)
    ent.build()
    ent.run()
    W = ent.model.nodes["W"].getExpectation()
    Wg, Wm = W[0], W[1]
    return Wg @ Wm.T
