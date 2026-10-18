"""Full vs simplified ReSPAIR on the same simulations, including scenarios that break the model's assumptions.

Scenarios
- base 0.5 / base 0.2: ``simulate.py`` (the benchmark simulation), link strength 0.5 and 0.2.
- the rest are generated here at link strength 0.2 with one change each:
  - low depth: 500 counts per spot instead of 4000 (count noise dominates).
  - isozyme decoys: for 100 of the 150 linked genes, a second unlinked gene whose local residual is correlated
    (rho = 0.8) with the linked gene, as for isozymes of one reaction. The decoy pairs are scored separately.
  - shifted grid: metabolites measured on pixels offset by half a spot from the gene spots; every field is a
    continuous GP sampled at both sets of locations. The simplified estimator gets metabolites averaged to the
    gene spots; the full estimator uses the native coordinates.
  - network prior: a reaction-network prior that contains 70% of the true links plus 3 decoy pairs per true link.
  - link on total signal (misspecified): the metabolite tracks the gene's whole log-expression (tissue structure
    and confounders included), not its local residual.
  - nonlinear link (misspecified): the metabolite tracks max(local residual − 1, 0).
  - many weak programs (misspecified): instead of one confounder and a few private factors, 40 shared programs,
    each touching 5% of genes and metabolites with small loadings (not clearly low rank).
  - zero-inflated metabolites (misspecified): intensities below the 25th percentile and a random 10% set to 0.

Metrics: AUC and power at 5% false-positive rate (true links vs all unlinked pairs; vs confounded unlinked
pairs; vs isozyme decoy pairs), plus the realised false-positive rate at p < 0.05.

Usage: python benchmarks/scripts/full_vs_simple.py [n_seeds] [scenario ...]
Writes results/full_vs_simple.json (merged with existing entries) and figures/fig_full_vs_simple.png.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import respair
from respair.full import fit_full, kernel_map, two_groups_pip
from simulate import gp_samples, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
OUT = os.environ.get("RESPAIR_OUT", os.path.join(ROOT, "results", "full_vs_simple.json"))
SCENARIOS = ["base 0.5", "base 0.2", "low depth", "isozyme decoys", "shifted grid", "network prior",
             "link on total signal", "nonlinear link", "many weak programs", "zero-inflated metabolites"]


def make(seed, sc, side=50, p=2000, q=200, n_conf_genes=300, n_conf_mets=40, conf_strength=0.6, n_links=150,
         link=0.2, n_neg=500, depth=4000, theta=10.0, rho=0.8, n_iso=100):
    if sc.startswith("base"):
        d = simulate(seed=seed, link_strength=float(sc.split()[1]))
        d["coords_m"] = d["coords"]
        return d
    rng = np.random.default_rng(1000 + seed)
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    n = len(coords)
    shifted = sc == "shifted grid"
    coords_m = coords + 0.5 if shifted else coords
    allc = np.vstack([coords, coords_m]) if shifted else coords

    def field(length, k):
        X = gp_samples(allc, length, k, rng)
        return (X[:n], X[n:]) if shifted else (X, X)

    if sc == "low depth":
        depth = 500
    logits_g, logits_m = (2.0 * f for f in field(12, 3))
    Z = np.exp(logits_g) / np.exp(logits_g).sum(1, keepdims=True)
    Z_m = np.exp(logits_m) / np.exp(logits_m).sum(1, keepdims=True)
    Bg = rng.normal(0, 0.8, (3, p))
    Bm = rng.normal(0, 0.8, (3, q))

    conf_pair = np.zeros((p, q), bool)
    if sc == "many weak programs":
        n_prog = 40
        lengths = rng.uniform(3, 10, n_prog)
        Hg = np.zeros((n, n_prog)); Hm = np.zeros((len(coords_m), n_prog))
        for r, l in enumerate(lengths):
            a, b = field(l, 1)
            Hg[:, r], Hm[:, r] = a[:, 0], b[:, 0]
        Lg = rng.normal(0, 0.35, (n_prog, p)) * (rng.random((n_prog, p)) < 0.05)
        Lm = rng.normal(0, 0.35, (n_prog, q)) * (rng.random((n_prog, q)) < 0.05)
        conf_g, conf_m = Hg @ Lg, Hm @ Lm
        conf_pair = ((Lg != 0).astype(int).T @ (Lm != 0).astype(int)) > 0
    else:
        Hg, Hm = field(6, 1)
        cg = rng.choice(p, n_conf_genes, replace=False)
        cm = rng.choice(q, n_conf_mets, replace=False)
        lam_g = np.zeros(p); lam_g[cg] = rng.choice([-1, 1], n_conf_genes) * rng.uniform(0.5, 1, n_conf_genes)
        lam_m = np.zeros(q); lam_m[cm] = rng.choice([-1, 1], n_conf_mets) * rng.uniform(0.5, 1, n_conf_mets)
        Fg_g, Fg_m = field(8, 3)
        Fm_g, Fm_m = field(8, 2)
        Phi_g = rng.normal(0, 0.4, (3, p)) * (rng.random((3, p)) < 0.2)
        Phi_m = rng.normal(0, 0.4, (2, q)) * (rng.random((2, q)) < 0.3)
        conf_g = np.outer(Hg[:, 0], lam_g * conf_strength) + Fg_g @ Phi_g
        conf_m = np.outer(Hm[:, 0], lam_m * conf_strength) + Fm_m @ Phi_m
        conf_pair[np.ix_(cg, cm)] = True

    # links (chosen first so that isozyme decoys can be built into the gene residuals)
    neg_genes = rng.choice(p, n_neg, replace=False)
    cand = np.setdiff1d(np.arange(p), neg_genes)
    link_g = rng.choice(cand, n_links, replace=True)
    link_m = rng.choice(q, n_links, replace=True)
    sign = rng.choice([-1, 1], n_links) * link

    smooth_g, smooth_m = (0.5 * f for f in field(1.5, p))
    local_g = smooth_g + 0.3 * rng.standard_normal((n, p))
    local_at_m = smooth_m if shifted else local_g
    decoy = np.zeros((p, q), bool)
    if sc == "isozyme decoys":
        free = np.setdiff1d(cand, link_g)
        dec = rng.choice(free, n_iso, replace=False)
        for i, j2 in enumerate(dec):
            j = link_g[i]
            local_g[:, j2] = rho * local_g[:, j] / local_g[:, j].std() * local_g[:, j2].std() \
                + np.sqrt(1 - rho ** 2) * local_g[:, j2]
            decoy[j2, link_m[i]] = True
    sd = local_g.std(0)

    base = rng.normal(0, 1, p)
    eta = base + (Z - Z.mean(0)) @ Bg + conf_g + local_g
    mu = np.exp(eta)
    mu = mu / mu.sum(1, keepdims=True) * depth
    counts = rng.poisson(rng.gamma(theta, mu / theta))

    if sc == "link on total signal":
        reg = eta - eta.mean(0)
        reg = reg / reg.std(0)
    elif sc == "nonlinear link":
        reg = np.maximum(local_at_m / sd - 1, 0)
        reg = (reg - reg.mean(0)) / reg.std(0)
    else:
        reg = local_at_m / sd
    W = np.zeros((q, p))
    W[link_m, link_g] = sign
    local_m = 0.5 * field(1.5, q)[1] + 0.5 * rng.standard_normal((len(coords_m), q))
    logm = (Z_m - Z_m.mean(0)) @ Bm + conf_m + reg @ W.T + local_m
    M = np.exp(logm + 3.0)
    if sc == "zero-inflated metabolites":
        M[logm < np.quantile(logm, 0.25, axis=0)] = 0
        M[rng.random(M.shape) < 0.1] = 0

    truth = np.zeros((p, q), bool)
    truth[link_g, link_m] = True
    decoy &= ~truth
    out = dict(G=counts, M=M, coords=coords, coords_m=coords_m, Z=Z, Z_m=Z_m, truth=truth,
               conf_pair=conf_pair, neg_genes=neg_genes, decoy=decoy)
    if sc == "network prior":
        prior = np.zeros((p, q), bool)
        tl = np.argwhere(truth)
        keep = tl[rng.random(len(tl)) < 0.7]
        prior[keep[:, 0], keep[:, 1]] = True
        n_dec = 3 * len(tl)
        prior[rng.choice(cand, n_dec), rng.choice(q, n_dec)] = True
        out["prior"] = prior
    return out


def roc_stats(score, pos, neg):
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    s = np.r_[score[pos], score[neg]]
    fpr, tpr, _ = roc_curve(y, s)
    return {"AUC": float(roc_auc_score(y, s)), "TPR@5%": float(np.interp(0.05, fpr, tpr))}


def evaluate(score, p, d):
    truth = d["truth"]
    null = ~truth
    out = {"all pairs": roc_stats(score, truth, null),
           "confounded pairs": roc_stats(score, truth, d["conf_pair"] & null),
           "FPR at p<0.05": float((p[null] < 0.05).mean())}
    if d.get("decoy") is not None and d["decoy"].any():
        out["isozyme decoys"] = roc_stats(score, truth, d["decoy"])
        out["FPR at p<0.05, decoys"] = float((p[d["decoy"]] < 0.05).mean())
    return out


def run(seed, sc):
    d = make(seed, sc)
    t = time.time()
    if np.allclose(d["coords"], d["coords_m"]) and d["coords"].shape == d["coords_m"].shape:
        M_on_g = d["M"]
    else:
        M_on_g = kernel_map(d["coords_m"], d["coords"]) @ d["M"]
    simple = respair.fit(d["G"], M_on_g, d["coords"], Z=d["Z"], neg_genes=d["neg_genes"])
    t_simple = time.time() - t
    t = time.time()
    full = fit_full(d["G"], d["M"], d["coords"], coords_m=d["coords_m"], Z=d["Z"], Z_m=d.get("Z_m"),
                    neg_genes=d["neg_genes"], prior_pairs=d.get("prior"))
    t_full = time.time() - t
    lp = lambda p: -np.log10(np.clip(p, 1e-300, 1))  # noqa: E731
    res = {"simplified": evaluate(lp(simple["p"]), simple["p"], d),
           "full": evaluate(lp(full["p"]), full["p"], d),
           "_K": [int(simple["K"]), int(full["K"])], "_time_s": [t_simple, t_full],
           "_F_draw_var_share": full["F_draw_var_share"]}
    if "prior" in d:
        pip_s, _ = two_groups_pip(simple["z"], d["prior"])
        res["simplified + prior (EB)"] = evaluate(pip_s, simple["p"], d)
        res["full + prior (EB)"] = evaluate(full["pip"], full["p"], d)
    return res


def summarise(per_seed):
    methods = [m for m in per_seed[0] if not m.startswith("_")]
    out = {}
    for m in methods:
        out[m] = {}
        for key, val in per_seed[0][m].items():
            if isinstance(val, dict):
                out[m][key] = {k: [float(np.mean([r[m][key][k] for r in per_seed])),
                                   float(np.std([r[m][key][k] for r in per_seed]))] for k in val}
            else:
                out[m][key] = [float(np.mean([r[m][key] for r in per_seed])),
                               float(np.std([r[m][key] for r in per_seed]))]
    return out


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    scenarios = sys.argv[2:] or SCENARIOS
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for sc in scenarios:
        per_seed = []
        for seed in range(n_seeds):
            per_seed.append(run(seed, sc))
            print(sc, "seed", seed, per_seed[-1]["_time_s"], flush=True)
        results[sc] = {"n_seeds": n_seeds, "mean_sd": summarise(per_seed), "per_seed": per_seed}
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)
        for m, v in results[sc]["mean_sd"].items():
            print(sc, m, {k: (round(x[0], 3) if isinstance(x, list) else {kk: round(xx[0], 3) for kk, xx in x.items()})
                          for k, x in v.items()}, flush=True)
