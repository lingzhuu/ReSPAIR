"""Main benchmark: published methods vs ReSPAIR on simulated data with known truth.

For every method the gene × metabolite pairs are ranked (by p-value, or by |score| for methods without a test), and we
report AUC and power at 5% false-positive rate, in two comparisons:
- all pairs: true links vs every pair without a link
- confounded pairs only: true links vs unlinked pairs affected by the unlabeled confounder (harder)

Only methods that test each gene–metabolite pair are included; integration or factor models without a pairwise test
(SpaMV, MEFISTO, MISTy) are not competitors. Per-seed rankings are cached in benchmarks/cache/ (not committed), so
re-running only computes what is missing.

Usage: python benchmarks/scripts/compare.py [--seeds 5] [--link 0.5] [--methods A,B,...]
Writes results/benchmark{tag}.json and results/figures/fig_benchmark_ROC{tag}.png.
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from respair import benchmarks as bm
from respair import core as sj
from respair.simulate import simulate
from plot_sim import INK, MUTED, GRID, SURF

CACHE = os.path.join(ROOT, "cache")
GRID_FPR = np.linspace(0, 1, 501)


def _neglog(p):
    return -np.log10(np.clip(p, 1e-300, 1))


def _prep(d):
    Yg = sj.pearson_residuals(d["G"])
    Ym = sj.standardize(sj.log_standardize(d["M"]))
    return Yg, Ym


# Each entry returns {method: ranking score (higher = more likely linked)} for one simulated data set.
def _pearson(d):
    Yg, Ym = _prep(d)
    return {"Pearson": _neglog(sj.pvals(sj.naive_z(Yg, Ym)))}


def _mantis(d):
    Yg, Ym = _prep(d)
    return {"MANTIS-SCI": _neglog(bm.mantis(Yg, Ym, d["coords"], CT=None, seed=1)),
            "MANTIS-SPC-CT": _neglog(bm.mantis(Yg, Ym, d["coords"], CT=d["Z"], seed=1))}


def _liana(d):
    out = bm.liana_global(d["G"], d["M"], d["coords"])
    return {"SpatialDM (Moran's R)": _neglog(out["morans"][1]), "Lee's L": np.abs(out["lee"][0])}


def _dutilleul(d):
    Yg, Ym = _prep(d)
    return {"Modified t-test": _neglog(bm.modified_ttest(Yg, Ym, d["coords"])[1])}


def _respair(d):
    Yg, Ym = _prep(d)
    fit = sj.fit(d["G"], d["M"], d["coords"], Z=d["Z"], neg_genes=d["neg_genes"], gene_resid=Yg, met_resid=Ym)
    return {"ReSPAIR": _neglog(fit["p"])}


RUNNERS = [(["Pearson"], _pearson), (["MANTIS-SCI", "MANTIS-SPC-CT"], _mantis),
           (["SpatialDM (Moran's R)", "Lee's L"], _liana), (["Modified t-test"], _dutilleul),
           (["ReSPAIR"], _respair)]
METHODS = [m for names, _ in RUNNERS for m in names]


def _cache_path(method, seed, link, tag=""):
    safe = "".join(c if c.isalnum() else "_" for c in method)
    return os.path.join(CACHE, f"{safe}_seed{seed}_link{link}{tag}.npy")


def scores(seed, link, methods, d=None, tag="", sim_kw=None):
    out = {}
    for names, fn in RUNNERS:
        names = [m for m in names if m in methods]
        if not names:
            continue
        paths = {m: _cache_path(m, seed, link, tag) for m in names}
        if not all(os.path.exists(p) for p in paths.values()):
            d = d if d is not None else simulate(seed=seed, link_strength=link, **(sim_kw or {}))
            t = time.time()
            res = fn(d)
            print(f"  seed {seed} {'/'.join(res)}: {time.time() - t:.0f}s", flush=True)
            os.makedirs(CACHE, exist_ok=True)
            for m, s in res.items():
                np.save(_cache_path(m, seed, link, tag), s)
        for m in names:
            out[m] = np.load(paths[m])
    return out


def roc_stats(score, pos, neg):
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    s = np.r_[score[pos], score[neg]]
    fpr, tpr, _ = roc_curve(y, s)
    return {"AUC": float(roc_auc_score(y, s)), "TPR@5%": float(np.interp(0.05, fpr, tpr)),
            "curve": np.interp(GRID_FPR, fpr, tpr)}


def plot(res, summary, n_seen, link, path):
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#8a6d3b", "#5f5e5a"]
    order = [m for m in METHODS if m != "ReSPAIR"] + ["ReSPAIR"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, sname in zip(axes, res):
        for m in order:
            if m not in summary[sname]:
                continue
            curve = np.mean([r["curve"] for r in res[sname][m]], axis=0)
            auc, tpr5 = summary[sname][m]["AUC"][0], summary[sname][m]["TPR@5%"][0]
            is_ours = m == "ReSPAIR"
            ax.plot(GRID_FPR, curve, color="#1f1f1e" if is_ours else colors[METHODS.index(m)],
                    lw=2.8 if is_ours else 1.7, zorder=3 if is_ours else 2,
                    label=f"{m}  AUC {auc:.2f}, power at 5% FPR {tpr5:.0%}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.axvline(0.05, color=INK, lw=1, ls=":")
        ax.text(0.06, 0.03, "FPR 5%", fontsize=9, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("power")
        ax.set_title("True links vs all unlinked pairs" if sname == "all pairs"
                     else "True links vs confounded unlinked pairs", loc="left")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_facecolor(SURF)
        ax.legend(frameon=False, loc="lower right", fontsize=8.3)
    seeds_note = ", ".join(f"{m} {n}" for m, n in n_seen.items() if n != max(n_seen.values()))
    title = f"ROC curves, simulation (link strength {link}, mean of {max(n_seen.values())} seeds"
    title += f"; {seeds_note} seeds)" if seeds_note else ")"
    fig.suptitle(title, fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--link", type=float, default=0.5)
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    methods = a.methods.split(",")
    tag = "" if a.link == 0.5 else f"_link{a.link}"

    res = {s: {m: [] for m in methods} for s in ["all pairs", "confounded pairs only"]}
    for seed in range(a.seeds):
        todo = methods
        d = simulate(seed=seed, link_strength=a.link)
        S = scores(seed, a.link, todo, d)
        sets = {"all pairs": ~d["truth"], "confounded pairs only": d["conf_pair"] & ~d["truth"]}
        for sname, neg in sets.items():
            for m in todo:
                res[sname][m].append(roc_stats(S[m], d["truth"], neg))
        print("seed", seed, "done", flush=True)

    summary = {s: {m: {k: [float(np.mean([r[k] for r in v])), float(np.std([r[k] for r in v]))]
                       for k in ["AUC", "TPR@5%"]} for m, v in res[s].items() if v} for s in res}
    n_seen = {m: len(res["all pairs"][m]) for m in methods if res["all pairs"][m]}
    out = {"link strength": a.link, "seeds per method": n_seen, "mean, sd over seeds": summary}
    with open(os.path.join(ROOT, "results", f"benchmark{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    plot(res, summary, n_seen, a.link, os.path.join(ROOT, "results", "figures", f"fig_benchmark_ROC{tag}.png"))
    for s in summary:
        print(s)
        for m, v in summary[s].items():
            print(f"  {m:24s} AUC {v['AUC'][0]:.3f}  power@5%FPR {v['TPR@5%'][0]:.3f}")
