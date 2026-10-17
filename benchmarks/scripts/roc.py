"""Supplementary ROC: ablations of ReSPAIR (each step on its own), with plain Pearson as reference.

The main comparison against published methods is compare.py.

Two comparisons:
- all pairs: true links vs every pair without a link
- confounded pairs only: true links vs unlinked pairs affected by the unlabeled confounder (harder)
Methods are ranked by p-value, so calibration does not matter here.

Usage: python benchmarks/scripts/roc.py [n_seeds] [link strength, default 0.5]
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from respair.simulate import simulate
from run_sim import all_pvals
from plot_sim import SERIES, INK, INK2, MUTED, GRID, SURF  # same colours and fonts

METHODS = ["Pearson", "Z-only + Pearson", "Z-only + spectral test",
           "Hidden factors + Pearson", "ReSPAIR"]
LABEL = {m: m for m in METHODS}
COLORS = SERIES + ["#008300", "#6250d6"]
GRID_FPR = np.linspace(0, 1, 501)


def roc_stats(p, pos, neg):
    score = -np.log10(np.clip(p, 1e-300, 1))
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    s = np.r_[score[pos], score[neg]]
    fpr, tpr, _ = roc_curve(y, s)
    curve = np.interp(GRID_FPR, fpr, tpr)
    return {"AUC": float(roc_auc_score(y, s)), "TPR@5%": float(np.interp(0.05, fpr, tpr)), "curve": curve}


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    link = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    tag = "" if link == 0.5 else f"_link{link}"
    res = {s: {m: [] for m in METHODS} for s in ["all pairs", "confounded pairs only"]}
    for seed in range(n_seeds):
        d = simulate(seed=seed, link_strength=link)
        P, _, _ = all_pvals(d, with_mantis=False)
        truth = d["truth"]
        sets = {"all pairs": ~truth, "confounded pairs only": d["conf_pair"] & ~truth}
        for sname, neg in sets.items():
            for m in METHODS:
                res[sname][m].append(roc_stats(P[m], truth, neg))
        print("seed", seed, flush=True)

    summary = {s: {m: {k: [float(np.mean([r[k] for r in v])), float(np.std([r[k] for r in v]))]
                       for k in ["AUC", "TPR@5%"]} for m, v in res[s].items()} for s in res}
    with open(os.path.join(ROOT, "results", f"ablation_roc{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, sname in zip(axes, res):
        for i, m in enumerate(METHODS):
            curve = np.mean([r["curve"] for r in res[sname][m]], axis=0)
            auc, tpr5 = summary[sname][m]["AUC"][0], summary[sname][m]["TPR@5%"][0]
            ax.plot(GRID_FPR, curve, color=COLORS[i], lw=2.4 if m == "ReSPAIR" else 1.8,
                    label=f"{LABEL[m]}  AUC {auc:.2f}, power at 5% FPR {tpr5:.0%}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.axvline(0.05, color=INK, lw=1, ls=":")
        ax.text(0.06, 0.03, "FPR 5%", fontsize=9, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("power")
        title = "True links vs all unlinked pairs" if sname == "all pairs" else "True links vs confounded unlinked pairs"
        ax.set_title(title, loc="left")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle(f"Supplementary: ReSPAIR ablations, ROC curves (simulation, link strength {link}, mean of {n_seeds} seeds)", fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(ROOT, "results", "figures", f"fig_ablation_ROC{tag}.png"), dpi=160)
    for s in summary:
        for m in METHODS:
            print(s, m, summary[s][m])
