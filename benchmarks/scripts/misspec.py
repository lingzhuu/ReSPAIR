"""Robustness: the same benchmark when the simulation breaks ReSPAIR's assumptions.

The default simulator follows ReSPAIR's own model (low-rank confounding, links on the gene's local residual), which
favours ReSPAIR. Each scenario here changes one thing:
- links on total expression: the metabolite follows the gene's whole log-expression, not only its local residual
- threshold links: the metabolite responds only where the gene is above its median
- many confounders: 10 extra unlabeled confounders, each on its own gene and metabolite set
- cell-type drift: cell-type effects change along the tissue (not removed by regressing on cell types)
- misregistration: metabolite map shifted by one spot
- low MSI resolution: metabolites measured on 2 × 2 blocks
- MSI dropout: 30% of metabolite values at the detection floor

Usage: python benchmarks/scripts/misspec.py [n_seeds, default 3] [link strength, default 0.5]
Writes results/misspec.json and results/figures/fig_misspec.png.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from simulate import simulate
from compare import METHODS, roc_stats, scores
from plot_sim import GRID, INK2, SURF

SCENARIOS = {
    "ReSPAIR's model (default)": {},
    "links on total expression": {"link_on": "total"},
    "threshold links": {"link_fn": "threshold"},
    "many confounders": {"n_extra_conf": 10},
    "cell-type drift": {"ct_drift": 1.0},
    "misregistration (1 spot)": {"misalign": 1},
    "low MSI resolution (2×2)": {"msi_block": 2},
    "MSI dropout (30%)": {"msi_dropout": 0.3},
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#1f1f1e"]


def tag_of(kw):
    return "".join(f"_{k}{v}" for k, v in sorted(kw.items()))


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    link = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    out = {}
    for name, kw in SCENARIOS.items():
        rows = {m: {"all pairs": [], "confounded pairs only": []} for m in METHODS}
        for seed in range(n_seeds):
            d = simulate(seed=seed, link_strength=link, **kw)
            S = scores(seed, link, METHODS, d, tag=tag_of(kw), sim_kw=kw)
            for sname, neg in {"all pairs": ~d["truth"], "confounded pairs only": d["conf_pair"] & ~d["truth"]}.items():
                for m in METHODS:
                    r = roc_stats(S[m], d["truth"], neg)
                    rows[m][sname].append([r["AUC"], r["TPR@5%"]])
        out[name] = {m: {s: dict(zip(["AUC", "TPR@5%"], np.mean(v, 0).tolist())) for s, v in r.items()}
                     for m, r in rows.items()}
        print(name, {m: round(out[name][m]["confounded pairs only"]["AUC"], 2) for m in METHODS}, flush=True)
    with open(os.path.join(ROOT, "results", "misspec.json"), "w") as f:
        json.dump({"link strength": link, "seeds": n_seeds, "mean over seeds": out}, f, indent=1)

    names = list(SCENARIOS)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    panels = [("all pairs", "AUC"), ("all pairs", "TPR@5%"), ("confounded pairs only", "AUC"),
              ("confounded pairs only", "TPR@5%")]
    y = np.arange(len(names))[::-1]
    for ax, (sname, metric) in zip(axes.ravel(), panels):
        for i, m in enumerate(METHODS):
            v = [out[n][m][sname][metric] for n in names]
            ax.scatter(v, y + (i - 3) * 0.1, s=60 if m == "ReSPAIR" else 34, color=COLORS[i], zorder=3,
                       edgecolor=SURF, linewidth=1, label=m)
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for yy in y[:-1] - 0.5:
            ax.axhline(yy, color=GRID, lw=0.6)
        what = "AUC" if metric == "AUC" else "Power at 5% FPR"
        vs = "all unlinked pairs" if sname == "all pairs" else "confounded unlinked pairs"
        ax.set_title(f"{what}\ntrue links vs {vs}", loc="left", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=7, fontsize=9.5)
    fig.suptitle(f"Robustness when the simulation breaks ReSPAIR's assumptions (link strength {link}, {n_seeds} seeds)",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left", color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig_misspec.png"), dpi=160)
