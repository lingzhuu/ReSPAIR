"""Figure: power at 5% false-positive rate and AUC, simplified vs full ReSPAIR, every scenario.

Reads results/full_vs_simple.json, full_vs_simple_misspec.json and full_vs_simple_semisim.json.
Writes results/figures/fig_full_vs_simple.png.
Usage: python benchmarks/scripts/plot_full_vs_simple.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from plot_sim import GRID, INK2, MUTED, SERIES, SURF

R = os.path.join(ROOT, "results")


def rows():
    out = []
    own = json.load(open(os.path.join(R, "full_vs_simple.json")))
    for sc, v in own.items():
        m = v["mean_sd"]
        label = f"{sc} (link 0.5)" if sc == "base 0.5" else f"{sc} (link 0.2)"
        out.append(("Benchmark simulation and variants (5 seeds)", label,
                    [m["simplified"]["all pairs"][k][0] for k in ("TPR@5%", "AUC")],
                    [m["full"]["all pairs"][k][0] for k in ("TPR@5%", "AUC")]))
        if sc == "isozyme decoys":
            out.append(("Benchmark simulation and variants (5 seeds)", "isozyme decoys: true vs decoy pairs",
                        [m["simplified"]["isozyme decoys"][k][0] for k in ("TPR@5%", "AUC")],
                        [m["full"]["isozyme decoys"][k][0] for k in ("TPR@5%", "AUC")]))
        if sc == "network prior":
            out.append(("Benchmark simulation and variants (5 seeds)", "network prior, ranked by PIP",
                        [m["simplified + prior (EB)"]["all pairs"][k][0] for k in ("TPR@5%", "AUC")],
                        [m["full + prior (EB)"]["all pairs"][k][0] for k in ("TPR@5%", "AUC")]))
    for f, group in [("full_vs_simple_misspec.json", "Misspecified scenarios, misspec.py (link 0.5, 3 seeds)"),
                     ("full_vs_simple_semisim.json", "Semi-synthetic on the SMA section, semisim.py (3 seeds)")]:
        d = json.load(open(os.path.join(R, f)))["results"]
        for sc, v in d.items():
            out.append((group, sc, v["simplified"]["all pairs"][::-1], v["full"]["all pairs"][::-1]))
    return out


if __name__ == "__main__":
    data = rows()
    groups = list(dict.fromkeys(g for g, *_ in data))
    ypos, labels, y, gy = [], [], 0, []
    for g in groups:
        gy.append((y, g)); y += 1
        for gg, lab, *_ in data:
            if gg == g:
                ypos.append(y); labels.append(lab); y += 1
        y += 0.4
    ypos = -np.array(ypos)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.32 * y + 1.6), sharey=True)
    for ax, i, title, lo in [(axes[0], 0, "Power at 5% false-positive rate", 0.0), (axes[1], 1, "AUC", 0.6)]:
        for yy, (_, _, s, f) in zip(ypos, data):
            ax.plot([s[i], f[i]], [yy, yy], color=GRID, lw=2, zorder=1)
            ax.scatter(s[i], yy, color=MUTED, s=38, zorder=2, label="simplified" if yy == ypos[0] else None)
            ax.scatter(f[i], yy, color=SERIES[0], s=38, zorder=3, label="full" if yy == ypos[0] else None)
        ax.set_xlim(lo, 1.01)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for y0, g in gy:
            ax.text(lo, -y0, g if i == 0 else "", fontsize=10, fontweight="bold", color=INK2, va="center")
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(labels, fontsize=9)
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2)
    fig.suptitle("Full vs simplified ReSPAIR, true links vs all unlinked pairs", fontsize=13, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(R, "figures", "fig_full_vs_simple.png"), dpi=150)
