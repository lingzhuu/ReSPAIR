"""Figures for the simulation benchmark, written to results/figures/.

Run run_sim.py 5 and sweep_sim.py 2 first (figure 2 reads results/sim_5seeds.json, figure 4 results/sweep.json).
Usage: python benchmarks/scripts/plot_sim.py
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
from respair import core as sj
from respair.simulate import simulate

FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# Colours: categorical series in fixed order; diverging blue–grey–red
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, SURF = "#1f1f1e", "#5f5e5a", "#8a8984", "#e6e5e0", "#fcfcfb"
DIVERGE = LinearSegmentedColormap.from_list("bgr", ["#104281", "#3987e5", "#f0efec", "#e66767", "#9b1c1c"])

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def spatial(ax, v, side, title):
    v = (v - v.mean()) / v.std()
    ax.imshow(v.reshape(side, side), cmap=DIVERGE, vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scatter(ax, x, y, title, color):
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, s=4, color=color, alpha=0.35, linewidths=0)
    ax.set_title(f"{title}\nr = {r:.2f}", fontsize=10.5)
    ax.set_xlabel("gene"); ax.set_ylabel("metabolite")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)


def fig1(d, fit, Yg, Ym):
    """A confounded pair with no link and a truly linked pair, before and after removing hidden factors."""
    side = int(np.sqrt(len(d["coords"])))
    conf_null = d["conf_pair"] & ~d["truth"]
    raw_r = np.abs(sj.standardize(Yg).T @ sj.standardize(Ym) / len(Yg))
    j, k = np.unravel_index(np.argmax(np.where(conf_null, raw_r, 0)), raw_r.shape)
    tj, tk = np.argwhere(d["truth"])[np.argmax(np.abs(fit["z"][d["truth"]]))]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    spatial(axes[0, 0], d["H"], side, "Unlabeled confounder (hypoxia-like gradient)")
    spatial(axes[0, 1], Yg[:, j], side, f"Gene {j} (confounded)\nraw")
    spatial(axes[0, 2], Ym[:, k], side, f"Metabolite {k} (confounded)\nraw")
    scatter(axes[0, 3], Yg[:, j], Ym[:, k], "No true link, yet strongly correlated in raw data", SERIES[1])

    zjk = fit["z"][j, k]
    axes[1, 0].axis("off")
    axes[1, 0].text(0.02, 0.62, "After ReSPAIR removes known\nstructure and hidden factors:", fontsize=11, color=INK, va="center")
    axes[1, 0].text(0.02, 0.30, f"z for this pair = {zjk:.2f}\n(not significant; false positive gone)", fontsize=11, color=INK2, va="center")
    spatial(axes[1, 1], fit["Eg"][:, j], side, f"Gene {j}\nresidual")
    spatial(axes[1, 2], fit["Em"][:, k], side, f"Metabolite {k}\nresidual")
    scatter(axes[1, 3], fit["Eg"][:, tj], fit["Em"][:, tk],
            f"Reference: residuals of a true link\n(gene {tj} – metabolite {tk}, z = {fit['z'][tj, tk]:.1f})", SERIES[0])
    fig.suptitle("Before and after removing hidden factors: spurious correlation vanishes, true link remains", fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "Spatial maps: standardised values, blue = low, red = high.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig1_overview.png"), dpi=160)
    plt.close(fig)


LABEL = {
    "Pearson": "Pearson\n(Desium / SpatialMETA-style)",
    "MANTIS-SCI": "MANTIS-SCI\n(no structure removed)",
    "MANTIS-SPC-CT": "MANTIS-SPC-CT\n(cell types removed)",
    "ReSPAIR": "ReSPAIR",
    "Z-only + Pearson": "Known structure only\n+ Pearson",
    "Z-only + spectral test": "Known structure only\n+ spectral test",
    "Hidden factors + Pearson": "Hidden factors removed\n+ Pearson",
    "ReSPAIR (uncalibrated)": "ReSPAIR\n(no negative-control calibration)",
}


def fig2():
    from run_sim import MAIN, ABLATION
    res = json.load(open(os.path.join(ROOT, "results", "sim_5seeds.json")))
    avg, sd = res["mean"], res["sd"]
    methods = MAIN + ABLATION
    gap = 0.8
    y = []
    pos = 0.0
    for i, m in enumerate(methods):
        if i == len(MAIN):
            pos += gap
        y.append(pos); pos += 1
    y = -np.array(y)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})
    h = 0.38
    for i, (key, name) in enumerate([("FPR confounded pairs", "confounded null pairs"), ("FPR other null pairs", "other null pairs")]):
        v = np.array([avg[m][key] for m in methods]) * 100
        e = np.array([sd[m][key] for m in methods]) * 100
        yy = y + (h / 2 + 0.02) * (1 - 2 * i)
        axes[0].barh(yy, v, height=h, color=SERIES[i], label=name, xerr=e,
                     error_kw={"ecolor": INK2, "lw": 1, "capsize": 0})
        for a, b, c in zip(yy, v, e):
            axes[0].text(b + c + 1.5, a, f"{b:.1f}%", va="center", fontsize=8.5, color=INK2)
    axes[0].axvline(5, color=INK, lw=1, ls="--")
    axes[0].set_title("False-positive rate (p < 0.05, dashed = 5%)", loc="left")
    axes[0].set_xlim(0, 105)
    axes[0].legend(frameon=False, loc="lower right")
    for ax, key, ref, title in [(axes[1], "BH empirical FDR", 10, "Empirical FDR after BH(0.1) (dashed = 10%)"),
                                (axes[2], "BH power", None, "Power for true links after BH(0.1)")]:
        v = np.array([avg[m][key] for m in methods]) * 100
        e = np.array([sd[m][key] for m in methods]) * 100
        ax.barh(y, v, height=0.55, color=SERIES[0], xerr=e, error_kw={"ecolor": INK2, "lw": 1, "capsize": 0})
        for a, b, c in zip(y, v, e):
            ax.text(min(b + c, 100) + 1.5, a, f"{b:.1f}%", va="center", fontsize=8.5, color=INK2)
        if ref:
            ax.axvline(ref, color=INK, lw=1, ls="--")
        ax.set_title(title, loc="left"); ax.set_xlim(0, 118)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[m] for m in methods])
    for t, m in zip(axes[0].get_yticklabels(), methods):
        if m == "ReSPAIR":
            t.set_fontweight("bold"); t.set_color(INK)
    axes[0].text(-0.02, y[0] + 0.75, "Existing methods vs ReSPAIR", transform=axes[0].get_yaxis_transform(), ha="right",
                 fontsize=10.5, fontweight="bold")
    axes[0].text(-0.02, y[len(MAIN)] + 0.75, "Ablation: each step of ReSPAIR", transform=axes[0].get_yaxis_transform(),
                 ha="right", fontsize=10.5, fontweight="bold")
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel("%")
    fig.suptitle("Benchmark on simulations with an unlabeled shared confounder (5 seeds, error bars = SD)",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig2_benchmark.png"), dpi=160)
    plt.close(fig)


def fig3(d, P):
    """QQ plot of p-values for null pairs."""
    null = ~d["truth"]
    names = ["Pearson", "MANTIS-SCI", "MANTIS-SPC-CT", "Z-only + spectral test", "ReSPAIR"]
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    m = int(null.sum())
    lim = -np.log10(0.5 / m)
    ends = []
    for i, name in enumerate(names):
        p = np.sort(np.clip(P[name][null], 1e-300, 1))
        idx = np.unique(np.concatenate([np.arange(2000), np.geomspace(1, m, 3000).astype(int) - 1]))
        exp = -np.log10((idx + 0.5) / m); obs = np.minimum(-np.log10(p[idx]), 30)
        ax.plot(exp, obs, color=SERIES[i], lw=2)
        ends.append([obs[0], name, SERIES[i]])
    # Label lines directly on the right, avoiding overlaps
    ends.sort(key=lambda t: t[0])
    last = -10
    for val, name, col in ends:
        yv = max(val, last + 1.6); last = yv
        ax.plot([lim, lim + 0.15], [val, yv], color=col, lw=1)
        ax.text(lim + 0.2, yv, LABEL[name].replace("\n", " "), color=INK, fontsize=9, va="center")
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("expected −log10(p)"); ax.set_ylabel("observed −log10(p) (capped at 30)")
    ax.set_xlim(0, lim + 3.6); ax.set_ylim(0, 32)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("P-values of null pairs (QQ plot, seed 0): closer to the diagonal is better", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_qq.png"), dpi=160)
    plt.close(fig)


def fig4():
    sw = json.load(open(os.path.join(ROOT, "results", "sweep.json")))
    methods = ["Pearson", "MANTIS-SCI", "MANTIS-SPC-CT", "Z-only + spectral test", "ReSPAIR"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    conf = sw["confounder strength"]; xs = [float(c) for c in conf]
    ends = []
    for i, m in enumerate(methods):
        ys = [conf[c][m]["FPR confounded pairs"] * 100 for c in conf]
        axes[0].plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=8, mec=SURF, mew=2)
        ends.append([ys[-1], m])
    ends.sort(); last = -10
    for val, m in ends:
        yv = max(val, last + 5); last = yv
        axes[0].text(xs[-1] + 0.04, yv, LABEL[m].replace("\n", " "), fontsize=8.5, va="center")
    axes[0].axhline(5, color=INK, lw=1, ls="--")
    axes[0].set_xlim(-0.05, 1.75); axes[0].set_ylim(0, 105)
    axes[0].set_xlabel("strength of unlabeled confounder (0 = none)"); axes[0].set_ylabel("FPR on confounded null pairs (%)")
    axes[0].set_title("Stronger confounding inflates other methods' FPR (dashed = 5%)", loc="left")

    link = sw["link strength"]; xs = [float(w) for w in link]
    pw = [link[w]["ReSPAIR"]["BH power"] * 100 for w in link]
    fd = [link[w]["ReSPAIR"]["BH empirical FDR"] * 100 for w in link]
    axes[1].plot(xs, pw, color=SERIES[0], lw=2, marker="o", ms=8, mec=SURF, mew=2)
    axes[1].plot(xs, fd, color=SERIES[1], lw=2, marker="o", ms=8, mec=SURF, mew=2)
    axes[1].text(xs[-1] + 0.01, pw[-1], "power", fontsize=9.5, va="center")
    axes[1].text(xs[-1] + 0.01, fd[-1], "empirical FDR", fontsize=9.5, va="center")
    for x, v in zip(xs, pw):
        axes[1].text(x, v + 4, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK2)
    axes[1].axhline(10, color=INK, lw=1, ls="--")
    axes[1].set_xlim(0.08, 0.6); axes[1].set_ylim(0, 110)
    axes[1].set_xlabel("link strength (change in log metabolite per SD of gene residual)")
    axes[1].set_ylabel("%")
    axes[1].set_title("ReSPAIR power vs link strength (BH 0.1, dashed = 10%)", loc="left")
    for ax in axes:
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_sweep.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run_sim import all_pvals
    d = simulate(seed=0)
    P, fit, (Yg, Ym) = all_pvals(d)
    fig1(d, fit, Yg, Ym)
    fig2()
    fig3(d, P)
    if os.path.exists(os.path.join(ROOT, "results", "sweep.json")):
        fig4()
