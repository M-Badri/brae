#!/usr/bin/env python3
"""Render the rhoSimpleFoam benchmark as TWO standalone figures (PNG + SVG each).

Two separate images rather than one two-panel figure: the panels answer different questions and are
posted separately, and a shared canvas forces them to share a size and a legend they do not both need.

WHY A DOT PLOT AND NOT BARS. The measured walls span 11 s to 1570 s at a single mesh size, so a linear
bar chart is unreadable and a log-scale BAR is a lie -- a bar's length is read as proportional, and on a
log axis it is not. A dot plot carries the same magnitude on a log axis without implying proportional
length, which is the correct form for wide-range magnitude data.

COLOR. Five arms, categorical, assigned in fixed slot order and never cycled. The palette is the
validated default (blue, orange, aqua, yellow, magenta); it clears the lightness band, chroma floor,
adjacent CVD separation and normal-vision floor in BOTH modes. Three light-mode slots sit below 3:1
contrast on the light surface, so the relief rule applies and every mark carries a direct value label.
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

SERIES = [("brae", "#2a78d6"), ("OpenFOAM 64c", "#eb6834"), ("AMGX", "#1baf7a"),
          ("PETSc", "#eda100"), ("spuma", "#e87ba4")]
INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#86847d", "#e6e4de", "#fcfcfb"

# largest mesh where ALL FIVE arms completed
PANEL = [
    ("aerofoilNACA0012",        1_024_000, [11.1,  44.6,  127.7, 132.7, 1495.2],  4.3),
    ("squareBend",              7_168_000, [50.9, 107.2, 1229.9, 1278.5, 1570.1], 10.1),
    ("squareBendLiq",           7_168_000, [47.6,  93.3, 1354.2, 1389.6,  619.4],  8.9),
    ("squareBendLiqNoNewtonian",7_168_000, [39.0,  68.0,  936.5,  972.2,  499.1], 10.2),
    ("injectorPipe",            2_576_294, [21.5,  33.8,  587.4,  580.2,  223.4],  5.2),
    ("angledDuct",              3_500_000, [23.0,  34.1,  445.4,  467.3,  217.5],  5.0),
]
# brae vs OpenFOAM-64c across the ladder
TREND = {
    "aerofoilNACA0012":        [(14938,1.99),(64000,2.38),(1024000,4.01),(10000000,8.46)],
    "squareBend":              [(14200,1.48),(896000,1.35),(7168000,2.11)],
    "squareBendLiq":           [(112000,0.98),(896000,1.23),(7168000,1.96),(24192000,2.02)],
    "squareBendLiqNoNewtonian":[(112000,1.01),(896000,1.16),(7168000,1.74),(24192000,1.85)],
    "angledDuct":              [(28000,1.21),(437500,0.96),(3500000,1.48)],
    "injectorPipe":            [(74650,1.09),(573858,1.09),(2576294,1.57)],
}

def cells(n):
    return f"{n/1e6:.1f}M".replace(".0M", "M") if n >= 1e6 else f"{n//1000}k"

# brae WARM against the same OpenFOAM wall. NOT like-for-like: OpenFOAM's timer excludes decomposePar
# but still carries MPI start-up and field reads, so these are an UPPER BOUND on brae's advantage until
# OpenFOAM is measured warm too. Plotted because the gap between the two brae curves is the real subject.
TREND_WARM = {
    "aerofoilNACA0012":        [(14938,4.00),(64000,7.00),(1024000,10.37),(10000000,23.03)],
    "squareBend":              [(14200,4.00),(896000,4.52),(7168000,10.61)],
    "squareBendLiq":           [(112000,4.25),(896000,5.60),(7168000,10.48)],
    "squareBendLiqNoNewtonian":[(112000,3.00),(896000,4.77),(7168000,6.67)],
    "angledDuct":              [(28000,3.00),(437500,3.78),(3500000,6.82)],
    "injectorPipe":            [(74650,3.00),(573858,3.67),(2576294,6.50)],
}

def _frame(fig, sub):
    fig.text(0.012, 0.955, "brae - rhoSimpleFoam, re-ported to CUDA",
             fontsize=17, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.893, sub, fontsize=10.3, color=INK2, ha="left", va="top")


def fig_arms(out):
    """Every arm at the largest mesh all five completed."""
    fig = plt.figure(figsize=(11.6, 7.2), dpi=200, facecolor=SURF)
    ax = fig.add_axes([0.235, 0.135, 0.745, 0.615])          # room above for title + legend
    ax.set_facecolor(SURF)
    ys = list(range(len(PANEL)))[::-1]
    DODGE = [0.20, 0.10, 0.0, -0.10, -0.20]
    threads = [[] for _ in SERIES]
    warm_xy = []
    for y, (case, n, vals, warm) in zip(ys, PANEL):
        ax.plot([min(min(vals), warm), max(vals)], [y, y], color=GRID, lw=1.4, zorder=1,
                solid_capstyle="round")
        ax.plot([warm, vals[0]], [y + DODGE[0]] * 2, color=SERIES[0][1], lw=2.0, alpha=.30, zorder=2)
        ax.plot([warm], [y + DODGE[0]], "o", ms=8.0, color=SURF, zorder=4,
                markeredgecolor=SERIES[0][1], markeredgewidth=2.0)
        warm_xy.append((warm, y + DODGE[0]))
        for i, ((name, col), v, dy) in enumerate(zip(SERIES, vals, DODGE)):
            ax.plot([v, v], [y, y + dy], color=GRID, lw=0.9, zorder=2)
            ax.plot([v], [y + dy], "o", ms=8.5, color=col, zorder=4,
                    markeredgecolor=SURF, markeredgewidth=1.6)
            threads[i].append((v, y + dy))
        ax.annotate(f"{vals[0]:.0f}s", (vals[0], y + DODGE[0]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color=INK, fontweight="bold", zorder=5)
        ax.annotate(f"{warm:.0f}s", (warm, y + DODGE[0]), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=SERIES[0][1], zorder=5)
        # the slowest arm carries a label too: aqua, yellow and magenta all sit below 3:1 contrast on
        # this surface, and the validator's relief rule wants them labelled, not left to colour alone.
        slow = max(vals)
        ax.annotate(f"{slow:.0f}s", (slow, y + DODGE[vals.index(slow)]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.2, color=INK3, zorder=5)
    # a thin thread per arm, so each reads as one series across the six cases
    for (name, col), pts in zip(SERIES, threads):
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=col, lw=1.0, alpha=.45, zorder=3)
    ax.plot([x for x, _ in warm_xy], [y for _, y in warm_xy], color=SERIES[0][1], lw=1.0,
            alpha=.30, ls=(0, (3, 2)), zorder=3)

    ax.set_yticks(ys, [f"{c}\n{cells(n)} cells" for c, n, _, _ in PANEL], fontsize=9.5, color=INK)
    ax.set_xscale("log"); ax.set_xlim(3.2, 4200)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 30, 100, 300, 1000, 3000]))
    ax.set_xticklabels(["5s", "10s", "30s", "100s", "300s", "1000s", "3000s"], fontsize=9)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("wall time for 100 SIMPLE iterations  (log scale, lower is better)",
                  fontsize=9.5, labelpad=9)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    _frame(fig, "15k to 24M cells, 100 fixed SIMPLE iterations, on a GH200 against all 64 Grace cores.")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c, markeredgecolor=SURF,
                          markeredgewidth=1.4, label=n) for n, c in SERIES]
    handles.insert(1, plt.Line2D([], [], marker="o", ls="", ms=8, color=SURF,
                                 markeredgecolor=SERIES[0][1], markeredgewidth=2.0, label="brae warm"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.845), ncol=6,
               frameon=False, fontsize=9.5, handletextpad=0.45, columnspacing=1.9)
    fig.text(0.012, 0.028,
             "Hollow marks are brae warm (iterations 101-200 alone); the bar to the filled mark is its "
             "start-up.  brae's timer includes that start-up, OpenFOAM's excludes decomposePar.",
             fontsize=8.2, color=INK3, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", facecolor=SURF, bbox_inches="tight", pad_inches=0.32)
    plt.close(fig)
    print(f"wrote {out}.png / .svg")


def fig_scaling(out, case="aerofoilNACA0012"):
    """One case, three curves, absolute seconds.

    The six-case ratio chart carried twelve lines and needed a paragraph to read. This shows ONE
    tutorial with the CPU as a real curve rather than an invisible baseline at 1x -- the reader sees
    three lines and the gaps between them are the whole story.
    """
    # cells, brae whole run, brae warm, OpenFOAM on 64 cores
    RUNS = {
        "aerofoilNACA0012": [(14938, 0.80, 0.44, 1.59), (64000, 1.49, 0.52, 3.55),
                             (1024000, 11.13, 4.31, 44.62), (10000000, 108.86, 40.00, 920.95)],
        "squareBend":       [(14200, 0.79, 0.34, 1.17), (896000, 7.03, 2.07, 9.50),
                             (7168000, 50.94, 10.13, 107.23)],
    }[case]
    xs   = [r[0] for r in RUNS]
    cold = [r[1] for r in RUNS]; warm = [r[2] for r in RUNS]; cpu = [r[3] for r in RUNS]
    fig = plt.figure(figsize=(9.6, 7.0), dpi=200, facecolor=SURF)
    ax = fig.add_axes([0.115, 0.135, 0.845, 0.635]); ax.set_facecolor(SURF)

    ax.plot(xs, cpu, "-o", lw=2.6, ms=8, color=SERIES[1][1], markeredgecolor=SURF,
            markeredgewidth=1.6, zorder=3, label="OpenFOAM, 64 Grace cores")
    ax.plot(xs, cold, "-o", lw=2.6, ms=8, color=SERIES[0][1], markeredgecolor=SURF,
            markeredgewidth=1.6, zorder=4, label="brae, one GPU (whole run)")
    ax.plot(xs, warm, "--o", lw=2.2, ms=7, color=SERIES[0][1], alpha=.62, markerfacecolor=SURF,
            markeredgecolor=SERIES[0][1], markeredgewidth=1.8, zorder=4,
            label="brae, one GPU (warm, iterations 101-200)")

    for x, c, w, p_ in RUNS:
        ax.annotate(f"{p_:.0f}s" if p_ >= 10 else f"{p_:.1f}s", (x, p_), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8.6, color=SERIES[1][1], fontweight="bold")
        ax.annotate(f"{c:.0f}s" if c >= 10 else f"{c:.1f}s", (x, c), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8.6, color=SERIES[0][1], fontweight="bold")
        ax.annotate(f"{w:.0f}s" if w >= 10 else f"{w:.1f}s", (x, w), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=8.2, color=SERIES[0][1], alpha=.85)
    # the headline gap, stated once where it is widest
    xl, cl, wl, pl = RUNS[-1]
    ax.annotate(f"{pl/cl:.1f}x faster", (xl, (cl*pl) ** 0.5), textcoords="offset points",
                xytext=(16, -4), fontsize=11, color=INK, fontweight="bold")
    ax.annotate(f"{pl/wl:.0f}x warm", (xl, (wl*cl) ** 0.5), textcoords="offset points",
                xytext=(16, -4), fontsize=9, color=SERIES[0][1], alpha=.85)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.1e4, 3.2e7); ax.set_ylim(0.3, 2600)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 3, 10, 30, 100, 300, 1000]))
    ax.set_yticklabels(["0.5s", "1s", "3s", "10s", "30s", "100s", "300s", "1000s"], fontsize=9)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("cells  (log scale)", fontsize=9.5, labelpad=9)
    ax.set_ylabel("wall time for 100 SIMPLE iterations  (log scale, lower is better)",
                  fontsize=9.5, labelpad=8)
    ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, handletextpad=0.7, labelspacing=0.55)

    _frame(fig, f"{case}, 15k to 10M cells.  One GH200 against all 64 Grace cores.")
    fig.text(0.012, 0.028,
             "brae's timer includes its whole start-up; OpenFOAM's excludes decomposePar.  The dashed "
             "curve is brae's solve alone, so it is an upper bound on the advantage.",
             fontsize=8.2, color=INK3, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", facecolor=SURF, bbox_inches="tight", pad_inches=0.32)
    plt.close(fig)
    print(f"wrote {out}.png / .svg")


def main(out="brae_benchmark"):
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 10,
                         "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK3, "ytick.color": INK3})
    fig_arms(f"{out}_arms")
    fig_scaling(f"{out}_scaling")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "brae_benchmark")
