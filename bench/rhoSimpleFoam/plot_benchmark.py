#!/usr/bin/env python3
"""Render the rhoSimpleFoam benchmark as a static figure (PNG/SVG) for publication.

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
    ("aerofoilNACA0012",        1_024_000, [11.1,  44.6,  127.7, 132.7, 1495.2]),
    ("squareBend",              7_168_000, [50.9, 107.2, 1229.9, 1278.5, 1570.1]),
    ("squareBendLiq",           7_168_000, [47.6,  93.3, 1354.2, 1389.6,  619.4]),
    ("squareBendLiqNoNewtonian",7_168_000, [39.0,  68.0,  936.5,  972.2,  499.1]),
    ("injectorPipe",            2_576_294, [21.5,  33.8,  587.4,  580.2,  223.4]),
    ("angledDuct",              3_500_000, [23.0,  34.1,  445.4,  467.3,  217.5]),
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

def main(out="brae_benchmark"):
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 10,
                         "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK3, "ytick.color": INK3})
    fig = plt.figure(figsize=(13.2, 7.4), dpi=200, facecolor=SURF)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1], wspace=0.30,
                          left=0.155, right=0.975, top=0.795, bottom=0.115)

    # ---- panel 1: dot plot, seconds on a log axis -------------------------------------------
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(SURF)
    ys = list(range(len(PANEL)))[::-1]
    # DODGE each series onto its own sub-row. AMGX and PETSc land within 4% of one another on every
    # case -- on a log axis their marks coincide exactly and one hides the other. A fixed vertical
    # offset per series keeps all five readable without implying an ordering the data does not have.
    DODGE = [0.20, 0.10, 0.0, -0.10, -0.20]
    for y, (case, n, vals) in zip(ys, PANEL):
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=1.4, zorder=1, solid_capstyle="round")
        for (name, col), v, dy in zip(SERIES, vals, DODGE):
            ax.plot([v, v], [y, y + dy], color=GRID, lw=0.9, zorder=2)
            ax.plot([v], [y + dy], "o", ms=8.5, color=col, zorder=3,
                    markeredgecolor=SURF, markeredgewidth=1.6)
        ax.annotate(f"{vals[0]:.0f}s", (vals[0], y + DODGE[0]), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=INK, fontweight="bold", zorder=4)
        ax.annotate(f"{max(vals):.0f}s", (max(vals), y + DODGE[vals.index(max(vals))]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=INK3, zorder=4)
    ax.set_yticks(ys, [f"{c}\n{cells(n)} cells" for c, n, _ in PANEL], fontsize=9.5, color=INK)
    ax.set_xscale("log"); ax.set_xlim(7, 4200)
    ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000, 3000]))
    ax.set_xticklabels(["10s", "30s", "100s", "300s", "1000s", "3000s"], fontsize=9)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("wall time for 100 SIMPLE iterations  (log scale — lower is better)",
                  fontsize=9.5, labelpad=9)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Every arm, at the largest mesh all five completed",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=12)

    # ---- panel 2: brae vs OpenFOAM-64c across mesh size --------------------------------------
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(SURF)
    for case, pts in TREND.items():
        hero = case == "aerofoilNACA0012"
        xs = [p[0] for p in pts]; vs = [p[1] for p in pts]
        ax2.plot(xs, vs, "-o", lw=2.4 if hero else 1.5, ms=7 if hero else 5,
                 color="#2a78d6" if hero else INK3, alpha=1 if hero else .55,
                 markeredgecolor=SURF, markeredgewidth=1.4, zorder=3 if hero else 2)
        dy = {"squareBend": 7, "squareBendLiq": -1, "squareBendLiqNoNewtonian": -11,
              "angledDuct": 6, "injectorPipe": -7}.get(case, -3)
        ax2.annotate(f"{vs[-1]:.2f}×", (xs[-1], vs[-1]), textcoords="offset points",
                     xytext=(9, dy), fontsize=9 if hero else 8.2,
                     color="#2a78d6" if hero else INK3, fontweight="bold" if hero else "normal")
    ax2.axhline(1.0, color=INK3, lw=1.3, zorder=1)
    ax2.annotate("parity — below this line the CPU wins", (1.35e4, 0.70), fontsize=8.3, color=INK3)
    ax2.annotate("aerofoilNACA0012", (1.0e7, 8.46), textcoords="offset points", xytext=(-6, 14),
                 fontsize=9, color="#2a78d6", fontweight="bold", ha="right")
    ax2.set_xscale("log"); ax2.set_xlim(1.1e4, 7.0e7); ax2.set_ylim(0.6, 9.6)
    ax2.set_xlabel("cells  (log scale)", fontsize=9.5, labelpad=9)
    ax2.set_ylabel("brae speed-up over OpenFOAM on 64 cores", fontsize=9.5, labelpad=8)
    ax2.grid(color=GRID, lw=0.8, zorder=0); ax2.set_axisbelow(True)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    ax2.set_title("The lead grows with the mesh", fontsize=11.5, color=INK,
                  fontweight="bold", loc="left", pad=12)

    # ---- titles + legend ---------------------------------------------------------------------
    fig.text(0.155, 0.945, "brae — OpenFOAM's rhoSimpleFoam, re-ported to CUDA",
             fontsize=17, color=INK, fontweight="bold", ha="left")
    fig.text(0.155, 0.898,
             "Six tutorials, 15 thousand to 24 million cells, 100 fixed SIMPLE iterations.  "
             "One NVIDIA GH200 against all 64 Grace cores.",
             fontsize=10.3, color=INK2, ha="left")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c, markeredgecolor=SURF,
                          markeredgewidth=1.4, label=n) for n, c in SERIES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.155, 0.868), ncol=5,
               frameon=False, fontsize=9.5, handletextpad=0.45, columnspacing=1.9)
    fig.text(0.155, 0.030,
             "AMGX and PETSc offload only the pressure equation and run a serial host, so their columns "
             "measure that design, not the libraries.   brae's timer includes its whole start-up; "
             "OpenFOAM's excludes decomposePar.",
             fontsize=8.2, color=INK3, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", facecolor=SURF, bbox_inches="tight", pad_inches=0.32)
    print(f"wrote {out}.png and {out}.svg")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "brae_benchmark")
