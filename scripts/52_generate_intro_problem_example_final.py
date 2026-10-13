from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [
    ROOT / "final_pricai" / "intro_problem_baseline_example.pdf",
    ROOT / "scripts" / "intro_problem_baseline_example.pdf",
    ROOT / "6.27_pricai" / "intro_problem_baseline_example.pdf",
]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"


NEG = "#C84B4B"
NEU = "#A5ABB3"
POS = "#3B78A8"
GREEN = "#228543"
GRAY = "#606060"


def rounded(ax, x, y, w, h, fc, ec, lw=1.6, radius=12, ls="-"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, s, size=10, color="#222222", weight="normal", ha="left", va="top", width=None):
    if width is not None:
        s = "\n".join(textwrap.wrap(s, width=width))
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)


def stacked_bar(ax, x, y, w, h, counts):
    total = float(sum(counts))
    left = x
    for count, color in zip(counts, [NEG, NEU, POS]):
        ww = w * count / total if total else 0
        ax.add_patch(Rectangle((left, y), ww, h, facecolor=color, edgecolor="white", linewidth=0.8))
        left += ww


def generate_fig1(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.25))
    ax.set_xlim(0, 1000)
    ax.set_ylim(425, 0)
    ax.axis("off")

    bar_x = 330
    bar_w = 470
    bar_h = 30
    row_gap = 78
    rows = [
        (92, "Reference", "100 reviews: 25 / 5 / 70", [25, 5, 70], "Neg. 25.0%", GRAY, "#F8F8F8"),
        (92 + row_gap, "After attack / no defense", "120 reviews: 45 / 5 / 70", [45, 5, 70], "Neg. 37.5%\n(+12.5 pp)", NEG, "#FFF7F7"),
        (92 + 2 * row_gap, "ReviewGuard weighted", "downweight suspicious cohort", [25, 5, 70], "Near ref. 25/5/70", GREEN, "#F2FBF4"),
    ]

    ref_x = bar_x + 0.25 * bar_w
    ax.plot([ref_x, ref_x], [73, 292], color="#333333", linewidth=1.2, linestyle=(0, (4, 3)))
    label(ax, ref_x - 78, 58, "25% negative reference", 8.2, color="#333333")

    def distribution_row(y, title, subtitle, counts, right_text, color, fill):
        rounded(ax, 22, y - 18, 940, 56, fill, color, lw=1.35, radius=8)
        label(ax, 43, y - 3, title, 9.8, color=color, weight="bold")
        label(ax, 43, y + 18, subtitle, 8.3, color="#333333")
        stacked_bar(ax, bar_x, y, bar_w, bar_h, counts)
        total = sum(counts)
        neg_w = bar_w * counts[0] / total
        neg_share = counts[0] / total
        if neg_share >= 0.3:
            pct_label = f"{counts[0] / total * 100:.1f}% neg"
        else:
            pct_label = f"{counts[0] / total * 100:.0f}% neg"
        pct_x = bar_x + neg_w / 2 - (26 if neg_share >= 0.3 else 0)
        label(ax, pct_x, y + 12, pct_label, 8.0, color="white", weight="bold", ha="center", va="center")
        label(ax, 822, y + 1, right_text, 7.9, color=color, weight="bold")

    for row in rows:
        distribution_row(*row)

    rounded(ax, 74, 322, 395, 52, "#F6F6F6", "#777777", lw=1.35, radius=8, ls=(0, (4, 3)))
    label(ax, 97, 338, "Hard deletion baseline", 9.5, color="#555555", weight="bold")
    label(ax, 97, 358, "may remove legitimate complaints", 8.3, color="#555555")

    rounded(ax, 525, 322, 405, 52, "#F2FBF4", GREEN, lw=1.45, radius=8)
    label(ax, 548, 338, "ReviewGuard weighting", 9.5, color=GREEN, weight="bold")
    label(ax, 548, 358, "keeps records but reduces suspicious contribution", 8.3, color="#333333")

    legend_y = 407
    for x, name, color in [(330, "Negative", NEG), (455, "Neutral", NEU), (570, "Positive", POS)]:
        ax.add_patch(Rectangle((x, legend_y - 13), 22, 10, facecolor=color, edgecolor="none"))
        label(ax, x + 30, legend_y - 17, name, 8.9, va="top")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    for output in OUTPUTS:
        generate_fig1(output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
