"""Generate Figure 2: ReviewGuard Framework.

This script redraws the paper's ReviewGuard framework diagram using fixed
matplotlib coordinates. It exports a vector PDF, high-resolution PNG, and
editable SVG for direct use in LaTeX/Overleaf.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "final_pricai"
OUT_PDF = OUT_DIR / "reviewguard_workflow_flow_only.pdf"
OUT_PNG = OUT_DIR / "reviewguard_workflow_flow_only.png"
OUT_SVG = OUT_DIR / "reviewguard_workflow_flow_only.svg"

CANVAS_W = 1800
CANVAS_H = 850
FIGSIZE = (18, 8.5)
DPI = 360
MIN_FONT_SIZE = 9.0
FONT_SCALE = 1.12
FONT_SIZE_DELTA = 4.0
BOX_EXPAND = 2.0

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "dejavusans"


COLORS = {
    "text": "#1F2328",
    "muted": "#5D6570",
    "stage": "#F8FAFC",
    "stage_edge": "#AEB6C2",
    "input_fill": "#F2F3F5",
    "input_edge": "#5D6570",
    "blue_fill": "#EAF2FF",
    "blue": "#174A9C",
    "orange_fill": "#FFF1E8",
    "orange": "#E65A1A",
    "green_fill": "#EAF8EE",
    "green": "#1F8A43",
    "fusion_fill": "#F4F7FF",
    "fusion": "#0F3D8A",
    "purple_fill": "#F5EAFF",
    "purple": "#8846B4",
    "yellow_fill": "#FFF8D8",
    "yellow": "#C28B00",
    "output_fill": "#EAF8EE",
    "output": "#1F8A43",
    "white": "#FFFFFF",
}


@dataclass(frozen=True)
class BoxSpec:
    name: str
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


BOXES: list[BoxSpec] = []
TEXT_AREAS: list[BoxSpec] = []
FONT_SIZES: list[float] = []
ARROW_ROUTES: list[tuple[str, list[tuple[float, float]]]] = []


def register_box(name: str, x: float, y: float, w: float, h: float) -> BoxSpec:
    box = BoxSpec(name, x, y, w, h)
    BOXES.append(box)
    return box


def draw_box(
    ax,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str,
    lw: float = 1.8,
    radius: float = 14,
    linestyle: str | tuple = "-",
    z: int = 1,
) -> BoxSpec:
    x -= BOX_EXPAND
    y -= BOX_EXPAND
    w += 2 * BOX_EXPAND
    h += 2 * BOX_EXPAND
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fill,
        edgecolor=stroke,
        linewidth=lw,
        linestyle=linestyle,
        zorder=z,
    )
    ax.add_patch(patch)
    return register_box(name, x, y, w, h)


def wrap_text(text: str, width: int | None) -> str:
    if width is None:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def draw_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 10.0,
    color: str = COLORS["text"],
    weight: str = "normal",
    ha: str = "left",
    va: str = "top",
    width: int | None = None,
    line_spacing: float = 1.12,
    z: int = 8,
    area_name: str | None = None,
    area_w: float | None = None,
    area_h: float | None = None,
):
    effective_size = size * FONT_SCALE + FONT_SIZE_DELTA
    if effective_size < MIN_FONT_SIZE:
        raise ValueError(f"Font size below minimum: {effective_size} for text {text!r}")
    FONT_SIZES.append(effective_size)
    rendered = wrap_text(text, width)
    ax.text(
        x,
        y,
        rendered,
        fontsize=effective_size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=line_spacing,
        zorder=z,
    )
    if area_name and area_w and area_h:
        TEXT_AREAS.append(BoxSpec(area_name, x, y, area_w, area_h))


def draw_stage(
    ax,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    title_x_offset: float = 24,
    title_y_offset: float = 18,
) -> BoxSpec:
    box = draw_box(
        ax,
        name,
        x,
        y,
        w,
        h,
        fill=COLORS["stage"],
        stroke=COLORS["stage_edge"],
        lw=1.3,
        radius=15,
        z=0,
    )
    draw_text(ax, x + title_x_offset, y + title_y_offset, title, size=17, weight="bold", color="#343941")
    return box


def draw_chip(
    ax,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    fill: str,
    stroke: str,
    size: float = 10.0,
) -> BoxSpec:
    box = draw_box(ax, name, x, y, w, h, fill=fill, stroke=stroke, lw=1.25, radius=8, z=3)
    draw_text(
        ax,
        x + w / 2,
        y + h / 2,
        label,
        size=size,
        color=stroke,
        weight="bold",
        ha="center",
        va="center",
        area_name=f"text:{name}",
        area_w=w - 12,
        area_h=h - 8,
    )
    return box


def draw_arrow(
    ax,
    name: str,
    points: list[tuple[float, float]],
    *,
    color: str = COLORS["fusion"],
    lw: float = 2.2,
    dashed: bool = False,
    arrow_size: float = 15,
    z: int = 5,
):
    if len(points) < 2:
        raise ValueError("Arrow needs at least two points")
    ARROW_ROUTES.append((name, points))
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            linewidth=lw,
            linestyle=(0, (6, 4)) if dashed else "-",
            solid_capstyle="round",
            zorder=z,
        )
    p0, p1 = points[-2], points[-1]
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=arrow_size,
            linewidth=lw,
            linestyle=(0, (6, 4)) if dashed else "-",
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )

def draw_stacked_cards(
    ax,
    x: float,
    y: float,
    fields: list[str],
    *,
    card_w: float = 145,
    card_h: float = 25,
    gap: float = 30,
) -> None:
    for idx, field in enumerate(fields):
        yy = y + idx * gap

        ax.add_patch(
            Rectangle(
                (x, yy),
                card_w,
                card_h,
                facecolor=COLORS["white"],
                edgecolor="#B8BEC7",
                linewidth=1.0,
                zorder=4,
            )
        )

        draw_text(
            ax,
            x + card_w / 2,
            yy + card_h / 2,
            field,
            size=10.0,
            va="center",
            ha="center",
            color=COLORS["text"],
        )

def draw_table(ax, name: str, x: float, y: float, w: float, h: float) -> BoxSpec:
    box = draw_box(ax, name, x, y, w, h, fill=COLORS["white"], stroke=COLORS["fusion"], lw=1.7, radius=11, z=2)
    draw_text(ax, x + 16, y + 16, "Weighted review table", size=12.5, weight="bold", color=COLORS["fusion"])
    table_x, table_y = x + 20, y + 48
    col_w = [94, 132, 86]
    row_h = 26
    headers = ["review", "sentiment", "weight"]
    rows = [["r1", "neg", "low"], ["r2", "pos", "high"]]
    for c, header in enumerate(headers):
        ax.add_patch(Rectangle((table_x + sum(col_w[:c]), table_y), col_w[c], row_h, facecolor=COLORS["fusion_fill"], edgecolor="#B8C6E6", linewidth=0.9, zorder=3))
        draw_text(ax, table_x + sum(col_w[:c]) + col_w[c] / 2, table_y + row_h / 2, header, size=9.0, weight="bold", color=COLORS["fusion"], ha="center", va="center")
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            yy = table_y + row_h * (r + 1)
            ax.add_patch(Rectangle((table_x + sum(col_w[:c]), yy), col_w[c], row_h, facecolor=COLORS["white"], edgecolor="#D6DCE6", linewidth=0.8, zorder=3))
            color = COLORS["orange"] if val == "low" else COLORS["green"] if val == "high" else COLORS["text"]
            draw_text(ax, table_x + sum(col_w[:c]) + col_w[c] / 2, yy + row_h / 2, val, size=9.0, color=color, weight="bold" if val in {"low", "high"} else "normal", ha="center", va="center")
    return box


def draw_stacked_bar(ax, x: float, y: float, w: float, h: float) -> None:
    parts = [
        ("neg", 0.31, "#D9534F"),
        ("neu", 0.22, "#ADB5BD"),
        ("pos", 0.47, "#3C7DB1"),
    ]
    cur = x
    for label, frac, color in parts:
        seg_w = w * frac
        ax.add_patch(Rectangle((cur, y), seg_w, h, facecolor=color, edgecolor="white", linewidth=1.1, zorder=4))
        draw_text(ax, cur + seg_w / 2, y + h / 2, label, size=9.0, color="white", weight="bold", ha="center", va="center")
        cur += seg_w


def draw_weight_effect(ax, x: float, y: float, w: float, h: float) -> BoxSpec:
    box = draw_box(ax, "weight_effect", x, y, w, h, fill=COLORS["white"], stroke=COLORS["stage_edge"], lw=1.4, radius=12, z=2)
    draw_text(ax, x + 18, y + 16, "Weight effect", size=12.0, weight="bold", color=COLORS["muted"])
    draw_text(ax, x + 18, y + 56, "ordinary review", size=9.5, color=COLORS["text"])
    ax.plot([x + 188, x + 280], [y + 62, y + 62], color=COLORS["green"], linewidth=7.0, solid_capstyle="round", zorder=5)
    draw_text(ax, x + 294, y + 62, "high", size=9.0, color=COLORS["green"], weight="bold", va="center")
    draw_text(ax, x + 18, y + 92, "coordinated spam", size=9.5, color=COLORS["text"])
    ax.plot([x + 208, x + 272], [y + 98, y + 98], color=COLORS["orange"], linewidth=3.0, solid_capstyle="round", zorder=5)
    draw_text(ax, x + 288, y + 98, "low", size=9.0, color=COLORS["orange"], weight="bold", va="center")
    return box


def draw_output_cards(ax, x: float, y: float) -> BoxSpec:
    outer = draw_box(ax, "outputs", x, y, 255, 145, fill=COLORS["output_fill"], stroke=COLORS["output"], lw=1.7, radius=13, z=2)
    draw_text(ax, x + 18, y + 16, "Outputs", size=12.5, weight="bold", color=COLORS["output"])
    labels = ["Recovered proportions", "Weighted review table", "Inspection record"]
    for idx, label in enumerate(labels):
        yy = y + 46 + idx * 30
        draw_box(ax, f"output_card_{idx}", x + 18, yy, 218, 24, fill=COLORS["white"], stroke="#B8D8C0", lw=0.9, radius=6, z=3)
        draw_text(ax, x + 127, yy + 12, label, size=9.0, color=COLORS["text"], ha="center", va="center")
    return outer


def check_layout() -> None:
    errors: list[str] = []
    for box in BOXES:
        if box.left < 0 or box.top < 0 or box.right > CANVAS_W or box.bottom > CANVAS_H:
            errors.append(f"Box outside canvas: {box}")
    if min(FONT_SIZES or [MIN_FONT_SIZE]) < MIN_FONT_SIZE:
        errors.append("A font below the minimum size was used")
    # Manual safe-routing checks: arrows are routed through margins/gaps or between
    # box centers; none are intended to cross dense text regions.
    for name, points in ARROW_ROUTES:
        for x, y in points:
            if not (0 <= x <= CANVAS_W and 0 <= y <= CANVAS_H):
                errors.append(f"Arrow point outside canvas: {name} {x},{y}")
    if errors:
        raise RuntimeError("\n".join(errors))


def build_figure():
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.axis("off")

    # 1. Title area.
    draw_text(ax, 36, 26, "ReviewGuard Framework", size=22.0, weight="bold")
    draw_text(
        ax,
        36,
        58,
        "Reliability-weighted sentiment proportion recovery from app-review streams",
        size=12.0,
        color=COLORS["muted"],
    )

    # 2. Stage 1.
    draw_stage(
        ax,
        "stage1",
        28,
        100,
        1762,
        335,
        "Stage 1. Evidence construction from raw reviews",
        title_x_offset=42,
        title_y_offset=12,
    )

    input_box = draw_box(
        ax,
        "input_stream",
        70,
        158,
        205,
        210,
        fill=COLORS["input_fill"],
        stroke=COLORS["input_edge"],
        lw=1.8,
    )

    draw_text(
        ax,
        92,
        184,
        "Input review\nstream",
        size=12.2,
        weight="bold",
        color=COLORS["input_edge"],
        line_spacing=1.05,
    )

    draw_stacked_cards(
        ax,
        100,
        244,
        ["text", "star rating", "timestamp", "app version"],
        card_w=145,
        card_h=25,
        gap=30,
    )

    pre_box = draw_box(ax, "preprocessing", 310, 150, 345, 220, fill=COLORS["blue_fill"], stroke=COLORS["blue"], lw=1.9)
    draw_text(ax, 342, 174, "Preprocessing", size=15.0, weight="bold", color=COLORS["blue"])
    chip_w, chip_h = 142, 34
    draw_chip(ax, "clean_text", 322, 222, chip_w, chip_h, "clean text", fill=COLORS["white"], stroke=COLORS["blue"])
    draw_chip(ax, "rating_label", 482, 222, chip_w + 18, chip_h, "rating label", fill=COLORS["white"], stroke=COLORS["blue"])
    draw_chip(ax, "time_bucket", 322, 270, chip_w, chip_h, "time bucket", fill=COLORS["white"], stroke=COLORS["blue"])
    draw_chip(ax, "version_attach", 482, 270, chip_w + 18, chip_h, "version attach", fill=COLORS["white"], stroke=COLORS["blue"], size=9.5)
    draw_chip(ax, "feature_table", 395, 325, 155, 30, "feature table", fill=COLORS["white"], stroke=COLORS["blue"], size=9.5)

    group_box = draw_box(ax, "group_reliability", 670, 140, 620, 160, fill=COLORS["orange_fill"], stroke=COLORS["orange"], lw=1.9)
    draw_text(ax, 710, 162, "Group reliability", size=14.5, weight="bold", color=COLORS["orange"])
    draw_text(ax, 710, 188, "coordinated-pattern evidence", size=10.5, color=COLORS["muted"])
    evidence = [
        ("Time burst", 688, 220),
        ("Duplicate text", 878, 220),
        ("Low-star cluster", 1086, 220),
        ("Version conc.", 688, 260),
        ("Semantic sim.", 878, 260),
        ("Cohort size", 1086, 260),
    ]
    for label, x, y in evidence:
        chip_width = 190 if label in {"Duplicate text", "Low-star cluster", "Semantic sim.", "Cohort size"} else 160
        draw_chip(ax, f"group_{label}", x, y, chip_width, 30, label, fill=COLORS["white"], stroke=COLORS["orange"], size=9.1)

    loss_box = draw_box(ax, "loss_reliability", 710, 318, 510, 100, fill=COLORS["green_fill"], stroke=COLORS["green"], lw=1.9)
    draw_text(ax, 735, 336, "Loss reliability", size=14.0, weight="bold", color=COLORS["green"])
    draw_text(ax, 735, 360, "text-rating consistency", size=10.3, color=COLORS["muted"])
    draw_chip(ax, "oof_prediction", 724, 382, 165, 28, "OOF prediction", fill=COLORS["white"], stroke=COLORS["green"], size=9.0)
    draw_chip(ax, "self_loss", 914, 382, 112, 28, "self-loss", fill=COLORS["white"], stroke=COLORS["green"], size=9.0)
    draw_chip(ax, "loss_mixture", 1071, 382, 138, 28, "loss mixture", fill=COLORS["white"], stroke=COLORS["green"], size=9.0)

    fusion_box = draw_box(ax, "fusion", 1310, 210, 305, 166, fill=COLORS["fusion_fill"], stroke=COLORS["fusion"], lw=1.9)
    draw_text(ax, 1330, 234, "Reliability fusion", size=14.0, weight="bold", color=COLORS["fusion"])
    draw_text(ax, 1332, 272, "combine cohort evidence\nand consistency evidence\nassign review weight", size=9.7, color=COLORS["text"], line_spacing=1.14)
    draw_chip(ax, "normal_high", 1324, 342, 136, 24, "normal high", fill=COLORS["white"], stroke=COLORS["green"], size=9.0)
    draw_chip(ax, "sus_low", 1472, 342, 136, 24, "suspect low", fill=COLORS["white"], stroke=COLORS["orange"], size=9.0)

    key_box = draw_box(ax, "key_idea", 1636, 238, 140, 150, fill=COLORS["yellow_fill"], stroke=COLORS["yellow"], lw=1.35, radius=12, linestyle=(0, (5, 4)))
    draw_text(ax, 1648, 258, "Key idea", size=11.2, color="#8A6200", weight="bold")
    draw_text(ax, 1648, 292, "Keep all\nreviews.\nReduce\nsuspicious\ninfluence.", size=9.5, color=COLORS["text"], line_spacing=0.95)

    # Stage 1 arrows.
    draw_arrow(ax, "input_to_pre", [(275, 263), (310, 263)], color=COLORS["input_edge"])
    draw_arrow(ax, "pre_to_group", [(655, 248), (670, 220)], color=COLORS["blue"])
    draw_arrow(ax, "pre_to_loss", [(655, 315), (710, 368)], color=COLORS["blue"])
    draw_arrow(ax, "oof_to_loss", [(889, 396), (914, 396)], color=COLORS["green"], lw=1.8, arrow_size=12)
    draw_arrow(ax, "self_to_mix", [(1026, 396), (1071, 396)], color=COLORS["green"], lw=1.8, arrow_size=12)
    draw_arrow(ax, "group_to_fusion", [(1290, 232), (1310, 272)], color=COLORS["orange"])
    draw_arrow(ax, "loss_to_fusion", [(1220, 368), (1310, 316)], color=COLORS["green"])

    # 3. Stage 2.
    draw_stage(ax, "stage2", 28, 462, 1762, 306, "Stage 2. Weighting, calibration, and recovery", title_x_offset=42)

    table_box = draw_table(ax, "weighted_table", 78, 560, 355, 130)

    base_box = draw_box(ax, "base_mode", 450, 516, 180, 84, fill=COLORS["fusion_fill"], stroke=COLORS["fusion"], lw=1.6, radius=12)
    draw_text(ax, 473, 530, "Base mode", size=12.2, color=COLORS["fusion"], weight="bold")
    draw_text(ax, 473, 556, "use learned\nweights", size=10.0, color=COLORS["text"], line_spacing=1.08)

    verified_box = draw_box(ax, "verified_sample", 454, 610, 248, 32, fill=COLORS["white"], stroke=COLORS["purple"], lw=1.35, radius=9, linestyle=(0, (5, 4)))
    draw_text(ax, 578, 626, "verified clean sample", size=9.4, color=COLORS["purple"], weight="bold", ha="center", va="center")

    cal_box = draw_box(ax, "calibration_mode", 450, 665, 250, 70, fill=COLORS["purple_fill"], stroke=COLORS["purple"], lw=1.6, radius=12)
    draw_text(ax, 464, 676, "Calibration mode", size=11.8, color=COLORS["purple"], weight="bold")
    draw_text(ax, 464, 705, "calibrated weights", size=9.4, color=COLORS["text"])

    agg_box = draw_box(ax, "weighted_aggregation", 760, 554, 340, 130, fill=COLORS["fusion_fill"], stroke=COLORS["fusion"], lw=1.7, radius=13)
    draw_text(ax, 783, 576, "Weighted aggregation", size=12.2, color=COLORS["fusion"], weight="bold")
    draw_text(ax, 783, 608, "recovered proportions", size=10.0, color=COLORS["muted"])
    draw_stacked_bar(ax, 787, 642, 205, 24)

    outputs_box = draw_output_cards(ax, 1150, 532)
    effect_box = draw_weight_effect(ax, 1438, 540, 340, 126)

    # Stage 2 arrows. The wrap arrow from fusion to weighted table runs through
    # the empty inter-stage gap to avoid crossing text.
    draw_arrow(ax, "fusion_to_table", [(1452, 376), (1452, 448), (55, 448), (55, 625), (78, 625)], color=COLORS["fusion"], lw=2.1)
    draw_arrow(ax, "table_to_base", [(433, 602), (450, 556)], color=COLORS["fusion"], lw=2.0)
    draw_arrow(ax, "table_to_cal", [(433, 660), (450, 700)], color=COLORS["purple"], lw=2.0)
    draw_arrow(ax, "verified_to_cal", [(578, 642), (578, 665)], color=COLORS["purple"], dashed=True, lw=1.8, arrow_size=12)
    draw_arrow(ax, "base_to_agg", [(630, 556), (760, 596)], color=COLORS["fusion"], lw=2.0)
    draw_arrow(ax, "cal_to_agg", [(700, 700), (760, 646)], color=COLORS["purple"], lw=2.0)
    draw_arrow(ax, "agg_to_outputs", [(1100, 619), (1150, 604)], color=COLORS["output"], lw=2.1)

    # 4. Legend area.
    legend_y = 798
    draw_text(ax, 64, legend_y, "Legend:", size=10.0, color=COLORS["muted"], weight="bold", va="center")
    legend = [
        (COLORS["blue"], "blue = preprocessing"),
        (COLORS["orange"], "orange = cohort evidence"),
        (COLORS["green"], "green = consistency evidence"),
        (COLORS["purple"], "purple = optional calibration"),
    ]
    lx = 165
    for color, label in legend:
        ax.add_patch(Rectangle((lx, legend_y - 8), 20, 16, facecolor=color, edgecolor=color, zorder=5))
        draw_text(ax, lx + 30, legend_y, label, size=9.0, color=COLORS["muted"], va="center")
        lx += 330
    legend_y2 = 834
    lx = 420
    ax.plot([lx, lx + 38], [legend_y2, legend_y2], color=COLORS["fusion"], linewidth=2.3, zorder=5)
    draw_text(ax, lx + 48, legend_y2, "solid arrow = main flow", size=9.0, color=COLORS["muted"], va="center")
    lx += 440
    ax.plot([lx, lx + 38], [legend_y2, legend_y2], color=COLORS["purple"], linewidth=2.0, linestyle=(0, (5, 4)), zorder=5)
    draw_text(ax, lx + 48, legend_y2, "dashed arrow = optional input", size=9.0, color=COLORS["muted"], va="center")

    check_layout()
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT_PDF, format="pdf", bbox_inches=None, pad_inches=0, facecolor="white")
    fig.savefig(OUT_SVG, format="svg", bbox_inches=None, pad_inches=0, facecolor="white")
    fig.savefig(OUT_PNG, format="png", dpi=DPI, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
