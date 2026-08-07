#!/usr/bin/env python3
"""
generate_hex_bg.py
==================
Generates a 1920×1080 transparent PNG with an abstract, fading geometric
pattern of interconnected hexagons.  Visual weight is concentrated in the
bottom-right corner using a power-curve alpha falloff.

Colors: dark slate → electric blue with a soft glow halo effect.
Output: hex_bg.png (RGBA)
"""

import math
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.collections import LineCollection

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080
HEX_RADIUS = 40  # circumradius in px
COL_SPACING = HEX_RADIUS * math.sqrt(3)  # ≈ 69.28 px
ROW_SPACING = HEX_RADIUS * 1.5  # 60 px

# Anchor point for visual weight (bottom-right region).
# matplotlib y-axis: 0 = bottom, HEIGHT = top, so "bottom-right" is low y.
ANCHOR = (0.88 * WIDTH, 0.15 * HEIGHT)

# Power exponent for alpha falloff — higher = faster fade
ALPHA_EXPONENT = 2.5
ALPHA_CUTOFF = 0.03  # skip hexagons below this alpha

# Color endpoints (RGB, 0-1 range)
COLOR_FAR = np.array([0.184, 0.310, 0.435])   # #2F4F6F  dark slate
COLOR_NEAR = np.array([0.490, 0.976, 1.000])  # #7DF9FF  electric blue
STROKE_COLOR_BASE = np.array([0.627, 0.824, 0.859])  # #A0D2DB pale cyan

# ──────────────────────────────────────────────
# Grid construction
# ──────────────────────────────────────────────

def build_grid():
    """Return list of (col, row, cx, cy) for every hex cell covering the canvas."""
    n_cols = int(WIDTH / COL_SPACING) + 3  # +margin
    n_rows = int(HEIGHT / ROW_SPACING) + 3
    cells = []
    for col in range(-1, n_cols):
        for row in range(-1, n_rows):
            cx = col * COL_SPACING
            cy = row * ROW_SPACING + (col % 2) * (ROW_SPACING / 2)
            cells.append((col, row, cx, cy))
    return cells


def compute_alpha(cx, cy):
    """Compute opacity for a hex centered at (cx, cy)."""
    dx = cx - ANCHOR[0]
    dy = cy - ANCHOR[1]
    dist = math.hypot(dx, dy)
    # Maximum possible distance (top-left corner to anchor)
    max_dist = math.hypot(ANCHOR[0], HEIGHT - ANCHOR[1])
    normalized = min(dist / max_dist, 1.0)  # clamp to avoid negative base
    alpha = (1.0 - normalized) ** ALPHA_EXPONENT
    return alpha


def color_for_alpha(alpha):
    """Interpolate fill color from dark slate (low α) to electric blue (high α)."""
    t = alpha  # use alpha directly as interpolation parameter
    rgb = (1 - t) * COLOR_FAR + t * COLOR_NEAR
    return (*rgb, alpha)


# ──────────────────────────────────────────────
# Neighbor lookup (offset hex grid, flat-top)
# ──────────────────────────────────────────────

EVEN_COL_OFFSETS = [(+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (0, +1)]
ODD_COL_OFFSETS = [(+1, +1), (+1, 0), (0, -1), (-1, 0), (-1, +1), (0, +1)]


def get_neighbors(col, row):
    """Yield (ncol, nrow) for the 6 hex neighbors."""
    offsets = EVEN_COL_OFFSETS if col % 2 == 0 else ODD_COL_OFFSETS
    for dc, dr in offsets:
        yield col + dc, row + dr


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────

def main():
    # Build grid & compute alphas
    cells = build_grid()
    cell_map = {}  # (col, row) → (cx, cy, alpha)
    for col, row, cx, cy in cells:
        a = compute_alpha(cx, cy)
        if a >= ALPHA_CUTOFF:
            cell_map[(col, row)] = (cx, cy, a)

    # Create figure — exact pixel dimensions via dpi=1
    fig, ax = plt.subplots(
        figsize=(WIDTH, HEIGHT),
        dpi=1,
        facecolor="none",
        edgecolor="none",
    )
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ── 1. Glow halos (drawn first, behind everything) ──
    for (col, row), (cx, cy, a) in cell_map.items():
        if a < 0.08:
            continue
        # 3 concentric circles with rapidly decaying alpha
        for i, (scale, afrac) in enumerate([(2.2, 0.04), (1.7, 0.06), (1.3, 0.08)]):
            halo = Circle(
                (cx, cy),
                radius=HEX_RADIUS * scale,
                facecolor=(*COLOR_NEAR, a * afrac),
                edgecolor="none",
                linewidth=0,
            )
            ax.add_patch(halo)

    # ── 2. Connection lines ──
    segments = []
    seg_colors = []
    visited = set()
    for (col, row), (cx, cy, a) in cell_map.items():
        for ncol, nrow in get_neighbors(col, row):
            edge_key = tuple(sorted(((col, row), (ncol, nrow))))
            if edge_key in visited:
                continue
            visited.add(edge_key)
            if (ncol, nrow) in cell_map:
                nx, ny, na = cell_map[(ncol, nrow)]
                line_a = 0.4 * min(a, na)
                if line_a >= 0.01:
                    segments.append([(cx, cy), (nx, ny)])
                    seg_colors.append((*STROKE_COLOR_BASE, line_a))

    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=0.8)
        ax.add_collection(lc)

    # ── 3. Hexagon patches ──
    for (col, row), (cx, cy, a) in cell_map.items():
        fill_rgba = color_for_alpha(a)
        stroke_rgba = (*STROKE_COLOR_BASE, a * 0.6)
        hex_patch = RegularPolygon(
            (cx, cy),
            numVertices=6,
            radius=HEX_RADIUS,
            orientation=0,  # flat-top
            facecolor=fill_rgba,
            edgecolor=stroke_rgba,
            linewidth=1.0,
        )
        ax.add_patch(hex_patch)

    # ── Save ──
    output_path = "hex_bg.png"
    fig.savefig(
        output_path,
        dpi=1,
        transparent=True,
        pad_inches=0,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"✓ Saved {output_path}")


if __name__ == "__main__":
    main()
