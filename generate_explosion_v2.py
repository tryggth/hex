#!/usr/bin/env python3
"""
generate_explosion_v2.py
========================
V2: Fixes line contrast (light slate edges), canvas padding (top 1/3 clear),
and preserves MCTS cyan glow path.

Generates a 3840×2160 transparent PNG comparing two decision trees side-by-side:
  - Left:  Minimax — massive, perfectly symmetrical, dense, light slate lines
  - Right: MCTS   — sparse, asymmetric, one deep path glowing cyan (#00FFFF)

Output: combinatorial_explosion_v2.png (RGBA)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import networkx as nx
import numpy as np

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
WIDTH, HEIGHT = 3840, 2160

# Layout regions (x ranges)
LEFT_X_MIN, LEFT_X_MAX = 100, 1800
RIGHT_X_MIN, RIGHT_X_MAX = 2040, 3740
Y_TOP = 1350       # root y — lower ⅔, top third clear for slide text
Y_BOTTOM = 100     # deepest leaf y

# Colors — brightened for dark-background readability
GRAY_NODE = "#90A4AE"       # Blue-gray (visible on dark slides)
GRAY_LEAF = "#B0BEC5"       # Lighter blue-gray for leaves
GRAY_EDGE = "#CBD5E1"       # Light slate silver
CYAN = "#00FFFF"
WHITE = "#FFFFFF"
DIVIDER_COLOR = "#556677"
TITLE_COLOR = "#E0E0E0"

# ──────────────────────────────────────────────
# 1. Build Minimax Tree (perfect ternary, depth 6)
# ──────────────────────────────────────────────

def build_minimax_tree(branching=3, depth=6):
    """Build a perfect k-ary tree and return (graph, root, leaf_set)."""
    G = nx.DiGraph()
    node_id = [0]  # mutable counter

    def _add(parent, d):
        if d > depth:
            return
        for _ in range(branching):
            child = node_id[0]
            node_id[0] += 1
            G.add_edge(parent, child)
            G.nodes[child]["depth"] = d
            _add(child, d + 1)

    root = "minimax_root"
    G.add_node(root, depth=0)
    node_id[0] = 1
    _add(root, 1)
    leaves = {n for n in G.nodes if G.out_degree(n) == 0}
    return G, root, leaves


# ──────────────────────────────────────────────
# 2. Build MCTS Tree (hand-crafted asymmetric)
# ──────────────────────────────────────────────

def build_mcts_tree():
    """
    Build a sparse, asymmetric MCTS-style tree.
    Returns (graph, root, primary_path_nodes, primary_path_edges, labels).
    """
    G = nx.DiGraph()

    # Node naming: descriptive for clarity
    edges = [
        # Branch A — lightly explored
        ("R", "A1"),
        ("A1", "A1a"),
        ("A1", "A1b"),
        ("A1a", "A1a1"),

        # Branch B — THE PRIMARY DEEP PATH  ★
        ("R", "B1"),
        ("B1", "B1a"),       # side dead-end
        ("B1", "B2"),        # ★ continue
        ("B2", "B2a"),       # side dead-end
        ("B2a", "B2a1"),
        ("B2", "B3"),        # ★ continue
        ("B3", "B3a"),       # small side branch
        ("B3a", "B3a1"),
        ("B3", "B4"),        # ★ continue
        ("B4", "B4a"),       # side dead-end
        ("B4", "B5"),        # ★ continue
        ("B5", "B5a"),       # side
        ("B5", "B6"),        # ★ continue
        ("B6", "B7"),        # ★ terminal — reward

        # Branch C — lightly explored
        ("R", "C1"),
        ("C1", "C1a"),

        # Branch D — barely touched
        ("R", "D1"),
    ]
    G.add_edges_from(edges)

    # Primary path
    primary_path_nodes = {"R", "B1", "B2", "B3", "B4", "B5", "B6", "B7"}
    primary_path_edges = {
        ("R", "B1"), ("B1", "B2"), ("B2", "B3"), ("B3", "B4"),
        ("B4", "B5"), ("B5", "B6"), ("B6", "B7"),
    }

    # Labels for path nodes
    labels = {
        "R":  "s₀",
        "B1": "Select",
        "B2": "Expand",
        "B4": "Simulate",
        "B7": "Reward ✓",
    }

    return G, "R", primary_path_nodes, primary_path_edges, labels


# ──────────────────────────────────────────────
# 3. Recursive Layout
# ──────────────────────────────────────────────

def recursive_layout(G, root, x_min, x_max, y_top, y_bottom):
    """
    Top-down recursive band-division layout.
    Returns dict {node: (x, y)}.
    """
    depths = nx.single_source_shortest_path_length(G, root)
    max_depth = max(depths.values()) if depths else 1
    y_step = (y_top - y_bottom) / max(max_depth, 1)
    pos = {}

    def _lay(node, xlo, xhi, y):
        cx = (xlo + xhi) / 2
        pos[node] = (cx, y)
        children = list(G.successors(node))
        if not children:
            return
        band = (xhi - xlo) / len(children)
        for i, child in enumerate(children):
            _lay(child, xlo + i * band, xlo + (i + 1) * band, y - y_step)

    _lay(root, x_min, x_max, y_top)
    return pos


# ──────────────────────────────────────────────
# 4. Drawing Helpers
# ──────────────────────────────────────────────

def draw_minimax(ax, G, root, leaves, pos):
    """Draw the massive light-slate Minimax tree."""
    # Edges — high contrast light slate
    segments = []
    for u, v in G.edges():
        segments.append([pos[u], pos[v]])
    lc = LineCollection(segments, colors=GRAY_EDGE, linewidths=1.3, alpha=0.75)
    ax.add_collection(lc)

    # Nodes
    internal = [n for n in G.nodes if n not in leaves and n != root]
    leaf_list = list(leaves)

    # Internal nodes
    if internal:
        xs = [pos[n][0] for n in internal]
        ys = [pos[n][1] for n in internal]
        ax.scatter(xs, ys, s=6, c=GRAY_NODE, alpha=0.75, zorder=3, edgecolors="none")

    # Leaf nodes — slightly lighter
    if leaf_list:
        xs = [pos[n][0] for n in leaf_list]
        ys = [pos[n][1] for n in leaf_list]
        ax.scatter(xs, ys, s=4, c=GRAY_LEAF, alpha=0.6, zorder=3, edgecolors="none")

    # Root node — larger and brighter
    rx, ry = pos[root]
    ax.scatter([rx], [ry], s=40, c=GRAY_NODE, alpha=0.95, zorder=4, edgecolors="none")


def draw_mcts(ax, G, root, pos, primary_nodes, primary_edges, labels):
    """Draw the sparse MCTS tree with glowing primary path."""

    # ── Non-path edges — brightened for dark bg consistency ──
    non_path_segs = []
    for u, v in G.edges():
        if (u, v) not in primary_edges:
            non_path_segs.append([pos[u], pos[v]])
    if non_path_segs:
        lc = LineCollection(non_path_segs, colors="#A0AEC0", linewidths=1.0, alpha=0.5)
        ax.add_collection(lc)

    # ── Primary path edges — glow layers + core ──
    path_segs = []
    for u, v in primary_edges:
        path_segs.append([pos[u], pos[v]])

    if path_segs:
        # Outer glow (wide, faint)
        lc_glow3 = LineCollection(path_segs, colors=CYAN, linewidths=14, alpha=0.06)
        ax.add_collection(lc_glow3)
        lc_glow2 = LineCollection(path_segs, colors=CYAN, linewidths=8, alpha=0.12)
        ax.add_collection(lc_glow2)
        lc_glow1 = LineCollection(path_segs, colors=CYAN, linewidths=4.5, alpha=0.25)
        ax.add_collection(lc_glow1)
        # Core line
        lc_core = LineCollection(path_segs, colors=CYAN, linewidths=2.5, alpha=0.95)
        ax.add_collection(lc_core)

    # ── Non-path nodes ──
    non_path = [n for n in G.nodes if n not in primary_nodes]
    if non_path:
        xs = [pos[n][0] for n in non_path]
        ys = [pos[n][1] for n in non_path]
        ax.scatter(xs, ys, s=25, c="#A0AEC0", alpha=0.55, zorder=3, edgecolors="none")

    # ── Primary path nodes — glow halos + node circles ──
    for node in primary_nodes:
        nx_, ny_ = pos[node]
        # Glow halos — larger for stronger bloom
        for radius, a in [(70, 0.03), (50, 0.06), (35, 0.10), (22, 0.18)]:
            halo = Circle((nx_, ny_), radius=radius, facecolor=CYAN, alpha=a,
                          edgecolor="none", linewidth=0)
            ax.add_patch(halo)
        # Node dot
        ax.scatter([nx_], [ny_], s=120, c=CYAN, alpha=1.0, zorder=5,
                   edgecolors=WHITE, linewidths=1.5)

    # ── Text labels with dark shadow for readability on any background ──
    for node, label in labels.items():
        nx_, ny_ = pos[node]
        offset_x = 40
        offset_y = 15
        # Dark shadow (rendered first, slightly offset)
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 0)]:
            ax.text(
                nx_ + offset_x + dx, ny_ + offset_y + dy, label,
                fontsize=18, fontweight="bold", color="#1a1a2e",
                ha="left", va="center",
                fontfamily="sans-serif",
                zorder=6,
            )
        # Foreground label in cyan
        ax.text(
            nx_ + offset_x, ny_ + offset_y, label,
            fontsize=18, fontweight="bold", color=CYAN,
            ha="left", va="center",
            fontfamily="sans-serif",
            zorder=7,
        )


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

def main():
    # ── Build trees ──
    mm_G, mm_root, mm_leaves = build_minimax_tree(branching=3, depth=6)
    mcts_G, mcts_root, mcts_primary_nodes, mcts_primary_edges, mcts_labels = build_mcts_tree()

    # ── Layout ──
    mm_pos = recursive_layout(mm_G, mm_root, LEFT_X_MIN, LEFT_X_MAX, Y_TOP, Y_BOTTOM)
    mcts_pos = recursive_layout(mcts_G, mcts_root, RIGHT_X_MIN, RIGHT_X_MAX, Y_TOP, Y_BOTTOM)

    # ── Create figure ──
    DPI = 100
    fig_w = WIDTH / DPI   # 38.4 inches
    fig_h = HEIGHT / DPI  # 21.6 inches
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI, facecolor="none")
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ── Draw ──
    draw_minimax(ax, mm_G, mm_root, mm_leaves, mm_pos)
    draw_mcts(ax, mcts_G, mcts_root, mcts_pos,
              mcts_primary_nodes, mcts_primary_edges, mcts_labels)

    # ── Divider line ──
    ax.plot(
        [1920, 1920], [Y_BOTTOM - 20, Y_TOP + 60],
        color=DIVIDER_COLOR, linewidth=1.0, alpha=0.3, zorder=1,
    )

    # ── Panel titles ──
    title_y = Y_TOP + 100
    ax.text(
        (LEFT_X_MIN + LEFT_X_MAX) / 2, title_y,
        "Minimax: Exhaustive Search",
        fontsize=28, fontweight="bold", color=TITLE_COLOR,
        ha="center", va="center", fontfamily="sans-serif",
    )
    ax.text(
        (RIGHT_X_MIN + RIGHT_X_MAX) / 2, title_y,
        "MCTS: Selective Exploration",
        fontsize=28, fontweight="bold", color=CYAN,
        ha="center", va="center", fontfamily="sans-serif",
    )

    # ── Subtitle annotations ──
    subtitle_y = Y_TOP + 50
    ax.text(
        (LEFT_X_MIN + LEFT_X_MAX) / 2, subtitle_y,
        f"{mm_G.number_of_nodes():,} nodes  ·  branching factor 3  ·  depth 6",
        fontsize=16, color="#999999",
        ha="center", va="center", fontfamily="sans-serif",
    )
    ax.text(
        (RIGHT_X_MIN + RIGHT_X_MAX) / 2, subtitle_y,
        f"{mcts_G.number_of_nodes()} nodes  ·  focused on most promising path",
        fontsize=16, color="#88CCCC",
        ha="center", va="center", fontfamily="sans-serif",
    )

    # ── Save ──
    output = "combinatorial_explosion_v2.png"
    fig.savefig(output, dpi=DPI, transparent=True, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {output}  ({mm_G.number_of_nodes():,} Minimax nodes, "
          f"{mcts_G.number_of_nodes()} MCTS nodes)")


if __name__ == "__main__":
    main()
