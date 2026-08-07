#!/usr/bin/env python3
"""
generate_icon.py
================
Generates PWA icons (192x192 and 512x512) featuring a cyan hexagon
with a dark navy fill and an "H" letter in the center.
Output: icons/icon-192x192.png, icons/icon-512x512.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import os

# Colors
NAVY = "#0a0e1a"
CYAN = "#00FFFF"
CARD_BG = "#131a2e"

SIZES = [192, 512]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")


def generate_icon(size):
    """Generate a single icon at the given pixel size."""
    DPI = 100
    fig_inches = size / DPI

    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=DPI)
    fig.patch.set_facecolor(NAVY)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Outer glow hex
    glow_hex = RegularPolygon(
        (0, 0), numVertices=6, radius=1.35,
        orientation=0,  # flat-top
        facecolor=CYAN, alpha=0.08, edgecolor="none"
    )
    ax.add_patch(glow_hex)

    # Main hexagon
    hex_patch = RegularPolygon(
        (0, 0), numVertices=6, radius=1.15,
        orientation=0,
        facecolor=CARD_BG, edgecolor=CYAN, linewidth=3.5
    )
    ax.add_patch(hex_patch)

    # "H" letter in the center
    ax.text(
        0, -0.05, "H",
        fontsize=size * 0.28, fontweight="bold", color=CYAN,
        ha="center", va="center", fontfamily="sans-serif"
    )

    # Save
    output_path = os.path.join(OUTPUT_DIR, f"icon-{size}x{size}.png")
    fig.savefig(output_path, dpi=DPI, facecolor=NAVY, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output_path} ({size}x{size})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating PWA icons...")
    for size in SIZES:
        generate_icon(size)
    print("Done!")


if __name__ == "__main__":
    main()
