"""Colours and matplotlib styling shared by every figure the project draws."""

from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
NEUTRAL = "#f0efec"
#: Categorical hues in fixed order; validated for colour-vision deficiency on SURFACE.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def pyplot():
    """matplotlib's pyplot, off-screen, with the project's styling applied."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def title(fig, heading: str, subtitle: str) -> None:
    fig.text(0.01, 0.985, heading, fontsize=12, fontweight="semibold", va="top")
    fig.text(0.01, 0.95, subtitle, fontsize=9, color=INK_SECONDARY, va="top", wrap=True)
