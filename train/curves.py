"""Training curves: loss, validation gap and learning rate per epoch, for one run or several overlaid."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.figure_style import GRID, INK_MUTED, INK_SECONDARY, SERIES, SURFACE, pyplot, title
from train.config import TrainConfig

HISTORY = "history.json"
FIGURE = "loss.png"
#: Panels are scaled to the epochs from this one on; the steep first epochs would flatten the rest.
ZOOM_FROM_EPOCH = 3
LABEL_SPACING_PX = 12


def _column(history: list[dict], key: str) -> list[float]:
    return [record[key] for record in history]


def _best(history: list[dict]) -> dict:
    return min(history, key=lambda record: record["val_loss"])


def _axes(plt, rows: int, heights: list[float]):
    fig, axes = plt.subplots(rows, 1, figsize=(7.5, 2.1 * sum(heights) + 0.9), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Epoch")
    return fig, axes


def _zoom(ax, curves: list[tuple[list[int], list[float]]]) -> None:
    tail = [v for epochs, values in curves for e, v in zip(epochs, values) if e >= ZOOM_FROM_EPOCH]
    if not tail:
        return
    low, high = min(tail), max(tail)
    pad = (high - low) * 0.08 or abs(high) * 0.05 or 0.01
    ax.set_ylim(low - pad, high + pad)


def _end_labels(ax, ends: list[tuple[float, float, str, str]]) -> None:
    """Label each line at its last point, nudged apart vertically so labels never overlap."""
    if not ends:
        return
    ax.figure.canvas.draw()
    to_px, to_data = ax.transData, ax.transData.inverted()
    placed = sorted((to_px.transform((x, y))[1], x, y, text, colour) for x, y, text, colour in ends)
    heights = [placed[0][0]]
    for py, *_ in placed[1:]:
        heights.append(max(py, heights[-1] + LABEL_SPACING_PX))
    for height, (_, x, y, text, colour) in zip(heights, placed):
        x_px = to_px.transform((x, y))[0]
        label_y = to_data.transform((x_px, height))[1]
        ax.annotate(text, (x, label_y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_SECONDARY, annotation_clip=False)
        ax.plot(x, y, "o", markersize=4, color=colour, markeredgecolor=SURFACE, markeredgewidth=1, clip_on=False)


def plot_history(history: list[dict], path: Path, name: str, budget: int | None = None) -> Path:
    """One run: training and validation loss, validation gap, learning rate; the kept epoch marked."""
    plt = pyplot()
    epochs = _column(history, "epoch")
    best = _best(history)
    fig, (loss_ax, gap_ax, rate_ax) = _axes(plt, 3, [1.6, 1.0, 0.7])

    losses = {"training": (SERIES[1], _column(history, "train_loss")), "validation": (SERIES[0], _column(history, "val_loss"))}
    for label, (colour, values) in losses.items():
        loss_ax.plot(epochs, values, color=colour, linewidth=2, label=label)
    loss_ax.set_title("Loss: training and validation", loc="left", fontsize=9, color=INK_SECONDARY)
    loss_ax.legend(loc="upper center", frameon=False, fontsize=8, ncols=2)
    _zoom(loss_ax, [(epochs, values) for _, values in losses.values()])

    gaps = _column(history, "val_gap_fidelity")
    gap_ax.plot(epochs, gaps, color=SERIES[0], linewidth=2)
    gap_ax.set_title("Validation gap fidelity (lower is better)", loc="left", fontsize=9, color=INK_SECONDARY)
    _zoom(gap_ax, [(epochs, gaps)])

    rate_ax.plot(epochs, _column(history, "learning_rate"), color=INK_MUTED, linewidth=2)
    rate_ax.set_title("Learning rate", loc="left", fontsize=9, color=INK_SECONDARY)
    rate_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for ax in (loss_ax, gap_ax, rate_ax):
        ax.axvline(best["epoch"], color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    rate_ax.annotate(f"kept: epoch {best['epoch']}", (best["epoch"], 1), xycoords=("data", "axes fraction"),
                     xytext=(-4, 0), textcoords="offset points", ha="right", va="top", fontsize=8, color=INK_MUTED)

    ran = f"{len(history)} of {budget} epochs" if budget else f"{len(history)} epochs"
    if budget and len(history) < budget:
        ran += ", stopped early"
    title(fig, name, f"{ran}; kept epoch {best['epoch']}: validation loss {best['val_loss']:.4f}, "
                     f"gap {best['val_gap_fidelity']:.4f}. Loss and gap axes are scaled from epoch {ZOOM_FROM_EPOCH}.")
    fig.subplots_adjust(top=0.86, right=0.86, hspace=0.38)
    _end_labels(loss_ax, [(epochs[-1], values[-1], label, colour) for label, (colour, values) in losses.items()])
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_runs(histories: dict[str, list[dict]], path: Path) -> Path:
    """Several runs overlaid: training (dashed) and validation (solid) loss, and validation gap; dots mark each kept epoch."""
    if len(histories) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs fit one figure; split them")
    plt = pyplot()
    fig, (loss_ax, gap_ax) = _axes(plt, 2, [1.4, 1.0])
    loss_curves, gap_curves, loss_ends, gap_ends = [], [], [], []
    for (label, history), colour in zip(histories.items(), SERIES):
        epochs = _column(history, "epoch")
        best = _best(history)
        train, val, gap = (_column(history, key) for key in ("train_loss", "val_loss", "val_gap_fidelity"))
        loss_ax.plot(epochs, val, color=colour, linewidth=2, label=label)
        loss_ax.plot(epochs, train, color=colour, linewidth=1.5, linestyle=(0, (4, 2)))
        gap_ax.plot(epochs, gap, color=colour, linewidth=2)
        for ax, key in ((loss_ax, "val_loss"), (gap_ax, "val_gap_fidelity")):
            ax.plot(best["epoch"], best[key], "o", markersize=6, color=colour, markeredgecolor=SURFACE, markeredgewidth=2)
        loss_curves += [(epochs, val), (epochs, train)]
        gap_curves.append((epochs, gap))
        loss_ends.append((epochs[-1], val[-1], label, colour))
        gap_ends.append((epochs[-1], gap[-1], label, colour))

    loss_ax.set_title("Loss: validation solid, training dashed", loc="left", fontsize=9, color=INK_SECONDARY)
    gap_ax.set_title("Validation gap fidelity (lower is better)", loc="left", fontsize=9, color=INK_SECONDARY)
    loss_ax.legend(loc="upper right", frameon=False, fontsize=8)
    _zoom(loss_ax, loss_curves)
    _zoom(gap_ax, gap_curves)
    title(fig, "Training curves", f"Dots mark the epoch each run kept (lowest validation loss). Axes are scaled from epoch {ZOOM_FROM_EPOCH}.")
    fig.subplots_adjust(top=0.84, right=0.74, hspace=0.38)
    if len(histories) <= 4:
        _end_labels(loss_ax, loss_ends)
        _end_labels(gap_ax, gap_ends)
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def read_history(run_dir: Path) -> list[dict]:
    return json.loads((Path(run_dir) / HISTORY).read_text(encoding="utf-8"))


def read_budget(run_dir: Path) -> int | None:
    config = Path(run_dir) / "config.yaml"
    return TrainConfig.load(config).optim.epochs if config.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories holding history.json")
    parser.add_argument("--out", type=Path, default=None, help="figure path; one run defaults to its loss.png")
    args = parser.parse_args()

    if len(args.runs) == 1:
        run = args.runs[0]
        out = plot_history(read_history(run), args.out or run / FIGURE, run.name, read_budget(run))
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = plot_runs({run.name: read_history(run) for run in args.runs}, args.out or Path("runs") / f"curves-{stamp}.png")
    print(out)


if __name__ == "__main__":
    main()
