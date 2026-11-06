"""Figures for inspecting waveforms. None of them labels or reports a heart rate."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from contactless_vitals.pipeline import COMBINED, RppgResult  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"
METHOD_COLOURS = {"green": "#2a78d6", "chrom": "#eb6834", "pos": "#1baf7a"}
METHOD_LABELS = {"green": "GREEN", "chrom": "CHROM", "pos": "POS"}
# Colour channels are drawn in their own colours, since that is what they are.
CHANNEL_COLOURS = ("#e34948", "#008300", "#2a78d6")
# Red and green are close for some colour-blind readers, so line style also differs.
CHANNEL_STYLES = ("-", (0, (5, 2)), (0, (1, 1.5)))
REGION_COLOURS = {"forehead": "#2a78d6", "left_cheek": "#eb6834", "right_cheek": "#1baf7a"}


def _style(ax) -> None:
    """Quiet axes, so the data carries the figure."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rgb_traces(result: RppgResult, path: Path) -> Path:
    """Colour change over time, one panel per source, on a shared scale."""
    sources = list(result.rgb)
    fig, axes = plt.subplots(
        len(sources), 1, figsize=(9, 1.7 * len(sources)), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, source in zip(np.atleast_1d(axes), sources):
        _style(ax)
        rgb = result.rgb[source]
        change = 100.0 * (rgb / np.nanmean(rgb, axis=0) - 1.0)
        for c, name in enumerate("RGB"):
            ax.plot(result.t, change[:, c], color=CHANNEL_COLOURS[c], ls=CHANNEL_STYLES[c],
                    lw=1.1, label=name)
        ax.set_title(source.replace("_", " "), loc="left", fontsize=10, color=TEXT)
        ax.set_ylabel("Change (%)")
    axes_flat = np.atleast_1d(axes)
    axes_flat[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3, frameon=False,
                        fontsize=9)
    axes_flat[-1].set_xlabel("Time (s)")
    fig.suptitle("Mean skin colour per region", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_waveforms(
    result: RppgResult,
    path: Path,
    source: str = COMBINED,
    reference: np.ndarray | None = None,
    reference_label: str = "Reference",
    span_s: float = 15.0,
) -> Path:
    """rPPG waveform per method. A reference, if given, is drawn behind each one."""
    methods = list(result.waveforms[source])
    fig, axes = plt.subplots(
        len(methods), 1, figsize=(9, 1.8 * len(methods)), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    shown = result.t <= result.t[0] + span_s
    for ax, method in zip(axes, methods):
        _style(ax)
        if reference is not None:
            ax.plot(result.t[shown], reference[shown], color=REFERENCE, lw=1.1,
                    label=reference_label)
        ax.plot(result.t[shown], result.waveforms[source][method][shown],
                color=METHOD_COLOURS[method], lw=1.5, label="rPPG")
        ax.set_title(METHOD_LABELS[method], loc="left", fontsize=10, color=TEXT)
        ax.set_ylabel("Amplitude (SD)")
    if reference is not None:
        axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, frameon=False,
                       fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"rPPG waveforms, {source} signal", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_spectra(result: RppgResult, path: Path, source: str = COMBINED) -> Path:
    """Power spectrum per method. Peaks are deliberately left unlabelled."""
    methods = list(result.waveforms[source])
    fig, axes = plt.subplots(
        len(methods), 1, figsize=(9, 1.6 * len(methods)), sharex=True
    )
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        _style(ax)
        x = result.waveforms[source][method]
        x = x[np.isfinite(x)]
        if x.size >= 8:
            freqs, power = signal.welch(
                x, result.fs, nperseg=min(x.size, int(10 * result.fs)), nfft=round(60 * result.fs)
            )
            keep = freqs <= 5.0
            ax.plot(freqs[keep], power[keep], color=METHOD_COLOURS[method], lw=1.5)
        ax.set_title(METHOD_LABELS[method], loc="left", fontsize=10, color=TEXT)
        ax.set_ylabel("Power")
        ax.set_yticks([])
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"Power spectra, {source} signal", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_roi_overlay(frame_rgb: np.ndarray, masks: dict[str, np.ndarray], path: Path) -> Path:
    """Skin masks drawn over a frame.

    Only call this with synthetic or public-domain images. Never use it on a
    dataset participant's face.
    """
    overlay = frame_rgb.astype(float)
    for name, mask in masks.items():
        tint = np.array(matplotlib.colors.to_rgb(REGION_COLOURS[name])) * 255.0
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * tint
    rows, cols = np.nonzero(np.any(np.stack(list(masks.values())), axis=0))
    pad = int(0.6 * (rows.max() - rows.min()))
    top, bottom = max(rows.min() - pad, 0), rows.max() + pad
    left, right = max(cols.min() - pad, 0), cols.max() + pad

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(overlay[top:bottom, left:right].astype(np.uint8))
    ax.set_axis_off()
    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=9, color=REGION_COLOURS[name],
                   label=name.replace("_", " "))
        for name in masks
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=3,
              frameon=False, fontsize=9)
    return _save(fig, path)


def plot_snr_by_scenario(
    scores: dict[str, dict[str, float]], path: Path, title: str
) -> Path:
    """Dot plot of SNR: one row per scenario, one dot per method."""
    scenarios = list(scores)
    methods = list(next(iter(scores.values())))
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(scenarios) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    rows = np.arange(len(scenarios))[::-1]
    markers = {"green": "o", "chrom": "s", "pos": "D"}
    offsets = {"green": 0.16, "chrom": 0.0, "pos": -0.16}
    for method in methods:
        ax.scatter(
            [scores[s][method] for s in scenarios],
            rows + offsets[method],
            s=52,
            marker=markers[method],
            color=METHOD_COLOURS[method],
            edgecolor=SURFACE,
            linewidth=1.2,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    ax.axvline(0, color=TEXT_MUTED, lw=1)
    ax.set_yticks(rows, [s.replace("_", " ") for s in scenarios], color=TEXT, fontsize=9)
    ax.set_xlabel("Pulse SNR (dB). Above 0, the pulse is stronger than everything else in the band")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=len(methods), frameon=False,
              fontsize=9)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)
