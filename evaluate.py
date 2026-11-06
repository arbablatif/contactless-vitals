"""Measure the pipeline and write the results shown in the README.

    python evaluate.py                                  synthetic only
    python evaluate.py --ubfc subject1 subject3 subject4    also real subjects

Every score compares a waveform with an answer already known: the frequency used
to make a synthetic video, or the contact sensor on a real recording. Nothing here
estimates heart rate. No API calls are made.

Synthetic results go to results/. UBFC results hold only per-subject scores, and
figures showing a participant's face are never produced.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from contactless_vitals import plots
from contactless_vitals.cli import OUT_DIR, reference_on_grid
from contactless_vitals.config import SignalConfig, SkinConfig
from contactless_vitals.data import synthetic as syn
from contactless_vitals.data import ubfc
from contactless_vitals.extract import cached_rgb_trace, extract_rgb_trace
from contactless_vitals.pipeline import COMBINED, RppgResult, run_rppg
from contactless_vitals.quality import (
    correlation_with_lag,
    shifted_correlations,
    snr_at_known_frequency,
    windowed_snr,
)

METHODS = ("green", "chrom", "pos")
SEEDS = (0, 1, 2, 3, 4)
RESULTS = Path("results")
FIGURES = RESULTS / "figures"


def score(result: RppgResult, spec: syn.SyntheticSpec) -> dict[str, dict[str, float]]:
    """SNR and correlation with the true pulse, for each method."""
    truth = syn.pulse_waveform(result.t, spec.pulse_hz)
    scores = {}
    for method in METHODS:
        x = result.waveforms[COMBINED][method]
        scores[method] = {
            "snr_db": snr_at_known_frequency(x, result.fs, spec.pulse_hz),
            "r": correlation_with_lag(x, truth, result.fs, max_lag_s=0.1)[0],
        }
    return scores


def score_over_seeds(spec: syn.SyntheticSpec) -> dict[str, dict[str, float]]:
    """Average over several noise seeds, keeping the worst case too."""
    runs = []
    for seed in SEEDS:
        seeded = syn.with_changes(spec, seed=seed)
        trace = extract_rgb_trace(syn.synthetic_frames(seeded), syn.ScriptedLandmarks(seeded))
        runs.append(score(run_rppg(trace), seeded))
    return {
        m: {
            "snr_db": float(np.mean([r[m]["snr_db"] for r in runs])),
            "snr_db_worst": float(np.min([r[m]["snr_db"] for r in runs])),
            "r": float(np.mean([r[m]["r"] for r in runs])),
        }
        for m in METHODS
    }


def run_portrait() -> dict:
    """The known pulse through real MediaPipe tracking, on a public-domain portrait."""
    from skimage import data

    from contactless_vitals.face.landmarks import MediaPipeLandmarks
    from contactless_vitals.face.skin import region_masks

    image = data.astronaut()
    spec = syn.SyntheticSpec(pulse_hz=1.3, duration_s=20.0, sway_px=3.0, sway_hz=0.4)
    with MediaPipeLandmarks() as provider:
        trace = extract_rgb_trace(syn.animate_portrait(image, spec), provider)
    with MediaPipeLandmarks() as provider:
        masks = region_masks(image, provider.detect(image, 0.0), SkinConfig())
    plots.plot_roi_overlay(image, masks, FIGURES / "roi_overlay.png")
    return score(run_rppg(trace), spec)


def run_synthetic() -> dict:
    results = {
        "scenarios": {n: score_over_seeds(s) for n, s in syn.SCENARIOS.items()},
        "pulse_rates": {
            f"{hz} Hz": score_over_seeds(syn.SyntheticSpec(pulse_hz=hz))
            for hz in (0.8, 1.2, 2.0, 3.0, 3.5)
        },
        "skin_tones": {
            **{
                f"{tone}, full pulse": score_over_seeds(syn.SyntheticSpec(skin_rgb=rgb))
                for tone, rgb in syn.SKIN_TONES.items()
            },
            "dark, half pulse": score_over_seeds(
                syn.SyntheticSpec(skin_rgb=syn.SKIN_TONES["dark"], pulse_amplitude=0.004)
            ),
        },
        "negative_control": {
            "no pulse": score_over_seeds(syn.SyntheticSpec(pulse_amplitude=0.0))
        },
        "real_tracking": {"portrait through MediaPipe": run_portrait()},
    }
    (RESULTS / "synthetic_results.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "synthetic_results.md").write_text(to_markdown(results))

    plots.plot_snr_by_scenario(
        {n: {m: v[m]["snr_db"] for m in METHODS} for n, v in results["scenarios"].items()},
        FIGURES / "snr_by_scenario.png",
        "Pulse SNR by condition, synthetic videos",
    )
    spec = syn.SCENARIOS["in_band_brightness"]
    result = run_rppg(extract_rgb_trace(syn.synthetic_frames(spec), syn.ScriptedLandmarks(spec)))
    truth = syn.pulse_waveform(result.t, spec.pulse_hz)
    plots.plot_waveforms(
        result,
        FIGURES / "waveforms_in_band_brightness.png",
        reference=(truth - truth.mean()) / truth.std(),
        reference_label="True pulse",
        span_s=8.0,
    )
    return results


def to_markdown(results: dict) -> str:
    """Results as tables: mean SNR in dB, with the worst seed in brackets."""
    titles = {
        "scenarios": "Conditions (pulse at 1.2 Hz)",
        "pulse_rates": "Pulse rates",
        "skin_tones": "Skin tones",
        "negative_control": "Negative control (should be below 0 dB)",
        "real_tracking": "Real MediaPipe tracking (single run)",
    }
    lines = [f"Mean SNR in dB over {len(SEEDS)} noise seeds, worst seed in brackets.", ""]
    for section, rows in results.items():
        lines += [f"### {titles[section]}", "", "| Case | GREEN | CHROM | POS |", "|---|---|---|---|"]
        for name, row in rows.items():
            cells = []
            for method in METHODS:
                cell = f"{row[method]['snr_db']:.1f}"
                if "snr_db_worst" in row[method]:
                    cell += f" ({row[method]['snr_db_worst']:.1f})"
                cells.append(cell)
            lines.append(f"| {name.replace('_', ' ')} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def run_ubfc(names: list[str], root: Path | None) -> None:
    """Score real subjects against the contact sensor. Results are per-subject scores."""
    from contactless_vitals.face.landmarks import MediaPipeLandmarks

    root = ubfc.resolve_root(root)
    if root is None:
        raise SystemExit(ubfc.access_instructions())
    subjects = {s.name: s for s in ubfc.discover_subjects(root)}
    cfg = SignalConfig()
    lines = [
        "Median per-window SNR in dB, and correlation with the contact PPG. Chance is",
        "the median and highest correlation with the same sensor signal shifted out of step.",
        "",
        "| Subject | Method | SNR (dB) | Correlation | Chance (median / max) | Lag (s) |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        subject = subjects[name]

        def compute():
            with MediaPipeLandmarks() as provider:
                return extract_rgb_trace(ubfc.subject_frames(subject), provider)

        trace = cached_rgb_trace(subject.video_path, OUT_DIR / "traces", compute)
        result = run_rppg(trace, cfg)
        truth = ubfc.read_ground_truth(subject.ground_truth_path)
        reference = reference_on_grid(result, truth, cfg)
        for method in METHODS:
            x = result.waveforms[COMBINED][method]
            snr = windowed_snr(x, result.t, result.fs, truth.t, truth.hr_bpm / 60.0)
            r, lag = correlation_with_lag(x, reference, result.fs)
            chance = shifted_correlations(x, reference, result.fs)
            lines.append(
                f"| {name} | {method.upper()} | {np.median(snr):.1f} | {r:.2f} "
                f"| {np.median(chance):.2f} / {chance.max():.2f} | {lag:+.2f} |"
            )
    (RESULTS / "ubfc_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ubfc", nargs="*", metavar="SUBJECT", help="also score these subjects")
    parser.add_argument("--root", help="UBFC-rPPG folder (default: from .env)")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    print(to_markdown(run_synthetic()))
    if args.ubfc:
        run_ubfc(args.ubfc, args.root)


if __name__ == "__main__":
    main()
