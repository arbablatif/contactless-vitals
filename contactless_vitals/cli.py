"""Command line: `python -m contactless_vitals run --mock`, `run --subject`, `check-data`.

Outputs waveforms and plots. It does not output a heart rate.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from contactless_vitals import plots
from contactless_vitals.config import SignalConfig
from contactless_vitals.data import synthetic, ubfc
from contactless_vitals.extract import cached_rgb_trace, extract_rgb_trace
from contactless_vitals.pipeline import COMBINED, RppgResult, run_rppg

OUT_DIR = Path("out")


def save_outputs(
    result: RppgResult,
    out_dir: Path,
    cfg: SignalConfig,
    reference: np.ndarray | None = None,
    reference_label: str = "Reference",
) -> list[Path]:
    """Write waveforms (CSV and NPZ), sliding windows and plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "waveforms.csv"
    columns = {"t_s": result.t}
    for source, methods in result.waveforms.items():
        for method, x in methods.items():
            columns[f"{source}_{method}"] = x
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*(np.round(c, 6) for c in columns.values())))

    npz_path = out_dir / "waveforms.npz"
    arrays = {"t": result.t, "fs": np.array(result.fs)}
    for method in result.waveforms[COMBINED]:
        arrays[f"{COMBINED}_{method}"] = result.waveforms[COMBINED][method]
        starts, windows = result.windows(COMBINED, method, cfg)
        arrays[f"{COMBINED}_{method}_window_starts_s"] = starts
        arrays[f"{COMBINED}_{method}_windows"] = windows
    np.savez_compressed(npz_path, **arrays)

    figures = out_dir / "figures"
    return [
        csv_path,
        npz_path,
        plots.plot_rgb_traces(result, figures / "rgb_traces.png"),
        plots.plot_waveforms(
            result, figures / "waveforms.png", reference=reference, reference_label=reference_label
        ),
        plots.plot_spectra(result, figures / "spectra.png"),
    ]


def run_mock(args, cfg: SignalConfig) -> list[Path]:
    """End to end on a synthetic video: no dataset, no model, no network."""
    spec = synthetic.with_changes(synthetic.SCENARIOS[args.scenario], pulse_hz=args.pulse_hz)
    trace = extract_rgb_trace(synthetic.synthetic_frames(spec), synthetic.ScriptedLandmarks(spec))
    result = run_rppg(trace, cfg)
    truth = synthetic.pulse_waveform(result.t, spec.pulse_hz)
    truth = (truth - truth.mean()) / truth.std()
    return save_outputs(
        result, OUT_DIR / f"mock_{args.scenario}", cfg, reference=truth,
        reference_label="True pulse",
    )


def run_subject(args, cfg: SignalConfig) -> list[Path]:
    """Run on one UBFC-rPPG subject. Traces are cached under out/traces."""
    from contactless_vitals.face.landmarks import MediaPipeLandmarks

    root = ubfc.resolve_root(args.root)
    if root is None:
        sys.exit(ubfc.access_instructions())
    matches = [s for s in ubfc.discover_subjects(root) if s.name == args.subject]
    if not matches:
        sys.exit(f"{args.subject} not found under {root}.\n\n{ubfc.access_instructions()}")
    subject = matches[0]

    def compute():
        with MediaPipeLandmarks() as provider:
            return extract_rgb_trace(ubfc.subject_frames(subject), provider)

    trace = cached_rgb_trace(subject.video_path, OUT_DIR / "traces", compute)
    result = run_rppg(trace, cfg)
    reference = reference_on_grid(result, ubfc.read_ground_truth(subject.ground_truth_path), cfg)
    return save_outputs(
        result, OUT_DIR / subject.name, cfg, reference=reference, reference_label="Contact PPG"
    )


def reference_on_grid(
    result: RppgResult, truth: ubfc.GroundTruth, cfg: SignalConfig
) -> np.ndarray:
    """Contact PPG resampled to the rPPG grid, filtered the same way, unit variance."""
    from contactless_vitals.signal.preprocess import bandpass, first_of_each_time

    unique = first_of_each_time(truth.t)
    x = np.interp(result.t, truth.t[unique], truth.ppg[unique])
    x = bandpass(x - x.mean(), result.fs, cfg.band_hz, cfg.filter_order)
    return (x - x.mean()) / x.std()


def check_data(args) -> int:
    """Report on the downloaded dataset, or explain how to get it."""
    root = ubfc.resolve_root(args.root)
    if root is None:
        print(ubfc.access_instructions())
        return 1
    lines, ok = ubfc.check_dataset(root)
    print("\n".join(lines))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="contactless_vitals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="extract rPPG waveforms and plots")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--mock", action="store_true", help="use a synthetic video")
    source.add_argument("--subject", help="UBFC-rPPG subject folder, such as subject1")
    run.add_argument("--root", help=f"dataset folder (default: ${ubfc.ROOT_ENV_VAR})")
    run.add_argument("--scenario", default="clean", choices=sorted(synthetic.SCENARIOS),
                     help="synthetic condition for --mock")
    run.add_argument("--pulse-hz", type=float, default=1.2,
                     help="known pulse frequency for --mock")

    check = commands.add_parser("check-data", help="validate a downloaded dataset")
    check.add_argument("--root", help=f"dataset folder (default: ${ubfc.ROOT_ENV_VAR})")

    args = parser.parse_args(argv)
    if args.command == "check-data":
        return check_data(args)

    cfg = SignalConfig()
    written = run_mock(args, cfg) if args.mock else run_subject(args, cfg)
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0
