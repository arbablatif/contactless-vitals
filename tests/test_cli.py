"""The command line, in the no-network --mock mode."""

import csv

import numpy as np
import pytest

from contactless_vitals import cli
from contactless_vitals.data import ubfc


def test_mock_run_writes_waveforms_and_plots(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "OUT_DIR", tmp_path)
    assert cli.main(["run", "--mock", "--scenario", "in_band_brightness"]) == 0

    out = tmp_path / "mock_in_band_brightness"
    for name in ("rgb_traces", "waveforms", "spectra"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 5000

    with open(out / "waveforms.csv") as f:
        header = next(csv.reader(f))
    assert header[0] == "t_s"
    assert {"combined_green", "combined_chrom", "combined_pos", "forehead_pos"} <= set(header)

    with np.load(out / "waveforms.npz") as saved:
        assert saved["combined_pos_windows"].shape[1] == 300


def test_outputs_contain_no_heart_rate(tmp_path, monkeypatch, capsys):
    """This stage stops at the waveform; heart rate belongs to a later one."""
    monkeypatch.setattr(cli, "OUT_DIR", tmp_path)
    cli.main(["run", "--mock"])
    printed = capsys.readouterr().out.lower().replace(str(tmp_path).lower(), "")
    out = tmp_path / "mock_clean"
    with open(out / "waveforms.csv") as f:
        header = f.readline().lower()
    with np.load(out / "waveforms.npz") as saved:
        keys = " ".join(saved.files).lower()
    for text in (printed, header, keys):
        assert "bpm" not in text and "heart" not in text and "_hr" not in text


def test_missing_dataset_explains_how_to_get_it(monkeypatch, capsys):
    monkeypatch.delenv(ubfc.ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(ubfc, "load_dotenv", lambda: None)
    assert cli.main(["check-data"]) == 1
    assert ubfc.DATASET_URL in capsys.readouterr().out


def test_mock_and_subject_cannot_be_combined():
    with pytest.raises(SystemExit):
        cli.main(["run", "--mock", "--subject", "subject1"])
