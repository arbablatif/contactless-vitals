# Contactless Vitals

A Python package that extracts a pulse waveform from ordinary face video, using remote photoplethysmography (rPPG). It tracks the face, measures the colour of three skin regions frame by frame, and separates the tiny pulse signal from motion and lighting changes with three classic algorithms: GREEN, CHROM and POS.

This is the signal core. It stops at the waveform and does not estimate heart rate yet.

**Status:** tested on synthetic video, on real MediaPipe tracking of a portrait with a known pulse added, and on one real UBFC-rPPG recording, where the waveform correlates 0.79 with a contact sensor. One person is a first check, not a measure of accuracy.

![With brightness flicker inside the pulse band, GREEN follows the flicker while CHROM and POS follow the true pulse](results/figures/waveforms_in_band_brightness.png)

## Problem

Each heartbeat changes skin colour very slightly, by well under 1% of its brightness. A camera can see this, which makes it possible to monitor a pulse without touching anyone.

The difficulty is that almost everything else changes the picture more than the pulse does. Head movement, flickering lights and shine on the skin are all far stronger. Some of them also repeat at heart-rate speeds, so a frequency filter alone can't remove them.

## Approach

1. **Track the face.** MediaPipe FaceLandmarker follows 478 landmarks through the video.
2. **Pick skin.** Landmarks define the forehead and both cheeks. A skin mask then removes hair, highlights and shadow, using each region's own median colour, so no skin tone is assumed.
3. **Average.** Each region becomes one mean RGB value per frame. Averaging hundreds of pixels is what lifts the pulse above sensor noise.
4. **Clean up timing.** The trace is resampled to even time steps. Short gaps are filled and long ones stay empty, so nothing is invented.
5. **Separate the pulse.** GREEN, CHROM and POS each turn the RGB trace into a pulse signal.
6. **Filter.** Smoothness-priors detrending, then a zero-phase 0.7 to 4 Hz band-pass.
7. **Output.** Waveforms for each method and each region, 10-second sliding windows, and plots.

| Algorithm | Idea | Protection against motion and lighting |
|---|---|---|
| **GREEN** (Verkruysse 2008) | Use the green channel, where the pulse is strongest | None |
| **CHROM** (de Haan 2013) | Combine colour differences so distortions cancel | Good, assumes a standard skin colour |
| **POS** (Wang 2017) | Project out the direction brightness changes move in | Good, learns from the data in each window |

[LEARNING.md](LEARNING.md) explains the physics and each algorithm in plain language.

## Architecture

```mermaid
flowchart LR
    subgraph Sources
        V[UBFC-rPPG video<br/>and ground truth]
        S[Synthetic video<br/>known pulse]
    end
    subgraph Extraction
        L[Face landmarks<br/>MediaPipe]
        R[Forehead and<br/>cheek regions]
        M[Skin mask]
        C[Mean RGB<br/>per frame]
    end
    subgraph Signal
        U[Uniform time grid<br/>and gap handling]
        A[GREEN / CHROM / POS]
        F[Detrend and<br/>0.7-4 Hz band-pass]
    end
    subgraph Outputs
        W[Waveforms and<br/>sliding windows]
        P[Plots]
    end
    V --> L
    S --> L
    L --> R --> M --> C --> U --> A --> F --> W --> P
```

| Module | Purpose |
|---|---|
| `contactless_vitals/data/ubfc.py` | Find subjects, validate files, read video and ground truth |
| `contactless_vitals/data/synthetic.py` | Synthetic videos with a known pulse |
| `contactless_vitals/face/` | Landmarks, region definitions, skin mask |
| `contactless_vitals/extract.py` | Frames to per-region mean RGB |
| `contactless_vitals/signal/` | Preprocessing, the three algorithms, windowing |
| `contactless_vitals/pipeline.py` | RGB trace to waveforms |
| `contactless_vitals/quality.py` | Scores against a known answer |
| `contactless_vitals/plots.py`, `cli.py` | Figures and the command line |

## Results

All numbers come from running `python evaluate.py`. Each scores the waveform against a frequency that is already known, so none of them is a heart-rate estimate. Full tables are in [results/synthetic_results.md](results/synthetic_results.md).

SNR compares the power at the pulse frequency and its second harmonic with everything else in the band. At 0 dB the two are equal, and at 10 dB the pulse has ten times the power. Values are the mean over 5 noise seeds, with the worst seed in brackets.

![Pulse SNR by condition for each algorithm](results/figures/snr_by_scenario.png)

| Condition (pulse at 1.2 Hz) | GREEN | CHROM | POS |
|---|---|---|---|
| Clean | **26.1** (25.2) | 18.7 (17.5) | 18.6 (16.9) |
| Slow lighting drift | **22.0** (21.7) | 18.3 (17.2) | 18.3 (16.5) |
| Brightness flicker inside the band | -15.9 (-16.0) | **18.5** (17.1) | **18.5** (16.9) |
| Specular shine inside the band | -16.8 (-16.9) | 15.6 (15.2) | **18.9** (17.6) |
| Head sway with tracking noise | **24.8** (23.7) | 18.5 (17.5) | 18.4 (17.0) |
| No pulse at all (should be below 0) | -5.8 (-7.9) | -4.7 (-7.9) | -5.0 (-8.5) |

Real MediaPipe tracking of a swaying portrait with a known 1.3 Hz pulse added (single run):

| Portrait through MediaPipe | GREEN | CHROM | POS |
|---|---|---|---|
| SNR (dB) | 11.1 | 16.9 | **17.7** |

What these show:

- **Distortion inside the band is the real test.** A band-pass filter can't help there. GREEN falls by about 42 dB, while CHROM and POS stay near their clean scores.
- **Robustness has a cost.** With nothing disturbing the signal, GREEN is about 7.5 dB ahead, because CHROM and POS add noise from the weaker channels.
- **Real tracking favours CHROM and POS.** In the portrait run GREEN drops to 11.1 dB. The synthetic head-sway test doesn't show this drop because its tracking is close to perfect.
- **No false pulse.** With no pulse in the video, every method stays below 0 dB.
- **Pulse rates across the band work.** From 0.8 to 3.5 Hz every method stays above 14 dB, even at its worst seed.

### Skin tone

| Synthetic skin tone | GREEN | CHROM | POS |
|---|---|---|---|
| Light, full pulse | 28.8 (28.0) | 21.9 (20.9) | 21.6 (20.0) |
| Medium, full pulse | 26.1 (25.2) | 18.7 (17.5) | 18.6 (16.9) |
| Dark, full pulse | 20.7 (19.7) | 12.6 (11.3) | 13.4 (11.7) |
| Dark, half pulse | 14.9 (14.0) | 6.9 (5.8) | 8.2 (6.9) |

Performance falls on darker skin, and this is a real weakness rather than a coding error. Sensor noise stays the same while the reflected light and the pulse shrink. CHROM and POS lose the most because they rely on the dim blue channel. The "half pulse" row is a rough stand-in for melanin absorbing light before it reaches the blood, not a measured value.

The skin mask itself kept enough pixels on all three tones.

### Tests

139 tests pass with `pytest`, including 8 that use the real MediaPipe model. They cover the filter response, the algorithms' cancellation maths, the skin mask, dropped frames, uneven and repeated timestamps, a fake dataset on disk, and `--mock` mode. None of them needs the dataset.

### UBFC-rPPG

One subject so far (`subject1`, 1,547 frames, about 53 seconds), from `python evaluate.py --ubfc subject1`. Scores are against the contact sensor recorded with the video, and are also in [results/ubfc_results.md](results/ubfc_results.md).

| Method | SNR (dB) | Correlation with sensor | Chance level (median / highest) |
|---|---|---|---|
| GREEN | 4.4 | 0.74 | 0.34 / 0.46 |
| CHROM | 4.4 | 0.77 | 0.35 / 0.45 |
| POS | **5.8** | **0.79** | 0.36 / 0.48 |

- **The pulse is really there.** Each correlation is well above its chance level, which comes from sliding the same sensor signal out of step. The chance level is not near zero because the search over small time shifts can line up most rhythms.
- **POS comes first here, but by a small margin.** With one person, that gap says nothing about which method is better in general.
- **Face tracking held throughout.** All three regions were usable in every frame.
- **Don't compare these SNRs directly with the synthetic ones.** A real pulse rate drifts, which spreads its power over more frequencies, and real skin adds noise the simulator doesn't have.

The real file also exposed problems synthetic video couldn't. Its timestamps are uneven and include one repeat and one 0.22 s jump, which broke the loader and threw the rate estimate off (26 Hz instead of 29 Hz). Both are fixed. The recorded times are worth keeping: in a one-off check, assuming an even frame rate instead lowered the POS correlation from 0.79 to 0.70.

## How to run

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS or Linux
pip install -r requirements.txt
```

**Mock mode** runs end to end on a synthetic video, with no dataset, model download or network:

```bash
python -m contactless_vitals run --mock
python -m contactless_vitals run --mock --scenario in_band_brightness --pulse-hz 1.5
```

Output goes to `out/mock_<scenario>/`: `waveforms.csv`, `waveforms.npz` (including sliding windows) and `figures/`.

**Real data.** UBFC-rPPG is not included and must never be committed.

1. Download DATASET_2 from the [authors' page](https://sites.google.com/view/ybenezeth/ubfcrppg) and check its conditions of use.
2. Copy `.env.example` to `.env` and set `UBFC_RPPG_ROOT` to the folder holding `subject1/`, `subject2/` and so on.
3. Validate the download, then run a subject:

```bash
python -m contactless_vitals check-data
python -m contactless_vitals run --subject subject1
```

The 3.7 MB face model downloads to `models/` on first use, and its checksum is verified.

**Tests and evaluation:**

```bash
pytest                        # everything
pytest -m "not mediapipe"     # skip the tests that need the face model
python evaluate.py            # regenerate results/
```

This project makes no LLM or other API calls, so it needs no API keys.

## Limitations

- **One real person so far.** A single recording shows the pipeline works end to end. It can't show accuracy across people, motion or lighting. Synthetic videos follow the same physical model the algorithms assume, so they prove the code is correct, not that it works well on people.
- **The portrait test is limited.** It adds a pulse to a still photo of one person and is a single run. It tests tracking and masking, not real skin behaviour.
- **Skin tone.** Quality falls on darker skin, as shown above. A dataset of 42 videos can't establish that a system works fairly across skin tones.
- **Ground-truth format is checked on one file.** In `subject1`, the three rows have one value per video frame, the time row spans the video in seconds, and the heart-rate row holds plausible values. Other subjects may hold surprises, so the loader still validates each file.
- **DATASET_2 only.** DATASET_1 is not supported.
- **Offline only.** Zero-phase filtering needs the whole recording, so this is not real time.
- **CHROM variant.** Its filtering runs over the whole trace rather than per window, because a 1.6 s window is too short to band-pass at 0.7 Hz. [LEARNING.md](LEARNING.md) explains why.
- **One face, good conditions.** Low light, heavy video compression and large head turns are untested. Compression in particular can erase a signal this small.
- **Not a medical device.** This is a research and portfolio project and must not be used for diagnosis or monitoring.

## References

- Verkruysse, Svaasand, Nelson. *Remote plethysmographic imaging using ambient light.* Optics Express, 2008.
- de Haan, Jeanne. *Robust pulse rate from chrominance-based rPPG.* IEEE Transactions on Biomedical Engineering, 2013.
- Wang, den Brinker, Stuijk, de Haan. *Algorithmic principles of remote PPG.* IEEE Transactions on Biomedical Engineering, 2017.
- Tarvainen, Ranta-aho, Karjalainen. *An advanced detrending method with application to HRV analysis.* IEEE Transactions on Biomedical Engineering, 2002.
- Bobbia, Macwan, Benezeth, Mansouri, Dubois. *Unsupervised skin tissue segmentation for remote photoplethysmography.* Pattern Recognition Letters, 2017. (UBFC-rPPG dataset.)

The portrait used in the tests is the NASA astronaut image bundled with scikit-image, which is in the public domain.
