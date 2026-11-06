# Learning Notes: How a Camera Sees a Pulse

These notes explain the ideas behind this project in plain language, so you can walk a client through them without reading the code.

## The short version

Every heartbeat pushes blood into the skin. Blood absorbs light, so skin gets very slightly darker with each beat. A camera can measure that change. The hard part is that movement and lighting change the picture far more than the pulse does. GREEN, CHROM and POS are three increasingly clever ways of pulling the pulse back out.

## Why skin colour changes with blood volume

Light hitting skin does two things:

- **Some bounces straight off the surface.** This is the shine you see on a forehead. It never touched the blood, so it carries no pulse information.
- **Some enters the skin, scatters around and comes back out.** On the way, some of it is absorbed. This light has picked up the skin's colour, and it carries the pulse.

Haemoglobin, the molecule in red blood cells, is a strong absorber of light. When the heart beats, a pressure wave swells the small vessels in the skin and there is briefly more blood in the light's path. More blood means more absorption, so less light returns to the camera. Between beats the vessels relax and slightly more light returns. Pulse oximeters on a fingertip work on the same principle, and that technique is called photoplethysmography (PPG). "Remote PPG" just means doing it from a distance with a camera.

Researchers still debate the exact details, including how much the signal comes from blood volume itself and how much from pressure changing the skin's structure. The practical result is the same: a small, regular, colour-specific change.

### Why green is the strongest channel

Each colour behaves differently:

| Colour | What happens |
|---|---|
| **Blue** | Absorbed strongly, but doesn't reach far into the skin, so it meets fewer pulsing vessels |
| **Green** | Absorbed strongly by haemoglobin and reaches the vessels: the best balance |
| **Red** | Reaches deep, but haemoglobin barely absorbs it, so the change is small |

This gives the pulse a colour fingerprint. In this project's synthetic videos the relative pulse strengths are 0.33 for red, 0.77 for green and 0.53 for blue, the figures used by de Haan and Jeanne.

### How small the signal is

The change is well under 1% of brightness. On a camera with 256 brightness levels, that's about 1 to 2 levels, and a single pixel's random noise is often larger than that.

That's why we average. Random noise partly cancels when pixels are averaged: averaging 400 pixels cuts noise by a factor of 20 (the square root of 400). The pulse is shared by all those pixels, so it doesn't cancel. This one step is what makes the pulse measurable.

It's also why video compression matters. When testing this project, a common lossy codec (MJPG) moved pixels by up to 3 levels, which is more than the pulse itself.

## The problem to solve

A useful way to think about what the camera records for each colour channel:

> **measured colour = light strength x (skin colour + shine + pulse)**

The pulse is the tiny term. Two things interfere with it:

- **Light strength changes.** Lamps flicker, clouds pass, and a head turning towards or away from a light changes how much light reaches the skin. This multiplies all the colours together.
- **Shine changes.** As the head moves, the amount of surface reflection changes. Shine has the colour of the light, not the skin.

**Why not just filter?** A band-pass filter keeps 0.7 to 4 Hz (42 to 240 beats per minute) and removes everything else. That works for slow drift. But a nodding head or a flickering light can sit at 1.9 Hz, right inside the band, and the filter keeps it along with the pulse. Separating those needs colour, not frequency. This is the idea that CHROM and POS are built on.

## The three algorithms

All three begin the same way: divide each channel by its own average. This is called **temporal normalisation**. It removes the skin's base colour and the camera's exposure, so only relative changes are left. After this step, a stable skin colour looks like (1, 1, 1) whoever the person is.

### GREEN: use the best channel

Take the green channel, and that's it.

- **Pulse:** the strongest of the three channels.
- **Motion and lighting:** no protection at all. A brightness change goes straight into the output.
- **When it wins:** when nothing is disturbing the signal, because it adds no noise from weaker channels. In our clean test it scored 26.1 dB against 18.6 dB for the others.
- **When it fails:** with brightness flicker inside the band it fell to -15.9 dB, meaning the flicker had about 40 times the power of the pulse.

### CHROM: make the distortion cancel

**Idea:** build two colour-difference signals in which a brightness change appears equally but the pulse does not. Subtract one from the other, and the distortion cancels while the pulse survives.

The two signals are:

- `X = 3R - 2G`
- `Y = 1.5R + G - 1.5B`

Check what a brightness change does. If all channels rise by a small amount `b`, then X rises by `3b - 2b = b`, and Y rises by `1.5b + b - 1.5b = b`. Both move by exactly the same amount.

The pulse behaves differently. Because of its colour fingerprint, it pushes X and Y in opposite directions.

So the output is `X - alpha x Y`, where `alpha` is the ratio of their strengths in each short window. If distortion dominates, alpha is close to 1 and the shared part cancels. The pulse, moving in opposite directions, adds up instead of cancelling.

- **Strength:** removes brightness changes, and most shine, even inside the pulse band.
- **Weakness:** the fixed numbers in X and Y assume a standard skin colour and white-ish light. Shine had the largest effect on CHROM in our tests: 15.6 dB, against 18.9 dB for POS.

### POS: remove the direction brightness moves in

**Idea:** think of each moment's colour as a point in 3D space, with red, green and blue as the axes. After normalisation, a brightness change moves that point along one known direction: (1, 1, 1). So we look at the signal from a viewpoint where that direction disappears.

Mathematically, that's a projection onto the plane at right angles to (1, 1, 1). POS uses two signals within that plane:

- `S1 = G - B`
- `S2 = G + B - 2R`

Check a brightness change again: `S1` becomes `b - b = 0` and `S2` becomes `b + b - 2b = 0`. Brightness doesn't just shrink, it disappears exactly. Our tests confirm this to within 1e-9.

Shine isn't removed by the projection, because it doesn't lie along (1, 1, 1). So POS combines the two signals as `S1 + alpha x S2`, with alpha worked out from the data in each 1.6-second window. This balances them so that shine cancels and the pulse remains.

- **Strength:** the only assumption is the data's own skin tone, so it adapts to each person and each moment. It was the most consistent in our tests.
- **Weakness:** like CHROM, it uses all three channels, so noise in a dim channel gets in.

### Side by side

| | GREEN | CHROM | POS |
|---|---|---|---|
| How it separates the pulse | Doesn't; picks the best channel | Fixed colour differences that cancel | Projection plus data-driven balance |
| Brightness changes | Passes straight through | Cancelled | Removed exactly |
| Shine | Passes straight through | Mostly cancelled | Balanced out per window |
| Assumes standard skin colour | No | Yes | No |
| Cost | None | Some extra noise | Some extra noise |

Why use short windows at all? Lighting and motion keep changing, so `alpha` must keep up. Each 1.6-second window gets its own alpha, and overlapping windows are blended together to avoid jumps at their joins. This is called overlap-add. CHROM uses tapered windows that overlap by half, as in its paper. POS steps one sample at a time, as in its paper.

## Design decisions

**Why forehead and cheeks?** They have a good blood supply, they're fairly flat, and they don't move much when someone talks or blinks. The landmark outlines stay clear of the eyes, brows and mouth.

**Why does the skin mask use relative thresholds?** Fixed colour limits are a common shortcut, but they can work less well on darker skin. Instead, each region's median colour defines what skin looks like for this person under this light, and pixels far from that median are dropped. This removes hair, glasses, highlights and shadow without assuming a skin tone.

**Why keep regions separate?** A shadow or fringe may spoil one region and not the others. Keeping them separate shows which regions are reliable. The combined signal first scales each region by its own average. Otherwise, one region dropping out briefly would make the mix jump in colour, and that jump would look like a signal.

**Why resample to a uniform grid?** Webcams don't deliver frames at perfectly even intervals, and filters assume evenly spaced samples.

**Why leave long gaps empty?** If the face is lost for a moment, a straight line across the gap is harmless. Doing that across two seconds would invent data. So gaps up to 0.5 seconds are filled, longer ones stay empty, and each good stretch is processed separately so filters can't smear across a gap.

**Why smoothness-priors detrending?** It finds the smoothest curve through the signal and subtracts it, which follows curved lighting drift better than a straight line. It also has none of a high-pass filter's start-up transient. We set it by cutoff frequency (0.25 Hz) so it behaves the same at any frame rate.

**Why a zero-phase filter?** An ordinary filter delays the signal, and the delay differs by frequency, which distorts the pulse shape. Running the filter forwards and then backwards cancels the delay. The cost is that it needs the whole recording, so it can't run live. It also doubles the attenuation, so we widen the design slightly to keep the true -3 dB points at 0.7 and 4 Hz.

**Why does CHROM differ slightly from the paper?** A 1.6-second window holds only about one cycle at 0.7 Hz, too short to band-pass well. So this version normalises with a moving average and filters X and Y over the whole trace, then calculates alpha and blends windows as the paper does.

**Why does every output have the same polarity?** Raw brightness falls when blood rises, so without a convention some methods would come out upside down. Here, up always means more blood.

**Why no heart rate yet?** Heart rate is a separate problem with its own decisions, and it can hide a bad waveform. Getting the waveform right first means later errors can be traced to the right stage. The tests do check energy at a frequency, but they're given the answer and check against it. Nothing in the package searches for one.

## How the code was tested

**Synthetic videos.** A generator builds videos from the model above, so the correct answer is known. Each condition targets one claim, such as "CHROM and POS survive in-band brightness changes". There's also a negative control: a video with no pulse, where a good pipeline must report nothing.

**A real face for MediaPipe.** MediaPipe won't reliably detect a drawn face, so the tracking test adds a known pulse to the public-domain astronaut portrait bundled with scikit-image, and moves it side to side.

**What the results do and don't prove.** Synthetic videos follow the same model the algorithms assume. They prove the code does what the theory says. They don't prove it works on real people, which is what UBFC-rPPG is for.

## Findings worth discussing with a client

- **There's no single best algorithm.** GREEN wins in calm conditions, and CHROM and POS win when things move. Robustness has a real cost, about 7.5 dB in our clean test.
- **Darker skin gave lower quality, even in simulation.** Dark skin with full pulse strength scored 12.6 to 13.4 dB for CHROM and POS, against about 22 dB for light skin. At half strength they fell to 6.8 and 8.2 dB. There's less reflected light for the same sensor noise, and CHROM and POS depend on the dim blue channel. Better lighting helps; code alone doesn't fully fix this. Any real product must be tested across skin tones.
- **Video quality is part of the system.** Compression, low light and auto-exposure can all destroy a signal of 1 to 2 grey levels. The camera setup matters as much as the algorithm.

## Questions a client might ask

**Is this a medical device?** No. It's a signal-processing project. Medical use would need clinical validation and regulatory approval.

**Does it work in the dark?** No. It relies on visible light reflected from skin.

**Could it work from a phone or webcam?** The method needs only an ordinary camera, but heavy video compression can damage the signal.

**Does it record anyone's face?** The pipeline only needs colour averages. For this project, dataset videos stay local and are never committed, and no figures show a participant's face.

**How accurate is it?** Not known yet. The waveform works on synthetic data. Accuracy on real people needs the UBFC-rPPG run, and heart-rate accuracy needs the next stage.

## Glossary

| Term | Meaning |
|---|---|
| **PPG** | Measuring pulse from changes in light absorbed by blood |
| **rPPG** | PPG from a distance, using a camera |
| **Landmark** | A tracked point on the face |
| **ROI** | Region of interest: the part of the image being measured |
| **Temporal normalisation** | Dividing a signal by its own average to leave only relative change |
| **Chrominance** | A colour-difference signal, with brightness removed |
| **Specular reflection** | Surface shine, carrying the colour of the light |
| **Band-pass filter** | Keeps a range of frequencies and removes the rest |
| **Detrending** | Removing slow drift |
| **Overlap-add** | Blending overlapping windows into one continuous signal |
| **SNR** | Signal-to-noise ratio, in decibels. Each 10 dB is ten times the power |
