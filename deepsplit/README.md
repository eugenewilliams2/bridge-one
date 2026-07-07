# BRIDGE ONE — Deep Engine

The desktop AI companion to BRIDGE ONE: neural stem separation (Demucs) and
an analyze-then-enhance auto-mastering pipeline. Runs 100% locally on your
Mac — no uploads, no accounts.

## Start it

Double-click **`Deep Split.command`** in the repo root (first run installs
the engine into `deepsplit/.venv` and downloads model weights). Keep the
Terminal window open. A status page opens at http://127.0.0.1:8765 with its
own drag-and-drop UI, and BRIDGE ONE (`index.html`) auto-detects the engine:

- MIX tab grows a **DEEP SPLIT (AI)** button — neural VOCALS/DRUMS/BASS/MUSIC
- MIX tab grows an **AI MIX (AUTO-BALANCE)** button — full auto-mix
- MASTER tab grows an **AI ENHANCE (DEEP ENGINE)** button — full auto-master

## What AI MIX actually does

Same discipline as AI ENHANCE, but it works *between* the elements and stops
at a balanced mix (not a master), leaving headroom for mastering:

1. **Separates** the track into stems (Demucs) and processes each adaptively
   (vocal HPF + de-ess + presence, drum transients, 808 mono + saturation,
   music-bed mud clearing)
2. **Unmasks the vocal** — carves ≤3 pockets in the music where the vocal is
   strongest so it cuts through without turning it up (A/B gated on a vocal-
   clarity metric; reverted if it doesn't help)
3. **Widens** the music bed to open space around the center vocal
4. **Sets the balance** from genre targets (vocal on top, drums under, bass
   controlled, music tucked back), clamped to ±12 dB per stem
5. **Gain-stages** the sum to ~-16 LUFS with ≥1 dB headroom — **no limiting**

Loads the balanced stems back into the mixer (faders at 0 = the AI balance)
so you can nudge anything, then **BOUNCE → MASTER** or **AI ENHANCE** to finish.

## What AI ENHANCE actually does

1. **Measures** LUFS, true peak, crest, correlation, noise floor, DC offset,
   clipping, and a 30-band long-term spectrum vs a genre target curve
2. **Reports issues** — mud, harshness, dull top, weak/excessive lows,
   narrow image, over-compression, resonances
3. **Separates stems** (Demucs htdemucs) and processes each adaptively:
   vocal HPF + de-essing + presence, drum transient recovery, 808 mono +
   saturation, mud clearing in the music bed
4. **Adaptive bus chain (Ozone-parity modules, all measured + A/B gated):**
   - **Dynamic EQ** — tames each resonant band only when it spikes (≤6 dB)
   - **Stabilizer** — broadband adaptive resonance smoothing: per STFT frame it
     builds a smooth cepstral spectral envelope and pulls down any peak poking
     above it (≤~4 dB, strength adapts to source harshness). A/B gated so it
     never dulls the track
   - **Matching EQ** — linear-phase move toward the genre curve (±4 dB cap,
     reverted if it doesn't measurably improve the spectrum)
   - **Multiband compressor** — 4-band (low/low-mid/mid/high), gentle ratios,
     ≤4 dB/band, per-band timing; A/B gated, falls back to broadband glue
   - **Harmonic exciter** — generates air (>6 kHz) and/or low-end weight when
     the tonal analysis says a band is lacking (harmonics, not just EQ)
   - **Multiband imager** — lows mono, mids natural, highs widened, mono-safe
5. **Finishes** at your target LUFS with a lookahead true-peak limiter and
   verifies the result

The band splitter is perfect-reconstruction (complementary subtraction — the
bands sum back to the source at ~-126 dBFS), so unprocessed bands stay clean.

**Linear-phase mode** (`?linphase=1`, or the LINEAR PHASE toggle in MASTER):
the matching EQ and multiband splits become phase-coherent (windowed-sinc FIR
crossovers, zero transient smearing) at the cost of a slower render. Off by
default (minimum-phase IIR — standard and fast).

**Tonal Balance meter**: every AI MIX / AI ENHANCE report includes a
`balance_meter` payload (30 log-spaced bands: genre target, before, after)
that the app draws as an Ozone-style curve-vs-target-pocket chart, with a
LOW/MID/HIGH numeric readout of how far the result sits from target.

Every action is listed in the report with the measured reason. On
well-balanced material the engine does very little — by design.

## Endpoints (for the curious)

`GET /health` · `POST /split` · `POST /enhance?genre=hiphop|rnb|pop&lufs=-9.5`
· `POST /aimix?genre=hiphop|rnb|pop&lufs=-16` · `GET /status?id=`
· `GET /stem?id=&name=` · `GET /report` · `GET /result`

`/aimix` writes the 4 balanced stems (fetch via `/stem?name=`) plus the summed
mix (`/result`); the report includes per-stem `balance` in dB.

`POST /adjust` (JSON `{ops, genre, lufs, ceiling}`) applies a list of cleanup
ops (`lowshelf`/`highshelf`/`bell`/`deess`/`width`/`saturate`/`dereverb`/
`reverb`) to the last AI master and re-limits — synchronous, returns fresh `scores`/`after`/
`balance_meter`. Powers **MASTER CHAT**: the app parses plain-English notes
("less low end, open up the top, tame the harshness") into ops locally and
re-applies the full cumulative set to the original master each time (no
generation loss; `reset` clears it, `save` downloads the result).

## Tuning

- `BRIDGESPLIT_MODEL=htdemucs_ft` — maximum separation quality, ~4× slower
- `BRIDGESPLIT_DEVICE=mps` — try Apple GPU (default cpu is safest on 8 GB)
- `BRIDGESPLIT_PORT=8765`

Requires: Python 3.10+, ffmpeg (`brew install ffmpeg`).
