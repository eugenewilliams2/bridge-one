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
- MASTER tab grows an **AI ENHANCE (DEEP ENGINE)** button — full auto-master

## What AI ENHANCE actually does

1. **Measures** LUFS, true peak, crest, correlation, noise floor, DC offset,
   clipping, and a 30-band long-term spectrum vs a genre target curve
2. **Reports issues** — mud, harshness, dull top, weak/excessive lows,
   narrow image, over-compression, resonances
3. **Separates stems** (Demucs htdemucs) and processes each adaptively:
   vocal HPF + de-essing + presence, drum transient recovery, 808 mono +
   saturation, mud clearing in the music bed
4. **Adaptive bus chain**: resonance notches (≤3, measured), linear-phase
   matching EQ toward the genre curve (±4 dB cap, A/B gated — reverted if it
   doesn't measurably improve the spectrum), dynamic low-band control, gentle
   glue, optional saturation, mono-safe stereo width
5. **Finishes** at your target LUFS with a lookahead true-peak limiter and
   verifies the result

Every action is listed in the report with the measured reason. On
well-balanced material the engine does very little — by design.

## Endpoints (for the curious)

`GET /health` · `POST /split` · `POST /enhance?genre=hiphop|rnb|pop&lufs=-9.5`
· `GET /status?id=` · `GET /stem?id=&name=` · `GET /report` · `GET /result`

## Tuning

- `BRIDGESPLIT_MODEL=htdemucs_ft` — maximum separation quality, ~4× slower
- `BRIDGESPLIT_DEVICE=mps` — try Apple GPU (default cpu is safest on 8 GB)
- `BRIDGESPLIT_PORT=8765`

Requires: Python 3.10+, ffmpeg (`brew install ffmpeg`).
