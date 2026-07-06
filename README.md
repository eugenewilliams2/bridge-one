# BRIDGE ONE — Mix · Master · Reference

A complete mixing + mastering console in **one HTML file**. No installs, no
accounts, no uploads — every sample stays on your device. Built to run great
in mobile Safari, so you can master straight from your phone.

Open `index.html` in any browser. That's the whole deployment story.

## The three rooms

### MIX
- Load individual stems (beat, vocals, adlibs…) **or load one full song and
  hit SPLIT** — the stem engine pulls it apart into VOCALS / DRUMS / BASS / MUSIC
- Per-stem gain, pan, mute, solo, low-cut, air shelf, and compressor
- BOUNCE → MASTER renders the mix offline and drops it straight into the
  mastering chain

### MASTER
- BS.1770-4 gated LUFS analysis, 4× oversampled true-peak metering
- Tone (sub/mid/presence/air), stereo width, tape saturation
- Loudness normalization to target + lookahead true-peak limiter
- Exports a 24-bit WAV, verified to hit the target LUFS and stay under the
  true-peak ceiling

### REF
- Search any song (iTunes catalog, 30-second previews) or load a local file
- Measures the reference's loudness, crest factor, and tonal balance
- Plain-English advice ("reference carries 3 dB more low end — try SUB up")
- MATCH TARGET sets your mastering loudness to the reference
- Loudness-matched reference playback, because louder always sounds "better"

## How the stem split works

DSP source separation, all on-device in a Web Worker: STFT (2048/512, Hann)
with harmonic/percussive median masking for drums, center-channel extraction
for vocals (180 Hz–11 kHz), harmonic low band for bass/808s, and the exact
residual as MUSIC. The four stems sum back to the original sample-for-sample.
It's built for rebalancing — expect some bleed, this is not a neural model.
Works best on stereo mixes.

## Engineering notes

- All heavy DSP (LUFS, true peak, limiting, WAV encode, stem split) runs in a
  Web Worker spawned from an inline blob — the UI never blocks
- K-weighting biquads fused into a single pass with O(blocks) memory
- Spectrum analyzer renders allocation-free per frame
- iOS quirks handled: iCloud placeholder files, audio-context auto-resume,
  silent-switch watchdog, manual WAV parser fallback

Handcrafted for GQ GENO.
