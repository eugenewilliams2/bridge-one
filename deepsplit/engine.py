"""BRIDGE ONE — Deep Enhance engine.

Hybrid DSP + ML audio processing: analyze the track, split it into stems
with Demucs, apply adaptive per-stem and bus processing, and finish at
commercial loudness. Every corrective stage is driven by measurements, is
bounded so it stays subtle on well-balanced material, and the big tonal
moves are A/B gated — if a stage doesn't measurably move the track toward
the genre target, it's reverted.

All torch/CPU. Nothing leaves the machine.
"""
import math, os, subprocess, tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

# ---------------------------------------------------------------- io
def load_audio(path):
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception:
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "2", "-acodec", "pcm_f32le", tmp],
                       check=True, capture_output=True)
        data, sr = sf.read(tmp, always_2d=True, dtype="float32")
        os.unlink(tmp)
    x = torch.from_numpy(np.ascontiguousarray(data.T))
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    return x[:2].contiguous(), sr

def save_wav24(x, sr, path):
    sf.write(path, x.T.numpy(), sr, subtype="PCM_24")

# ---------------------------------------------------------------- metering
def lufs(x, sr):
    try:
        v = float(AF.loudness(x, sr))
        return v if math.isfinite(v) else -70.0
    except Exception:
        return -70.0

def true_peak_db(x, sr):
    step = max(1, x.shape[1] // (sr * 240))  # cap the resample cost on long files
    y = AF.resample(x[:, ::step] if step > 1 else x, sr, sr * 4)
    return 20 * math.log10(float(y.abs().max()) + 1e-12)

def crest_db(x):
    peak = float(x.abs().max())
    rms = float(x.pow(2).mean().sqrt())
    return 20 * math.log10((peak + 1e-12) / (rms + 1e-12))

def correlation(x):
    l, r = x[0] - x[0].mean(), x[1] - x[1].mean()
    return float((l * r).sum() / ((l.pow(2).sum() * r.pow(2).sum()).sqrt() + 1e-12))

N_BANDS = 30
BAND_HZ = np.geomspace(25, 18000, N_BANDS)

def band_spectrum(x, sr, fine=False):
    """Long-term average spectrum in dB on a log grid (30 bands, or 1/24-oct fine grid)."""
    mono = x.mean(dim=0)
    win = 8192
    if mono.numel() < win * 2:
        mono = torch.nn.functional.pad(mono, (0, win * 2 - mono.numel()))
    hop = max(win, mono.numel() // 400)  # ≤ ~400 frames
    frames = mono.unfold(0, win, hop) * torch.hann_window(win)
    mag = torch.fft.rfft(frames, dim=1).abs().pow(2).mean(dim=0).sqrt().numpy()
    freqs = np.fft.rfftfreq(win, 1 / sr)
    grid = np.geomspace(25, min(18000, sr / 2 - 200), 240) if fine else BAND_HZ
    out = np.zeros(len(grid))
    for i, f in enumerate(grid):
        lo, hi = f / 1.12, f * 1.12
        sel = mag[(freqs >= lo) & (freqs <= hi)]
        out[i] = 20 * np.log10(sel.mean() + 1e-12) if sel.size else -120.0
    return grid, out

def normalize_curve(bands_db):
    """Anchor a spectrum at its 300 Hz–3 kHz mean so curves compare shape, not level."""
    sel = (BAND_HZ >= 300) & (BAND_HZ <= 3000)
    return bands_db - bands_db[sel].mean()

# Genre tonal targets (dB relative to mids) — approximations of the long-term
# spectra of modern commercial masters in each lane.
def _target(points):
    f = np.array([p[0] for p in points]); g = np.array([p[1] for p in points])
    return np.interp(np.log10(BAND_HZ), np.log10(f), g)

# Genre targets — the median long-term spectrum of real commercial masters
# (12 tracks/genre via iTunes; Travis from 8). The very top (>10 kHz) is
# tempered up from the raw measurement to undo AAC-preview rolloff. These are
# how good masters ACTUALLY measure: bass-heavy, gently darkening top — not the
# flat/bright shapes a naïve target assumes.
TARGETS = {
    "hiphop": _target([(25, 11), (45, 11), (80, 11), (120, 10), (200, 8.3), (400, 5.1),
                       (800, 0.1), (1500, -3.1), (3000, -8.3), (5000, -10), (8000, -11),
                       (12000, -11), (18000, -11)]),
    "rnb":    _target([(25, 8), (45, 10), (80, 10), (120, 10), (200, 9.7), (400, 5),
                       (800, 1), (1500, -3.1), (3000, -9.6), (5000, -11), (8000, -12),
                       (12000, -12), (18000, -12)]),
    "pop":    _target([(25, 5), (45, 10), (80, 10), (120, 10), (200, 7.7), (400, 5.3),
                       (800, 1), (1500, -2.8), (3000, -9.7), (5000, -11), (8000, -11.5),
                       (12000, -11.5), (18000, -11.5)]),
    "travis": _target([(25, 12), (45, 16), (80, 16), (120, 14.4), (200, 9.3), (400, 5.7),
                       (800, -0.2), (1500, -3.7), (3000, -8.6), (5000, -10.7), (8000, -12),
                       (12000, -12), (18000, -12)]),
    "toliver": _target([(25, 14.8), (45, 16), (80, 16), (120, 12.1), (200, 7.3), (400, 4.8),
                        (800, 0.9), (1500, -2.9), (3000, -8.2), (5000, -10.7), (8000, -12),
                        (12000, -12), (18000, -12)]),
    # GQ GENO house curve — the melodic-trap voicing GQ picked in a blind A/B of
    # his own track (started from the Toliver lane; tune this as his sound evolves).
    "geno":   _target([(25, 14.8), (45, 16), (80, 16), (120, 12.1), (200, 7.3), (400, 4.8),
                       (800, 0.9), (1500, -2.9), (3000, -8.2), (5000, -10.7), (8000, -12),
                       (12000, -12), (18000, -12)]),
    # GQ GENO — PUNCHY: the alt "D" master he liked — tighter/smaller low end and
    # more dynamics than the main lane. Controlled-sub voicing + a firmer sub
    # control + a touch quieter (see GENRE_PROFILE).
    "geno_punch": _target([(25, 11), (45, 11), (80, 11), (120, 10), (200, 8.3), (400, 5.1),
                           (800, 0.1), (1500, -3.1), (3000, -8.3), (5000, -10), (8000, -11),
                           (12000, -11), (18000, -11)]),
}
# Per-genre reference loudness (median of the same real masters).
GENRE_LUFS = {"hiphop": -8.9, "rnb": -10.2, "pop": -8.6, "travis": -8.3, "toliver": -9.3,
              "geno": -9.3, "geno_punch": -10.0}
# Per-genre reference correlation — how wide real masters in each lane actually
# are. The imager uses this so it never over-widens past the genre norm.
GENRE_CORR = {"hiphop": 0.90, "rnb": 0.85, "pop": 0.75, "travis": 0.90, "toliver": 0.89,
              "geno": 0.89, "geno_punch": 0.90}

# Per-lane processing profile — the sub-control / clarity / loudness a lane wants
# when the caller doesn't override them. Lanes not listed use the enhance()
# fallbacks (sub_control 0.6, clarity 0.6, and the caller's target LUFS).
GENRE_PROFILE = {
    # GQ GENO — the "v2" clarity voicing GQ picked: de-mud + slight de-box, vocal
    # pushed forward at 3.2 kHz, a touch of air on top. Keeps the big 808 (gentle
    # sub control). This is his signature master.
    "geno":       {"sub_control": 0.6,
                   "clarity_eq": {"mud": 2.0, "box": 1.0, "presence": 2.0, "air": 1.5}},
    # GQ GENO — PUNCHY: same v2 clarity voicing as the main lane, but tighter low
    # and more dynamic (louder crest, a touch quieter).
    "geno_punch": {"sub_control": 1.0, "target_lufs": -10.0,
                   "clarity_eq": {"mud": 2.0, "box": 1.0, "presence": 2.0, "air": 1.5}},
}

# ---------------------------------------------------------------- primitives
def biquad(kind, sr, f0, Q, gain_db=0.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * Q)
    if kind == "lowshelf":
        b0 = A * ((A + 1) - (A - 1) * cw + 2 * math.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cw + 2 * math.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - 2 * math.sqrt(A) * alpha
    elif kind == "highshelf":
        b0 = A * ((A + 1) + (A - 1) * cw + 2 * math.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cw + 2 * math.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - 2 * math.sqrt(A) * alpha
    elif kind == "peak":
        b0, b1, b2 = 1 + alpha * A, -2 * cw, 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * cw, 1 - alpha / A
    elif kind == "highpass":
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "lowpass":
        b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    else:
        raise ValueError(kind)
    b = torch.tensor([b0 / a0, b1 / a0, b2 / a0])
    a = torch.tensor([1.0, a1 / a0, a2 / a0])
    return b, a

def apply_biquad(x, b, a):
    return AF.lfilter(x, a_coeffs=a.to(x.dtype), b_coeffs=b.to(x.dtype), clamp=False)

def onepole(x, sr, ms):
    a = 1 - math.exp(-1 / (ms * 0.001 * sr))
    return AF.lfilter(x, a_coeffs=torch.tensor([1.0, -(1 - a)]),
                      b_coeffs=torch.tensor([a, 0.0]), clamp=False)

def fir_from_curve(grid_hz, gains_db, sr, ntaps=4097):
    """Linear-phase FIR matching an arbitrary log-grid gain curve."""
    nfft = 1 << (ntaps.bit_length() + 1)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    lg = np.interp(np.log10(np.maximum(freqs, 1.0)), np.log10(grid_hz), gains_db,
                   left=gains_db[0], right=gains_db[-1])
    mag = 10 ** (lg / 20)
    ir = np.roll(np.fft.irfft(mag), nfft // 2)
    mid = nfft // 2
    ir = ir[mid - ntaps // 2: mid + ntaps // 2 + 1] * np.hanning(ntaps)
    return torch.tensor(ir, dtype=torch.float32)

def fft_convolve(x, h):
    n = x.shape[1] + h.numel() - 1
    nfft = 1 << (n - 1).bit_length()
    X = torch.fft.rfft(x, nfft)
    H = torch.fft.rfft(h, nfft)
    y = torch.fft.irfft(X * H, nfft)[:, :n]
    delay = h.numel() // 2
    return y[:, delay:delay + x.shape[1]]

def sliding_max(v, k):
    return torch.nn.functional.max_pool1d(v[None, None], kernel_size=2 * k + 1,
                                          stride=1, padding=k)[0, 0]

# ---------------------------------------------------------------- analysis
def analyze(x, sr):
    grid, spec = band_spectrum(x, sr)
    norm = normalize_curve(spec)
    m = {
        "lufs": round(lufs(x, sr), 1),
        "true_peak_db": round(true_peak_db(x, sr), 2),
        "crest_db": round(crest_db(x), 1),
        "correlation": round(correlation(x), 3),
        "duration_s": round(x.shape[1] / sr, 1),
        "sample_rate": sr,
        "clipped_samples": int((x.abs() > 0.999).sum()),
        "rms_db": round(20 * math.log10(float(x.pow(2).mean().sqrt()) + 1e-12), 1),
        "dc_offset": round(float(x.mean()), 5),
        "spectrum_db": [round(v, 2) for v in norm.tolist()],
    }
    # noise floor: 5th percentile of short-window RMS
    win = int(0.05 * sr)
    n = (x.shape[1] // win) * win
    if n:
        rms = x[:, :n].reshape(2, -1, win).pow(2).mean(dim=(0, 2)).sqrt()
        m["noise_floor_db"] = round(20 * math.log10(float(np.percentile(rms.numpy(), 5)) + 1e-12), 1)
    else:
        m["noise_floor_db"] = -120.0
    return m, norm

def out_spectrum(x, sr):
    """Mid-anchored long-term spectrum of a signal, for the tonal-balance meter."""
    _, spec = band_spectrum(x, sr)
    return [round(float(v), 2) for v in normalize_curve(spec).tolist()]

def ref_profile(path):
    """Analyze a reference track → the target payload for reference-matching:
    its measured 30-band curve, loudness, correlation, and crest."""
    x, sr = load_audio(path)
    m, _ = analyze(x, sr)
    return {"curve": m["spectrum_db"], "lufs": m["lufs"],
            "corr": m["correlation"], "crest": m["crest_db"]}

def balance_meter(target, before, after):
    """Payload the app's Tonal Balance meter draws: log-freq band centers, the
    genre target curve, and the source (before) + processed (after) spectra."""
    return {"hz": [round(float(f)) for f in BAND_HZ],
            "target": [round(float(v), 2) for v in target.tolist()],
            "before": before, "after": after}

def region_db(norm, lo, hi):
    sel = (BAND_HZ >= lo) & (BAND_HZ <= hi)
    return float(norm[sel].mean())

def detect_issues(m, norm, target):
    d = norm - target
    issues = []
    def reg(lo, hi):
        sel = (BAND_HZ >= lo) & (BAND_HZ <= hi)
        return float(d[sel].mean())
    if reg(35, 110) < -2.0: issues.append(f"weak low end ({reg(35,110):+.1f} dB vs target)")
    if reg(35, 110) > 3.0: issues.append(f"excessive low end ({reg(35,110):+.1f} dB vs target)")
    if reg(200, 500) > 2.2: issues.append(f"muddiness in the low mids ({reg(200,500):+.1f} dB)")
    if reg(2500, 5000) > 2.2: issues.append(f"harshness in the presence region ({reg(2500,5000):+.1f} dB)")
    if reg(8000, 16000) < -2.5: issues.append(f"dull top end ({reg(8000,16000):+.1f} dB)")
    if m["crest_db"] < 8: issues.append(f"over-compressed dynamics (crest {m['crest_db']} dB)")
    if m["correlation"] > 0.97: issues.append("very narrow stereo image (near mono)")
    if m["correlation"] < 0.15: issues.append(f"phase risk — low L/R correlation ({m['correlation']})")
    if m["clipped_samples"] > 200: issues.append(f"clipping in the source ({m['clipped_samples']} samples)")
    # only call it noise if there actually are quiet stretches well below the music
    if m["noise_floor_db"] > -55 and m["noise_floor_db"] < m["rms_db"] - 15:
        issues.append(f"raised noise floor ({m['noise_floor_db']} dBFS)")
    if abs(m["dc_offset"]) > 0.002: issues.append("DC offset")
    return issues

def find_resonances(x, sr, max_n=3):
    grid, fine = band_spectrum(x, sr, fine=True)
    smooth = np.convolve(fine, np.ones(25) / 25, mode="same")
    diff = fine - smooth
    peaks = []
    for i in range(2, len(grid) - 2):
        if 80 <= grid[i] <= 8000 and diff[i] > 6 and diff[i] >= diff[i-1] and diff[i] >= diff[i+1]:
            peaks.append((float(diff[i]), float(grid[i])))
    peaks.sort(reverse=True)
    out, used = [], []
    for exc, f in peaks:
        if all(abs(math.log2(f / u)) > 0.5 for u in used):
            out.append((f, min(6.0, exc - 3.0)))
            used.append(f)
        if len(out) >= max_n:
            break
    return out

# ---------------------------------------------------------------- quality score
def quality_score(x, sr, target):
    _, spec = band_spectrum(x, sr)
    d = normalize_curve(spec) - target
    w = np.where((BAND_HZ > 60) & (BAND_HZ < 12000), 1.0, 0.5)
    sd = float(np.sqrt(np.mean(w * d * d)))
    cr = crest_db(x)
    pen = max(0.0, 8 - cr) * 0.7
    co = correlation(x)
    pen += max(0.0, 0.15 - co) * 3
    return sd + pen

def excerpt(x, sr, secs=30):
    n = x.shape[1]
    if n <= secs * sr: return x
    mid = n // 2
    return x[:, mid - secs * sr // 2: mid + secs * sr // 2]

# ---------------------------------------------------------------- genre estimate
# Feature centroids per genre: [sub, low-mid, presence, air, crest]. Derived from
# the typical spectral/dynamic signature of each lane — an estimate from the
# audio itself, not a trained model (and the user can always override it).
GENRE_FP = {
    "hiphop": [5.0, 1.5, -0.5, -1.0, 10.5],
    "rnb":    [3.0, 1.0, -0.5,  0.5, 12.5],
    "pop":    [2.0, 0.0,  0.5,  1.5, 11.0],
}
GENRE_W = [1.4, 0.7, 0.8, 1.2, 0.5]  # sub / air / crest discriminate most

def classify_genre(m, norm):
    feats = [region_db(norm, 35, 120), region_db(norm, 150, 400),
             region_db(norm, 2000, 5000), region_db(norm, 9000, 16000),
             (m["crest_db"] - 11.0)]  # centered so crest scale matches dB feats
    fp = {g: v[:4] + [v[4] - 11.0] for g, v in GENRE_FP.items()}
    dists = {}
    for g, c in fp.items():
        d = sum(GENRE_W[i] * (feats[i] - c[i]) ** 2 for i in range(5))
        dists[g] = d
    # softmax over negative distance → confidence
    import math as _m
    exps = {g: _m.exp(-d / 6.0) for g, d in dists.items()}
    tot = sum(exps.values()) or 1.0
    conf = {g: exps[g] / tot for g in exps}
    best = max(conf, key=conf.get)
    return best, round(conf[best], 2), {g: round(conf[g], 2) for g in conf}

# ---------------------------------------------------------------- quality score
def _clamp100(v):
    return int(round(max(0.0, min(100.0, v))))

def score_result(x, sr, target, src_m, target_lufs):
    """Honest 0–100 scores from the actual measured output — every number
    traces to a measurement, nothing is invented."""
    _, spec = band_spectrum(x, sr)
    nc = normalize_curve(spec)
    # judge the broad tonal tilt, not band-to-band ripple — smooth the curve
    # (a 5-band triangular window) before comparing to the target
    ker = np.array([1, 2, 3, 2, 1], float); ker /= ker.sum()
    ncs = np.convolve(nc, ker, mode="same")
    # only judge where there's content — bands >45 dB below the loudest are empty
    present = np.clip((spec - (spec.max() - 45)) / 10.0, 0.0, 1.0)
    w = np.where((BAND_HZ > 50) & (BAND_HZ < 14000), 1.0, 0.4) * present
    d = np.clip(ncs - target, -12, 12)
    wsum = float(w.sum()) + 1e-9
    dev = float(np.sqrt((w * d * d).sum() / wsum))
    # presence/mud/harshness judged vs the genre target, not an absolute bright-mix
    # assumption — so an intentionally dark, on-target hip-hop master isn't punished
    # for being dark; only a master more buried/muddier than the genre norm is.
    def treg(lo, hi):
        sel = (BAND_HZ >= lo) & (BAND_HZ <= hi)
        return float(target[sel].mean())
    pres = region_db(nc, 2000, 5000) - treg(2000, 5000)   # + = more present than target
    mud = region_db(nc, 200, 500) - treg(200, 500)        # + = muddier than target
    harsh = max(0.0, (region_db(nc, 2500, 5000) - treg(2500, 5000)) - 3.0)
    cr = crest_db(x)
    co = correlation(x)
    lu = lufs(x, sr)
    tp = true_peak_db(x, sr)

    # penalties kick in past a real-world mastering tolerance, so a genuinely
    # good, on-target result reaches the 90s and only real problems drag it down
    balance = _clamp100(100 - max(0.0, dev - 1.5) * 11)
    clarity = _clamp100(100 - min(45.0, max(0.0, -1.0 - pres) * 10) - min(25.0, max(0.0, mud - 2.5) * 8))
    dynamics = _clamp100(100 - max(0.0, abs(cr - 12.0) - 2.0) * 8 - max(0.0, 8.0 - cr) * 8)
    punch = _clamp100(100 - max(0.0, src_m["crest_db"] - cr - 2.0) * 10 - max(0.0, 10.0 - cr) * 6)
    stereo = _clamp100(100 - max(0.0, co - 0.96) * 350 - max(0.0, 0.2 - co) * 220 - max(0.0, 0.5 - co) * 30)
    loudness = _clamp100(100 - max(0.0, abs(lu - target_lufs) - 0.5) * 14)
    translation = _clamp100(100 - max(0.0, 0.1 - co) * 250 - harsh * 10 - max(0.0, tp) * 12)
    depth = _clamp100(dynamics * 0.4 + stereo * 0.3 + balance * 0.3)
    mix = _clamp100(balance * 0.30 + clarity * 0.25 + punch * 0.20 + dynamics * 0.15 + stereo * 0.10)
    master = _clamp100(loudness * 0.25 + balance * 0.25 + translation * 0.20 + clarity * 0.15 + dynamics * 0.15)
    overall = _clamp100(0.20 * balance + 0.20 * clarity + 0.15 * dynamics + 0.10 * punch +
                        0.10 * stereo + 0.10 * loudness + 0.15 * translation)
    return {"overall": overall, "mix": mix, "master": master, "clarity": clarity,
            "punch": punch, "balance": balance, "depth": depth, "stereo": stereo,
            "dynamics": dynamics, "loudness": loudness, "translation": translation}

# ---------------------------------------------------------------- stem stage
_sep_model = None
def separate(x, sr, model_name, device, prog):
    """Run Demucs directly (no demucs.api in this build): returns stem dict + model sr."""
    global _sep_model
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    if _sep_model is None:
        _sep_model = get_model(model_name)
        _sep_model.eval()
    msr = _sep_model.samplerate
    wav = AF.resample(x, sr, msr) if sr != msr else x
    ref = wav.mean(0)
    wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(_sep_model, wav_n[None], device=device, split=True,
                          segment=7, overlap=0.15, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    return dict(zip(_sep_model.sources, out)), msr

def deess(v, sr, actions):
    band = apply_biquad(v, *biquad("highpass", sr, 5500, 0.707))
    band = apply_biquad(band, *biquad("lowpass", sr, 9500, 0.707))
    env = onepole(band.pow(2).mean(dim=0, keepdim=True), sr, 4).sqrt()
    voiced = env[env > env.max() * 0.05]
    if voiced.numel() < 100: return v
    thr = float(voiced.median()) * 2.5
    over = (env / thr).clamp(min=1.0)
    gr = 1.0 - (1.0 - over.pow(-0.6)).clamp(0, 0.6)   # up to ~8 dB into the band
    if float((gr < 0.9).float().mean()) < 0.01: return v
    gr = onepole(gr, sr, 15)
    actions.append("vocals: de-essed sibilant peaks (dynamic 5.5–9.5 kHz)")
    return v - band * (1 - gr)

def process_stems(stems, sr, m, norm, target, actions):
    d = norm - target
    def reg(lo, hi):
        sel = (BAND_HZ >= lo) & (BAND_HZ <= hi)
        return float(d[sel].mean())
    v = stems["vocals"]
    v = apply_biquad(v, *biquad("highpass", sr, 70, 0.707))
    v = deess(v, sr, actions)
    _, vspec = band_spectrum(v, sr)
    vn = normalize_curve(vspec)
    if region_db(vn, 2500, 5000) < -1.0:
        v = apply_biquad(v, *biquad("peak", sr, 3200, 0.9, 1.5))
        actions.append("vocals: +1.5 dB presence at 3.2 kHz (vocal sat behind the beat)")
    stems["vocals"] = v

    dr = apply_biquad(stems["drums"], *biquad("highpass", sr, 30, 0.707))
    if m["crest_db"] < 10.5:
        fast = onepole(dr.abs().mean(dim=0, keepdim=True), sr, 1.5)
        slow = onepole(dr.abs().mean(dim=0, keepdim=True), sr, 60)
        att = ((fast - slow) / (slow + 1e-6)).clamp(0, 2)
        dr = dr * (1 + 0.18 * att)
        actions.append("drums: transient attack restored (+ up to 3 dB on hits)")
    stems["drums"] = dr

    b = stems["bass"]
    mono = b.mean(dim=0, keepdim=True).repeat(2, 1)
    hi = b - apply_biquad(b, *biquad("lowpass", sr, 120, 0.707))
    b = apply_biquad(mono, *biquad("lowpass", sr, 120, 0.707)) + hi
    b = apply_biquad(b, *biquad("highpass", sr, 24, 0.707))
    drive = 1.15
    b = torch.tanh(b * drive) / math.tanh(drive)
    actions.append("bass/808: mono’d below 120 Hz + light saturation for definition")
    stems["bass"] = b

    if reg(200, 500) > 2.2:
        cut = min(2.5, reg(200, 500) - 1.0)
        stems["other"] = apply_biquad(stems["other"], *biquad("peak", sr, 320, 1.0, -cut))
        actions.append(f"music: -{cut:.1f} dB at 320 Hz to clear mud around the vocal")
    return stems

# ---------------------------------------------------------------- bus stages
def stage_match_eq(x, sr, target, actions, passes=2):
    """Linear-phase matching EQ toward the target curve. Iterates so a source
    far from target converges precisely instead of stopping at the ±4 dB/pass
    cap — this is what makes reference-matching actually land on the curve."""
    applied = False
    for p in range(passes):
        _, spec = band_spectrum(x, sr)
        diff = np.clip(target - normalize_curve(spec), -4, 4)
        diff[BAND_HZ < 30] = 0
        diff[BAND_HZ > 17000] = 0
        diff = np.convolve(diff, np.ones(3) / 3, mode="same")
        if np.abs(diff).max() < (0.75 if p == 0 else 0.4):
            break
        x = fft_convolve(x, fir_from_curve(BAND_HZ, diff, sr))
        applied = True
    if applied:
        actions.append("bus: matching EQ toward target curve (linear phase, iterated to convergence)")
    return x, applied

def stage_low_control(x, sr, actions):
    low = apply_biquad(apply_biquad(x, *biquad("lowpass", sr, 150, 0.707)),
                       *biquad("lowpass", sr, 150, 0.707))
    env = onepole(low.pow(2).mean(dim=0, keepdim=True), sr, 40).sqrt()
    avg = float(env.mean())
    over = (env / (avg * 1.6 + 1e-9)).clamp(min=1.0)
    gr = over.pow(-0.5).clamp(min=10 ** (-3 / 20))   # ≤3 dB
    if float((gr < 0.95).float().mean()) < 0.02:
        return x
    actions.append("bus: dynamic low-band control (≤3 dB when the low end blooms)")
    return x - low * (1 - onepole(gr, sr, 80))

def stage_glue(x, sr, actions):
    env = onepole(x.pow(2).mean(dim=0, keepdim=True), sr, 50).sqrt()
    thr = float(env.mean()) * 10 ** (3 / 20)
    over_db = 20 * torch.log10((env / thr).clamp(min=1.0))
    gr_db = over_db * (1 - 1 / 1.4)
    gr = 10 ** (-gr_db.clamp(max=3.0) / 20)
    actions.append("bus: 1.4:1 glue compression (≤3 dB)")
    return x * onepole(gr, sr, 150)

def stage_saturate(x, sr, actions):
    drive = 1.12
    actions.append("bus: subtle tape-style saturation for warmth/density")
    return torch.tanh(x * drive) / math.tanh(drive)

def _softclip(x, c, thr=0.88):
    """Soft-clip everything above thr·c toward the ceiling (tanh knee) — flattens
    peaks so the limiter needn't dip as deep, letting the average loudness rise.
    A lower thr clips more of the waveform (harder maximization)."""
    t = thr * c
    xa = x.abs()
    return torch.where(xa > t, torch.sign(x) * (t + (c - t) * torch.tanh((xa - t) / (c - t))), x)

def _limit_peaks(x, sr, c):
    """Lookahead true-peak limiter: g[i] = min(need[i..i+look]) guarantees
    peak·g ≤ c at every sample; the 2.5 ms min-window spreads the gain change
    so it isn't a per-sample brickwall. No valley-filling smoothing."""
    look = max(4, int(0.0025 * sr))
    peak = x.abs().amax(dim=0)
    need = torch.clamp(c / peak.clamp(min=1e-9), max=1.0)
    padn = torch.nn.functional.pad(need[None, None], (0, look), value=1.0)
    g = -torch.nn.functional.max_pool1d(-padn, kernel_size=look + 1, stride=1)[0, 0]
    return x * g, -20 * math.log10(float(g.min()) + 1e-9)

def _fft_up4(x):
    """Ideal band-limited 4× upsample via FFT zero-padding (fast, one FFT pair)."""
    n = x.shape[1]
    X = torch.fft.rfft(x, n)
    Xp = torch.zeros(x.shape[0], (4 * n) // 2 + 1, dtype=X.dtype)
    Xp[:, :X.shape[1]] = X
    return torch.fft.irfft(Xp, 4 * n) * 4.0

def _fft_down4(y, n):
    """Inverse of _fft_up4: 4× downsample back to n samples."""
    Y = torch.fft.rfft(y, y.shape[1])
    return torch.fft.irfft(Y[:, :n // 2 + 1], n) / 4.0

def _tp_limit(x, sr, ceiling_db):
    """True-peak limiter: FFT-oversample 4×, limit with a tiny lookahead (the
    inter-sample overshoots sit within one base sample, so a ~16-sample window
    at 4× catches them and keeps max_pool cheap), then downsample back."""
    c = 10 ** ((ceiling_db - 1.0) / 20)
    n = x.shape[1]
    up = _fft_up4(x)
    look = 16
    peak = up.abs().amax(dim=0)
    need = torch.clamp(c / peak.clamp(min=1e-9), max=1.0)
    padn = torch.nn.functional.pad(need[None, None], (0, look), value=1.0)
    g = -torch.nn.functional.max_pool1d(-padn, kernel_size=look + 1, stride=1)[0, 0]
    return _fft_down4(up * g, n)

def stage_loudness(x, sr, target_lufs, ceiling_db, actions):
    c = 10 ** ((ceiling_db - 0.3) / 20)                 # base-rate limiter target
    gain_db = target_lufs - lufs(x, sr)
    y, gr_db = x, 0.0
    for it in range(5):                                 # drive into clip+limiter until it lands on target
        thr = max(0.45, 0.9 - 0.11 * it)                # clip harder each pass if we're still short
        y, gr_db = _limit_peaks(_softclip(x * 10 ** (min(gain_db, 24.0) / 20), c, thr), sr, c)
        err = target_lufs - lufs(y, sr)
        if abs(err) < 0.5:
            break
        gain_db += err
    if true_peak_db(y, sr) > ceiling_db:               # only oversample-limit if inter-sample peaks are over
        y = _tp_limit(y, sr, ceiling_db)
    actions.append(f"bus: normalized to {target_lufs} LUFS, true-peak limited "
                   f"(max GR {gr_db:.1f} dB, ceiling {ceiling_db} dBTP)")
    return y

# ---------------------------------------------------------------- Ozone-parity
def fir_lowpass(fc, sr, ntaps=4097):
    """Linear-phase windowed-sinc lowpass (constant group delay)."""
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.sinc(2 * fc / sr * n) * (2 * fc / sr) * np.hanning(ntaps)
    h /= h.sum()
    return torch.tensor(h, dtype=torch.float32)

def split_bands(x, sr, freqs, linphase=False):
    """Perfect-reconstruction band split by complementary subtraction — the
    bands sum back to x exactly, so unprocessed bands are transparent.
    linphase=False → doubled 2nd-order IIR crossovers (~24 dB/oct, min-phase,
    cheap). linphase=True → linear-phase windowed-sinc crossovers (phase-
    coherent, zero smearing, slower). fft_convolve is delay-compensated, so
    the linear-phase bands stay time-aligned and still telescope to x."""
    bands, rem = [], x
    for f in freqs:
        if linphase:
            lp = fft_convolve(rem, fir_lowpass(f, sr))
        else:
            lp = apply_biquad(apply_biquad(rem, *biquad("lowpass", sr, f, 0.707)),
                              *biquad("lowpass", sr, f, 0.707))
        bands.append(lp)
        rem = rem - lp
    bands.append(rem)
    return bands

def _band_thr(env, q=0.80):
    es = env.flatten()
    if es.numel() > 1_000_000:
        es = es[::es.numel() // 1_000_000]
    return float(torch.quantile(es, q))

def stage_multiband_comp(x, sr, actions, linphase=False):
    """Ozone Dynamics: 4-band compressor. Each band is tightened only where it
    peaks above its own 80th-percentile level, with gentle ratios and per-band
    timing (slow lows, fast highs). ≤4 dB GR/band."""
    bands = split_bands(x, sr, [120, 500, 3500], linphase)
    names = ["low", "low-mid", "mid", "high"]
    ratio = [2.2, 2.0, 1.8, 2.0]
    atk_ms = [30, 22, 15, 8]
    out, touched = None, []
    for i, b in enumerate(bands):
        env = onepole(b.pow(2).mean(0, keepdim=True), sr, atk_ms[i]).sqrt()
        thr = _band_thr(env)
        if thr < 1e-5:
            out = b if out is None else out + b
            continue
        over_db = 20 * torch.log10((env / thr).clamp(min=1.0))
        gr = 10 ** (-(over_db * (1 - 1 / ratio[i])).clamp(max=4.0) / 20)
        gr = onepole(gr, sr, [130, 100, 70, 45][i])
        if float((gr < 0.9).float().mean()) > 0.02:
            b = b * gr
            touched.append(names[i])
        out = b if out is None else out + b
    if touched:
        actions.append("multiband comp: " + ", ".join(touched) +
                       " band(s) tightened (gentle ratio, ≤4 dB" +
                       (", linear phase)" if linphase else ")"))
    return out

def stage_dynamic_eq(x, sr, actions):
    """Ozone Dynamic EQ: instead of static notches, attenuate each resonant
    band only in the instants it spikes — more transparent, keeps the tone
    everywhere else."""
    res = find_resonances(x, sr, max_n=3)
    n = 0
    for f, exc in res:
        bp = apply_biquad(apply_biquad(x, *biquad("highpass", sr, f / 1.30, 1.0)),
                          *biquad("lowpass", sr, f * 1.30, 1.0))
        env = onepole(bp.pow(2).mean(0, keepdim=True), sr, 8).sqrt()
        thr = _band_thr(env, 0.75)
        if thr < 1e-5:
            continue
        red = (1 - (env / thr).clamp(min=1.0).pow(-0.7)).clamp(0, 0.5)  # ≤6 dB
        red = onepole(red, sr, 25)
        if float((red > 0.06).float().mean()) < 0.02:
            continue
        x = x - bp * red
        n += 1
        actions.append(f"dynamic EQ: {f:.0f} Hz tamed only when it spikes (≤6 dB, dynamic)")
    return x

def stage_stabilizer(x, sr, actions, amount=0.55, n_fft=2048, hop=512, keep=32):
    """Ozone Stabilizer (Tame): broadband adaptive resonance smoothing. For
    every STFT frame it builds a smooth spectral envelope (low-quefrency
    cepstral lift) and pulls down any bin poking above it, bounded to
    ≤ ~4·amount dB. This flattens harsh peaks and ringing across the whole
    spectrum, frame by frame, without touching parts that are already smooth."""
    win = torch.hann_window(n_fft)
    floor = 10 ** (-(4.0 * amount) / 20)   # deepest attenuation per bin
    outs, touched = [], 0.0
    for ch in range(x.shape[0]):
        st = torch.stft(x[ch], n_fft=n_fft, hop_length=hop, window=win,
                        return_complex=True, center=True)
        mag = st.abs()
        cep = torch.fft.rfft(torch.log(mag + 1e-9), dim=0)   # cepstrum along freq
        cep[keep:, :] = 0                                     # keep smooth envelope
        env = torch.exp(torch.fft.irfft(cep, n=mag.shape[0], dim=0))
        g = torch.where(mag > env, (env / (mag + 1e-9)).pow(amount).clamp(min=floor),
                        torch.ones_like(mag))
        touched += float((g < 0.95).float().mean())
        outs.append(torch.istft(st * g, n_fft=n_fft, hop_length=hop, window=win,
                                center=True, length=x.shape[1]))
    if touched / x.shape[0] > 0.01:
        actions.append(f"stabilizer: adaptive spectral smoothing — resonances & harshness "
                       f"tamed across the spectrum (≤{4.0 * amount:.0f} dB/peak)")
    return torch.stack(outs)

def stage_multiband_image(x, sr, m, actions, target_corr=0.88, linphase=False):
    """Ozone Imager: mono the lows for club/mono safety, then scale the overall
    side energy toward the genre's real correlation — so a too-wide source gets
    reined in and a too-narrow one gets opened, but neither past the norm."""
    low, rest = split_bands(x, sr, [120], linphase)
    lowm = (low[0] + low[1]) * 0.5
    x2 = torch.stack([lowm, lowm]) + rest              # lows collapsed to mono
    mid = (x2[0] + x2[1]) * 0.5
    side = (x2[0] - x2[1]) * 0.5
    scale = 1.0
    for _ in range(4):                                   # converge side scale to hit target correlation
        c = correlation(torch.stack([mid + side * scale, mid - side * scale]))
        if abs(c - target_corr) < 0.02:
            break
        scale *= 1.18 if c > target_corr else 0.85       # wider if too correlated, narrower if too wide
        scale = max(0.2, min(1.6, scale))
    out = torch.stack([mid + side * scale, mid - side * scale])
    verb = "widened" if scale > 1.03 else "narrowed" if scale < 0.97 else "held"
    actions.append(f"multiband imager: lows mono · image {verb} to genre width "
                   f"(corr → {correlation(out):.2f})" + (" · linear phase" if linphase else ""))
    return out

def stage_exciter(x, sr, norm, target, actions):
    """Ozone Exciter / Low End Focus: generate harmonics rather than just EQ —
    tape-style warmth in the low band and/or airy odd harmonics up top, only
    where the tonal analysis says the band is lacking."""
    changed = []
    if region_db(norm - target, 8000, 16000) < -1.0:
        high = x - apply_biquad(x, *biquad("lowpass", sr, 6000, 0.707))
        x = x + torch.tanh(high * 3.0) * 0.14
        changed.append("air above 6 kHz")
    if region_db(norm - target, 35, 90) < -1.0:
        low = apply_biquad(x, *biquad("lowpass", sr, 90, 0.707))
        x = x + (torch.tanh(low * 2.0) / math.tanh(2.0) - low) * 0.35
        changed.append("low-end weight")
    if changed:
        actions.append("harmonic exciter: added " + " + ".join(changed) + " (generated harmonics, not just EQ)")
    return x

def stage_sub_control(x, sr, target, actions, strength=1.0):
    """When the mix sits well above the genre target below ~90 Hz, a low-shelf
    pulls the sub toward target. The matching EQ caps at ±4 dB/pass and can't
    fully tame a grossly hot 808; this does — a tighter, more translatable low
    end that stops the sub from masking the vocal. Runs AFTER matching EQ so it
    has the final say on the low, and only fires when the excess is real."""
    _, spec = band_spectrum(x, sr)
    nc = normalize_curve(spec)
    sel = (BAND_HZ >= 35) & (BAND_HZ <= 90)
    excess = float(nc[sel].mean()) - float(target[sel].mean())
    if excess < 1.5 or strength <= 0:
        return x
    cut = min(6.0, (excess - 0.5) * strength)
    x = apply_biquad(x, *biquad("lowshelf", sr, 80, 0.7, -cut))
    actions.append(f"sub control: -{cut:.1f} dB low-shelf at 80 Hz (sub was +{excess:.1f} dB over target — tightened so the vocal breathes)")
    return x

def stage_clarity(x, sr, actions, strength=1.0, eq=None):
    """Vocal + music clarity. Two modes:

    • simple (default) — shave low-mid mud (~300 Hz) + a touch of presence
      (~3.2 kHz) scaled by `strength`. Unchanged behaviour for the generic lanes.
    • tuned — when `eq` is a dict, apply an explicit 4-band clarity treatment
      (dB amounts): `mud` cut ~300 Hz, `box` cut ~520 Hz (music congestion),
      `presence` boost ~3.2 kHz (vocal cuts), `air` high-shelf ~10.5 kHz
      (openness/separation up top). This is how the GQ GENO lane is voiced.

    All boosts stay gentle and the top shelf sits above the sibilance band, so
    it reads as air, not ess."""
    if eq:
        mud = float(eq.get("mud", 0.0)); box = float(eq.get("box", 0.0))
        pres = float(eq.get("presence", 0.0)); air = float(eq.get("air", 0.0))
        parts = []
        if mud > 0.05:
            x = apply_biquad(x, *biquad("peak", sr, 300, 1.1, -min(4.0, mud))); parts.append(f"-{min(4.0,mud):.1f}@300")
        if box > 0.05:
            x = apply_biquad(x, *biquad("peak", sr, 520, 1.2, -min(3.0, box))); parts.append(f"-{min(3.0,box):.1f}@520")
        if pres > 0.05:
            x = apply_biquad(x, *biquad("peak", sr, 3200, 0.9, min(3.5, pres))); parts.append(f"+{min(3.5,pres):.1f}@3.2k")
        if air > 0.05:
            x = apply_biquad(x, *biquad("highshelf", sr, 10500, 0.7, min(3.5, air))); parts.append(f"+{min(3.5,air):.1f}@air")
        if parts:
            actions.append("clarity: " + ", ".join(parts) + " (vocal + music separation)")
        return x
    if strength <= 0:
        return x
    mud = min(2.5, 2.2 * strength)
    pres = min(2.0, 1.6 * strength)
    x = apply_biquad(x, *biquad("peak", sr, 300, 1.1, -mud))
    x = apply_biquad(x, *biquad("peak", sr, 3200, 0.9, pres))
    actions.append(f"clarity: -{mud:.1f} dB mud at 300 Hz + {pres:.1f} dB presence at 3.2 kHz (vocal cuts through)")
    return x

# ---------------------------------------------------------------- pipeline
def enhance(path_in, path_out, genre="hiphop", target_lufs=-9.5, ceiling_db=-1.0,
            use_stems=True, model="htdemucs", device="cpu", linphase=False,
            ref_curve=None, ref_corr=None, ref_name=None, sub_control=None,
            clarity=None, clarity_eq=None, progress=None):
    def prog(p, msg):
        if progress: progress(p, msg)

    prog(2, "loading audio")
    x, sr = load_audio(path_in)

    prog(6, "analyzing source")
    m, norm = analyze(x, sr)
    genre_conf, genre_scores = None, None
    if ref_curve is not None:
        # reference match: aim at a specific commercial track's measured curve
        target = np.asarray(ref_curve, dtype=float)
        tcorr = ref_corr if ref_corr else 0.88
        genre = "reference"
    else:
        if genre == "auto" or genre not in TARGETS:
            genre, genre_conf, genre_scores = classify_genre(m, norm)
        target = TARGETS[genre]
        tcorr = GENRE_CORR.get(genre, 0.88)
    # resolve per-lane processing from the profile only where the caller didn't override
    prof = GENRE_PROFILE.get(genre, {})
    if sub_control is None:
        sub_control = prof.get("sub_control", 0.6)
    if clarity is None:
        clarity = prof.get("clarity", 0.6)
    if clarity_eq is None:
        clarity_eq = prof.get("clarity_eq")
    issues = detect_issues(m, norm, target)
    actions = []
    if ref_curve is not None:
        actions.append(f"reference match: aiming at {ref_name or 'your reference track'}'s exact spectrum + width")
    if genre_conf is not None:
        actions.append(f"genre auto-detected: {genre} ({int(genre_conf*100)}% confidence, from spectrum + dynamics)")
    if linphase:
        actions.append("linear-phase mode: matching EQ + multiband splits are phase-coherent (zero smearing, slower render)")

    if abs(m["dc_offset"]) > 0.002:
        x = x - x.mean(dim=1, keepdim=True)
        actions.append("removed DC offset")
    x = apply_biquad(x, *biquad("highpass", sr, 20, 0.707))

    if use_stems:
        try:
            prog(12, "separating stems (demucs — the long part, hang tight)")
            stems, ssr = separate(x, sr, model, device, None)
            stems = {k: v.float() for k, v in stems.items()}
            if ssr != sr:
                sr = ssr  # demucs works at its own rate; continue there
            prog(46, "processing stems adaptively")
            stems = process_stems(stems, sr, m, norm, target, actions)
            x = sum(stems.values())
            del stems
            actions.insert(0, "ML source separation → per-stem processing → remix")
        except Exception as e:  # noqa: BLE001 — stems are an upgrade, not a requirement
            actions.append(f"stem stage skipped ({e}) — bus-only processing")
    prog(54, "dynamic EQ (resonance control)")
    x = stage_dynamic_eq(x, sr, actions)

    # Stabilizer (Ozone) — strength adapts to how harsh/resonant the source is;
    # A/B gated so it only stays if it doesn't dull the track.
    harsh = region_db(norm - target, 2500, 6000)
    amount = 0.5 + min(0.35, max(0.0, harsh - 1.0) * 0.15)
    prog(58, "stabilizer (adaptive resonance smoothing, A/B gated)")
    st_before = quality_score(excerpt(x, sr), sr, target)
    y = stage_stabilizer(x, sr, actions, amount=amount)
    if quality_score(excerpt(y, sr), sr, target) <= st_before + 0.05:
        x = y
    elif actions and actions[-1].startswith("stabilizer"):
        actions[-1] += " — reverted (dulled the track)"
    del y

    prog(62, "tonal balance (matching EQ, A/B gated)")
    ex_before = quality_score(excerpt(x, sr), sr, target)
    y, applied = stage_match_eq(x, sr, target, actions)
    if applied:
        if quality_score(excerpt(y, sr), sr, target) <= ex_before + 0.05:
            x = y
        else:
            actions[-1] += " — reverted (no measurable improvement)"
    del y

    prog(66, "sub control + clarity")
    x = stage_sub_control(x, sr, target, actions, strength=sub_control)
    x = stage_clarity(x, sr, actions, strength=clarity, eq=clarity_eq)

    prog(68, "low-end control")
    x = stage_low_control(x, sr, actions)

    # Multiband compressor (Ozone Dynamics) — A/B gated; falls back to broadband
    # glue if it doesn't help. Replaces the single-band glue as the main tool.
    if m["crest_db"] > 9.0:
        prog(74, "multiband compression (A/B gated)")
        s_before = quality_score(excerpt(x, sr), sr, target)
        y = stage_multiband_comp(x, sr, actions, linphase)
        if quality_score(excerpt(y, sr), sr, target) <= s_before + 0.05:
            x = y
        else:
            actions[-1] += " — reverted (broadband glue instead)"
            x = stage_glue(x, sr, actions)
        del y

    prog(80, "harmonic exciter")
    x = stage_exciter(x, sr, norm, target, actions)
    warm = region_db(norm - target, 200, 500)
    if warm < 0.5 and m["crest_db"] > 9:
        x = stage_saturate(x, sr, actions)

    prog(86, "multiband stereo image")
    x = stage_multiband_image(x, sr, m, actions, tcorr, linphase)

    prog(90, "loudness + true-peak limiting")
    x = stage_loudness(x, sr, target_lufs, ceiling_db, actions)

    prog(96, "verifying + encoding")
    after = {
        "lufs": round(lufs(x, sr), 1),
        "true_peak_db": round(true_peak_db(x, sr), 2),
        "crest_db": round(crest_db(x), 1),
        "correlation": round(correlation(x), 3),
    }
    scores = score_result(x, sr, target, m, target_lufs)
    actions.append(f"quality score: {scores['overall']}/100 "
                   f"(balance {scores['balance']} · clarity {scores['clarity']} · "
                   f"punch {scores['punch']} · dynamics {scores['dynamics']} · "
                   f"loudness {scores['loudness']} · translation {scores['translation']})")
    meter = balance_meter(target, m["spectrum_db"], out_spectrum(x, sr))
    save_wav24(x, sr, path_out)
    prog(100, "done")
    return {"genre": genre, "genre_confidence": genre_conf, "genre_scores": genre_scores,
            "analysis": m, "issues": issues, "actions": actions,
            "after": after, "scores": scores, "balance_meter": meter}

# ---------------------------------------------------------------- chat cleanup
def _bus_deess(x, sr, amt):
    """Bus de-esser: pull down 5.5–9.5 kHz only when it spikes, by `amt`·70%."""
    band = apply_biquad(apply_biquad(x, *biquad("highpass", sr, 5500, 0.707)),
                        *biquad("lowpass", sr, 9500, 0.707))
    env = onepole(band.pow(2).mean(0, keepdim=True), sr, 4).sqrt()
    thr = _band_thr(env, 0.75)
    if thr < 1e-6:
        return x
    red = (1 - (env / thr).clamp(min=1.0).pow(-0.7)).clamp(0, min(0.7, 0.7 * amt))
    return x - band * onepole(red, sr, 15)

def convolve_causal(x, ir):
    """Full causal convolution, truncated to input length (ir[0] aligns x[0])."""
    n = x.shape[1] + ir.shape[-1] - 1
    nfft = 1 << ((n - 1).bit_length())
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(ir, nfft), nfft)
    return y[:, :x.shape[1]]

def _reverb_ir(sr, decay_s, predelay_ms, seed):
    """Synthetic room impulse: exponentially-decaying noise tail + a few early
    reflections + predelay. One per channel (different seeds) for stereo width."""
    g = torch.Generator().manual_seed(seed)
    n = int(decay_s * sr)
    t = torch.arange(n) / sr
    tau = decay_s / 6.908                       # RT60 → exp time constant
    ir = torch.randn(n, generator=g) * torch.exp(-t / tau)
    for d_ms, gn in [(7, 0.5), (13, 0.42), (19, 0.34), (29, 0.28), (41, 0.22)]:
        i = int(d_ms / 1000 * sr)
        if i < n:
            ir[i] += gn
    pd = int(predelay_ms / 1000 * sr)
    if pd > 0:
        ir = torch.cat([torch.zeros(pd), ir])[:n]
    return ir

def stage_reverb(x, sr, amt, size=1.0):
    """Convolution reverb — adds room/space as a wet send on top of the dry."""
    decay = 0.5 + size * 1.3
    ir = torch.stack([_reverb_ir(sr, decay, 18, 1), _reverb_ir(sr, decay, 23, 2)])
    ir = apply_biquad(ir, *biquad("lowpass", sr, 7000, 0.707))   # darker, natural tail
    ir = apply_biquad(ir, *biquad("highpass", sr, 200, 0.707))   # keep lows out of the verb
    ir = ir / (ir.abs().max() + 1e-9)
    wet = convolve_causal(x, ir)
    wet = wet / (wet.abs().max() + 1e-9) * (float(x.abs().max()) + 1e-9)
    return x + wet * (0.4 * amt)                # dry stays intact, ambience added

def stage_dereverb(x, sr, amt, n_fft=2048, hop=512):
    """Spectral de-reverb: per frequency bin, track a decaying peak reference;
    when the bin's energy falls below it (the reverb tail between hits/words),
    duck it. Direct/transient sound sits at the peak and passes untouched.
    Bounded to ≤ ~12·amt dB so it reduces the wash without gating artifacts."""
    win = torch.hann_window(n_fft)
    floor = 10 ** (-(12.0 * amt) / 20)
    rel = math.exp(-hop / (0.28 * sr))          # 280 ms tail memory
    thr = 0.5
    outs = []
    for ch in range(x.shape[0]):
        st = torch.stft(x[ch], n_fft=n_fft, hop_length=hop, window=win,
                        return_complex=True, center=True)
        mag = st.abs()
        ref = torch.empty_like(mag)
        r = mag[:, 0].clone(); ref[:, 0] = r
        for f in range(1, mag.shape[1]):         # recursive decaying peak follower
            r = torch.maximum(mag[:, f], r * rel)
            ref[:, f] = r
        ratio = mag / (ref + 1e-9)               # ~1 at peaks, <1 in tails
        g = torch.where(ratio < thr,
                        (ratio / thr).clamp(min=floor).pow(amt * 1.5),
                        torch.ones_like(mag))
        outs.append(torch.istft(st * g, n_fft=n_fft, hop_length=hop, window=win,
                                center=True, length=x.shape[1]))
    return torch.stack(outs)

def stage_tape(x, sr, amt):
    """Analog tape emulation: low head-bump, asymmetric soft saturation
    (even + odd harmonics = warmth), gentle high-frequency rolloff."""
    y = apply_biquad(x, *biquad("lowshelf", sr, 60, 0.8, 1.2 * amt))   # head bump
    drive = 1.0 + 1.6 * amt
    bias = 0.14 * amt                                                  # asymmetry → even harmonics
    y = (torch.tanh(drive * y + bias) - math.tanh(bias)) / math.tanh(drive)
    y = y - y.mean(dim=1, keepdim=True)                               # kill residual DC
    y = apply_biquad(y, *biquad("highshelf", sr, 12000, 0.707, -1.1 * amt))  # HF rolloff
    return y

def stage_vintagecomp(x, sr, amt):
    """Opto / vari-mu style bus compressor: slow, program-dependent, soft, with
    a touch of tube harmonic color. Smooth 'glue' rather than fast control."""
    env = onepole(x.pow(2).mean(0, keepdim=True), sr, 30).sqrt()
    thr = _band_thr(env, 0.6)
    if thr < 1e-6:
        return x
    ratio = 2.5
    over_db = 20 * torch.log10((env / thr).clamp(min=1.0))
    gr = 10 ** (-(over_db * (1 - 1 / ratio)).clamp(max=6.0 * amt) / 20)
    gr = onepole(gr, sr, 130)                                         # slow opto release
    y = x * gr
    d = 1.0 + 0.3 * amt                                              # subtle tube color
    y = torch.tanh(d * y) / math.tanh(d)
    return y * 10 ** ((3.0 * amt) / 20)                              # gentle makeup

def stage_vintageeq(x, sr, amt):
    """Pultec-style program EQ: the classic low boost + slightly-higher cut
    (tight, resonant lows) plus a broad, silky top-end air shelf."""
    y = apply_biquad(x, *biquad("lowshelf", sr, 60, 0.9, 2.0 * amt))   # low boost
    y = apply_biquad(y, *biquad("peak", sr, 200, 1.0, -1.2 * amt))     # low-mid cut (the trick)
    y = apply_biquad(y, *biquad("highshelf", sr, 12000, 0.6, 1.8 * amt))  # air
    return y

def adjust_master(path_in, path_out, ops, genre="hiphop", target_lufs=-9.5, ceiling_db=-1.0):
    """Apply a list of natural-language-derived cleanup ops to an existing
    master, then re-normalize + true-peak limit. Ops are always applied to the
    original AI master (the app sends the full cumulative set each time), so
    there's no generation loss and 'reset' just clears the list."""
    x, sr = load_audio(path_in)
    before_spec = out_spectrum(x, sr)
    src_crest = crest_db(x)
    for op in ops:
        t = op.get("type")
        g = float(op.get("gain", 0))
        if t == "lowshelf":
            x = apply_biquad(x, *biquad("lowshelf", sr, float(op.get("f", 80)), 0.707, g))
        elif t == "highshelf":
            x = apply_biquad(x, *biquad("highshelf", sr, float(op.get("f", 10000)), 0.707, g))
        elif t == "bell":
            x = apply_biquad(x, *biquad("peak", sr, float(op.get("f", 1000)), float(op.get("q", 1.0)), g))
        elif t == "deess":
            x = _bus_deess(x, sr, float(op.get("amt", 0.5)))
        elif t == "width":
            mult = float(op.get("mult", 1.0))
            mid = (x[0] + x[1]) * 0.5
            side = apply_biquad(((x[0] - x[1]) * 0.5 * mult)[None], *biquad("highpass", sr, 120, 0.707))[0]
            x = torch.stack([mid + side, mid - side])
        elif t == "saturate":
            d = float(op.get("drive", 1.1))
            x = torch.tanh(x * d) / math.tanh(d)
        elif t == "dereverb":
            x = stage_dereverb(x, sr, float(op.get("amt", 0.5)))
        elif t == "reverb":
            x = stage_reverb(x, sr, float(op.get("amt", 0.4)), float(op.get("size", 1.0)))
        elif t == "tape":
            x = stage_tape(x, sr, float(op.get("amt", 0.5)))
        elif t == "vintagecomp":
            x = stage_vintagecomp(x, sr, float(op.get("amt", 0.5)))
        elif t == "vintageeq":
            x = stage_vintageeq(x, sr, float(op.get("amt", 0.5)))
    x = stage_loudness(x, sr, target_lufs, ceiling_db, [])
    tgt = TARGETS.get(genre, TARGETS["hiphop"])
    save_wav24(x, sr, path_out)
    return {"after": {"lufs": round(lufs(x, sr), 1),
                      "true_peak_db": round(true_peak_db(x, sr), 2),
                      "crest_db": round(crest_db(x), 1),
                      "correlation": round(correlation(x), 3)},
            "scores": score_result(x, sr, tgt, {"crest_db": src_crest}, target_lufs),
            "balance_meter": balance_meter(tgt, before_spec, out_spectrum(x, sr))}

# ---------------------------------------------------------------- AI mix
# Per-genre stem loudness offsets relative to the lead vocal (LUFS). These are
# where a mix engineer sits each element: vocal on top, drums just under, bass
# controlled, music bed tucked back so the vocal owns the center.
MIX_BALANCE = {
    "hiphop": {"vocals": 0.0, "drums": -1.5, "bass": -2.5, "other": -5.5},
    "rnb":    {"vocals": 0.0, "drums": -3.0, "bass": -3.0, "other": -4.5},
    "pop":    {"vocals": 0.0, "drums": -2.5, "bass": -3.5, "other": -4.0},
    "geno":   {"vocals": 0.0, "drums": -1.5, "bass": -2.5, "other": -5.5},  # GQ GENO — vocal-forward melodic trap
    "geno_punch": {"vocals": 0.0, "drums": -1.5, "bass": -2.5, "other": -5.5},  # GQ GENO — punchy
}

def widen(x, amt):
    mid = (x[0] + x[1]) * 0.5
    side = (x[0] - x[1]) * 0.5 * amt
    return torch.stack([mid + side, mid - side])

def vocal_presence(mix, sr):
    """Energy in the 1–4 kHz vocal-presence band (dB, mid-anchored) — clarity proxy."""
    _, spec = band_spectrum(mix, sr)
    return region_db(normalize_curve(spec), 1000, 4000)

def mask_carve(music, vocal, sr, actions):
    """Cut the music stem in the ≤3 bands where the vocal is strongest, so the
    vocal cuts through without turning it up (frequency-masking reduction)."""
    _, vspec = band_spectrum(vocal, sr)
    vn = normalize_curve(vspec)
    cand = sorted(((float(vn[i]), float(BAND_HZ[i])) for i in range(len(BAND_HZ))
                   if 350 <= BAND_HZ[i] <= 5000 and vn[i] > 1.0), reverse=True)
    used, n = [], 0
    for exc, f in cand:
        if all(abs(math.log2(f / u)) > 0.5 for u in used):
            cut = min(3.0, 0.55 * exc)
            music = apply_biquad(music, *biquad("peak", sr, f, 2.5, -cut))
            used.append(f); n += 1
        if n >= 3:
            break
    if n:
        actions.append(f"music: carved {n} pocket(s) so the vocal cuts through (mask reduction, ≤3 dB)")
    return music, n

def aimix(path_in, stem_dir, mix_path, genre="hiphop", mix_target_lufs=-16.0,
          model="htdemucs", device="cpu", progress=None):
    """AI mix: separate → adaptively process each stem → unmask the vocal →
    set the balance from genre targets → sum to a headroom-safe mix (no
    limiting; mastering happens after). Saves the 4 processed stems (levels
    baked in) plus the summed mix, and reports every move with its reason."""
    def prog(p, msg):
        if progress: progress(p, msg)

    prog(2, "loading audio")
    x, sr = load_audio(path_in)

    prog(6, "analyzing source")
    m, norm = analyze(x, sr)
    genre_conf, genre_scores = None, None
    if genre == "auto" or genre not in TARGETS:
        genre, genre_conf, genre_scores = classify_genre(m, norm)
    target = TARGETS[genre]
    issues = detect_issues(m, norm, target)
    actions = ["ML source separation → per-stem processing → intelligent balance"]
    if genre_conf is not None:
        actions.append(f"genre auto-detected: {genre} ({int(genre_conf*100)}% confidence, from spectrum + dynamics)")
    if abs(m["dc_offset"]) > 0.002:
        x = x - x.mean(dim=1, keepdim=True)
        actions.append("removed DC offset")
    x = apply_biquad(x, *biquad("highpass", sr, 20, 0.707))

    prog(12, "separating stems (demucs — the long part, hang tight)")
    stems, ssr = separate(x, sr, model, device, None)
    stems = {k: v.float() for k, v in stems.items()}
    sr = ssr

    prog(50, "processing stems adaptively")
    stems = process_stems(stems, sr, m, norm, target, actions)

    prog(64, "unmasking the vocal (A/B gated)")
    before = vocal_presence(stems["vocals"] + stems["other"], sr)
    carved, ncuts = mask_carve(stems["other"].clone(), stems["vocals"], sr, actions)
    if ncuts:
        if vocal_presence(stems["vocals"] + carved, sr) >= before - 0.05:
            stems["other"] = carved
        else:
            actions[-1] += " — reverted (didn't help clarity)"

    prog(72, "widening the music bed")
    stems["other"] = widen(stems["other"], 1.18)
    actions.append("music: +18% stereo width to open space around the center vocal")

    prog(78, "setting the balance")
    bal = MIX_BALANCE.get(genre, MIX_BALANCE["hiphop"])
    lu = {k: lufs(stems[k], sr) for k in stems}
    vref = lu["vocals"] if lu["vocals"] > -55 else max(lu.values())
    gains = {k: max(-12.0, min(12.0, (vref + bal[k]) - lu[k]))
             for k in ("vocals", "drums", "bass", "other")}

    # gain-stage the summed balance to the mix target, then guarantee headroom
    mix_raw = sum(stems[k] * 10 ** (gains[k] / 20) for k in stems)
    glob = mix_target_lufs - lufs(mix_raw, sr)
    peak = float((mix_raw * 10 ** (glob / 20)).abs().max())
    if peak > 0:
        glob += min(0.0, -20 * math.log10(peak) - 1.0)   # keep ≥1 dB headroom
    rec = {k: round(gains[k] + glob, 1) for k in gains}

    # bake the balance into the stems (faders sit at 0), sum the mix
    for k in stems:
        stems[k] = stems[k] * 10 ** (rec[k] / 20)
    mix = sum(stems.values())
    peak = max([float(mix.abs().max())] + [float(stems[k].abs().max()) for k in stems])
    if peak > 0.999:                                       # global clip safety
        s = 0.999 / peak
        for k in stems:
            stems[k] = stems[k] * s
        mix = mix * s
    actions.append("balance: " + " · ".join(f"{k} {rec[k]:+.1f} dB" for k in
                   ("vocals", "drums", "bass", "other")))
    actions.append(f"mix bus: gain-staged to ~{mix_target_lufs} LUFS with ≥1 dB "
                   "headroom — no limiting (mix stage, mastering comes after)")

    prog(90, "encoding stems + mix")
    os.makedirs(stem_dir, exist_ok=True)
    for k in ("vocals", "drums", "bass", "other"):
        save_wav24(stems[k], sr, os.path.join(stem_dir, k + ".wav"))
    save_wav24(mix, sr, mix_path)

    prog(96, "verifying")
    after = {
        "lufs": round(lufs(mix, sr), 1),
        "peak_db": round(20 * math.log10(float(mix.abs().max()) + 1e-12), 2),
        "crest_db": round(crest_db(mix), 1),
        "correlation": round(correlation(mix), 3),
    }
    scores = score_result(mix, sr, target, m, mix_target_lufs)
    actions.append(f"quality score: {scores['mix']}/100 mix "
                   f"(balance {scores['balance']} · clarity {scores['clarity']} · "
                   f"punch {scores['punch']} · dynamics {scores['dynamics']} · stereo {scores['stereo']})")
    meter = balance_meter(target, m["spectrum_db"], out_spectrum(mix, sr))
    prog(100, "done")
    return {"genre": genre, "genre_confidence": genre_conf, "genre_scores": genre_scores,
            "analysis": m, "issues": issues, "actions": actions,
            "balance": rec, "after": after, "scores": scores, "balance_meter": meter}
