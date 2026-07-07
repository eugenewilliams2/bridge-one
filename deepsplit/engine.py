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

TARGETS = {
    "hiphop": _target([(25, 4), (45, 6.5), (80, 6.5), (120, 3.5), (200, 0.5), (400, -0.5),
                       (800, -0.5), (1500, 0), (3000, 0), (5000, 0.5), (8000, 1), (12000, 2), (18000, 0.5)]),
    "rnb":    _target([(25, 2.5), (45, 5), (80, 5), (120, 3), (200, 0.5), (400, -0.5),
                       (800, 0), (1500, 0), (3000, 0.5), (5000, 1), (8000, 1.5), (12000, 2.5), (18000, 1)]),
    "pop":    _target([(25, 1), (45, 3.5), (80, 4), (120, 2.5), (200, 0.5), (400, -0.5),
                       (800, 0), (1500, 0.5), (3000, 1), (5000, 1.5), (8000, 1.5), (12000, 2.5), (18000, 1)]),
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
    pres = region_db(nc, 2000, 5000)
    mud = region_db(nc, 200, 500)
    harsh = max(0.0, region_db(nc, 2500, 5000) - 3.0)
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
def stage_match_eq(x, sr, target, actions):
    _, spec = band_spectrum(x, sr)
    diff = np.clip(target - normalize_curve(spec), -4, 4)
    diff[BAND_HZ < 30] = 0
    diff[BAND_HZ > 17000] = 0
    diff = np.convolve(diff, np.ones(3) / 3, mode="same")
    if np.abs(diff).max() < 0.75:
        return x, False
    h = fir_from_curve(BAND_HZ, diff, sr)
    actions.append(f"bus: matching EQ toward genre curve (max {np.abs(diff).max():.1f} dB, linear phase)")
    return fft_convolve(x, h), True

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

def stage_loudness(x, sr, target_lufs, ceiling_db, actions):
    cur = lufs(x, sr)
    x = x * 10 ** ((target_lufs - cur) / 20)
    c = 10 ** ((ceiling_db - 0.3) / 20)
    look = max(8, int(0.0015 * sr))
    a = x.abs().max(dim=0).values
    target = c / sliding_max(a, look).clamp(min=c)
    # lookahead = sliding minimum; pad with unity gain so edges stay sane
    padded = torch.nn.functional.pad(-target[None, None], (look, look), value=-1.0)
    g = -torch.nn.functional.max_pool1d(padded, kernel_size=2 * look + 1, stride=1)[0, 0]
    gr_db = -20 * math.log10(float(g.min()) + 1e-9)
    g = onepole(g[None], sr, 40)[0].clamp(max=1.0)
    y = (x * g).clamp(-c, c)
    tp = true_peak_db(y, sr)
    if tp > ceiling_db:
        y = y * 10 ** ((ceiling_db - tp) / 20)
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

def stage_multiband_image(x, sr, m, actions, linphase=False):
    """Ozone Imager: width per band — lows mono, mids natural, highs opened.
    Narrower sources get pushed wider; already-wide ones are left alone."""
    corr = m["correlation"]
    hi_w = 1.35 if corr > 0.9 else 1.15 if corr > 0.6 else 1.0
    bands = split_bands(x, sr, [120, 2500], linphase)
    widths = [0.0, 1.12, hi_w]          # low fully mono, mid gentle, high wide
    out = None
    for b, wd in zip(bands, widths):
        mid = (b[0] + b[1]) * 0.5
        side = (b[0] - b[1]) * 0.5 * wd
        band = torch.stack([mid + side, mid - side])
        out = band if out is None else out + band
    actions.append(f"multiband imager: lows mono · mids natural · highs ×{hi_w:.2f} (mono-safe"
                   + (", linear phase)" if linphase else ")"))
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

# ---------------------------------------------------------------- pipeline
def enhance(path_in, path_out, genre="hiphop", target_lufs=-9.5, ceiling_db=-1.0,
            use_stems=True, model="htdemucs", device="cpu", linphase=False, progress=None):
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
    actions = []
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
    x = stage_multiband_image(x, sr, m, actions, linphase)

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
