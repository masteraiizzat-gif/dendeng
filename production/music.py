"""Original soundtrack + sound design for the DENDENG PADU commercial (fully synthesised,
no third-party music -> no licensing issues). 120 BPM, D minor, 30.0 s, 48 kHz stereo.

usage: python3 music.py <src_dir> <work_dir>   -> work_dir/soundtrack.wav
Original footage audio (sizzle / stirring) is mixed in as foley.
"""
import sys, subprocess
import numpy as np
from scipy import signal

SRC, WORK = sys.argv[1], sys.argv[2]
SR, DUR = 48000, 30.0
N = int(SR * DUR)
rng = np.random.default_rng(11)
L = np.zeros(N); R = np.zeros(N)
BEAT = 0.5

def tt(d): return np.arange(int(d * SR)) / SR
def sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")
def filt(x, kind, f, order=2): return signal.sosfilt(sos(kind, f, order), x)
def add(x, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= N: return
    if i < 0: x = x[-i:]; i = 0
    x = x[:N - i]
    gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(x)] += x * gain * gl * 1.414; R[i:i + len(x)] += x * gain * gr * 1.414
def add_st(l, r, t0, gain=1.0):
    i = int(t0 * SR); n = min(len(l), N - i)
    L[i:i + n] += l[:n] * gain; R[i:i + n] += r[:n] * gain
def saw(f, t, ph=0.0):
    return 2 * ((f * t + ph) % 1.0) - 1
def noise(d): return rng.standard_normal(int(d * SR))

# ------------------------------------------------------------------ instruments
def kick():
    t = tt(0.45)
    f = 46 + 120 * np.exp(-t * 32)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 6.5)
    x += 0.35 * filt(noise(0.45), "high", 2500) * np.exp(-t * 180)
    return np.tanh(x * 1.6)

def snare():
    t = tt(0.3)
    n = filt(noise(0.3), "band", [900, 7000]) * np.exp(-t * 16)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 28)
    return 0.8 * n + 0.6 * tone

def clap():
    t = tt(0.35); x = np.zeros(len(t))
    for k, d in enumerate((0, 0.011, 0.022)):
        i = int(d * SR); seg = filt(noise(0.35), "band", [1000, 4500])[: len(t) - i]
        x[i:] += seg * np.exp(-tt(0.35)[: len(t) - i] * (60 if k < 2 else 13))
    return x

def hat(open_=False):
    d = 0.28 if open_ else 0.05; t = tt(d)
    return filt(noise(d), "high", 7500, 4) * np.exp(-t * (11 if open_ else 75))

def impact(big=1.0):
    """deep cinematic hit: sub drop + braam chord + noise crack"""
    d = 3.2; t = tt(d)
    f = 58 * np.exp(-t * 0.5) + 22
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6)
    crack = filt(noise(d), "low", 3500) * np.exp(-t * 9)
    br = np.zeros(len(t))
    for fr in (36.71, 55.0, 73.42, 87.31, 110.0):
        for det in (-0.25, 0.0, 0.3):
            br += saw(fr * (1 + det / 100 * 2), t, rng.uniform())
    cutoff_env = 250 + 2600 * np.exp(-t * 3)
    # time-varying low-pass via block processing
    out = np.zeros(len(t)); blk = 1024; zi = None
    for i in range(0, len(t), blk):
        s = signal.butter(2, cutoff_env[i], "low", fs=SR, output="sos")
        if zi is None: zi = signal.sosfilt_zi(s) * 0
        out[i:i + blk], zi = signal.sosfilt(s, br[i:i + blk], zi=zi)
    br = out / 9 * np.exp(-t * 1.25)
    x = 1.3 * sub + 0.55 * crack + 0.9 * br
    return np.tanh(x * 1.4) * big

def bass_hit():
    d = 1.6; t = tt(d)
    f = 41 + 40 * np.exp(-t * 18)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.4)
    x += 0.4 * np.tanh(3 * saw(41.2, t)) * np.exp(-t * 5)
    x += 0.4 * filt(noise(d), "low", 1800) * np.exp(-t * 14)
    return np.tanh(x * 1.8)

def taiko():
    d = 0.9; t = tt(d)
    f = 62 + 60 * np.exp(-t * 20)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5)
    x += 0.5 * filt(noise(d), "band", [150, 900]) * np.exp(-t * 18)
    return np.tanh(x * 1.5)

def riser(d):
    t = tt(d); k = t / d
    n = noise(d); out = np.zeros(len(t)); blk = 1024; zi = None
    for i in range(0, len(t), blk):
        fc = 250 * (40 ** k[i]); s = signal.butter(2, [fc * 0.7, min(fc * 1.4, 20000)], "band", fs=SR, output="sos")
        if zi is None: zi = signal.sosfilt_zi(s) * 0
        out[i:i + blk], zi = signal.sosfilt(s, n[i:i + blk], zi=zi)
    tone = np.sin(2 * np.pi * np.cumsum(180 * (6 ** k)) / SR) * 0.25
    return (out * 1.4 + tone) * k ** 2.2

def whoosh(d=0.55):
    t = tt(d); k = t / d
    env = np.exp(-((k - 0.62) / 0.2) ** 2)
    n = noise(d); out = np.zeros(len(t)); blk = 512; zi = None
    for i in range(0, len(t), blk):
        fc = 400 + 3800 * np.exp(-((k[i] - 0.62) / 0.25) ** 2)
        s = signal.butter(2, [fc * 0.6, fc * 1.6], "band", fs=SR, output="sos")
        if zi is None: zi = signal.sosfilt_zi(s) * 0
        out[i:i + blk], zi = signal.sosfilt(s, n[i:i + blk], zi=zi)
    return out * env * 2.2

def rev_cymbal(d=1.0):
    t = tt(d); return filt(noise(d), "high", 4500, 2) * (t / d) ** 3

def sparkle(d=1.2):
    t = tt(d); x = np.zeros(len(t))
    for k in range(9):
        t0 = k * d / 12; f = rng.choice([1760, 2093, 2349, 2637, 3136, 3520, 4186])
        i = int(t0 * SR); tl = t[: len(t) - i]
        x[i:] += np.sin(2 * np.pi * f * tl) * np.exp(-tl * 7) * 0.3
    return x

def pluck(f, d=0.35):
    t = tt(d)
    x = saw(f, t) * 0.6 + np.sign(np.sin(2 * np.pi * f * t)) * 0.3
    x = filt(x, "low", 2600) * np.exp(-t * 9)
    return x

def pad_note(freqs, d, cutoff=1400):
    t = tt(d); l = np.zeros(len(t)); r = np.zeros(len(t))
    for f in freqs:
        for det, side in ((-8, -1), (0, 0), (8, 1)):
            v = saw(f * 2 ** (det / 1200), t, rng.uniform())
            l += v * (0.5 - 0.35 * side); r += v * (0.5 + 0.35 * side)
    env = np.minimum(1, t / 0.5) * np.minimum(1, (d - t) / 0.4).clip(0, 1)
    l = filt(l, "low", cutoff) * env; r = filt(r, "low", cutoff) * env
    return l / (3 * len(freqs)), r / (3 * len(freqs))

def bass_note(f, d):
    t = tt(d)
    x = 0.6 * np.sin(2 * np.pi * f * t) + 0.5 * filt(saw(f, t), "low", 420)
    env = np.minimum(1, t / 0.005) * np.exp(-t * 4.5)
    return np.tanh(x * env * 1.5)

def sizzle(d, density=420):
    t = tt(d)
    hiss = filt(noise(d), "band", [2500, 11000]) * (0.35 + 0.25 * filt(np.abs(noise(d)), "low", 6))
    imp = (rng.uniform(0, 1, len(t)) < density / SR) * rng.uniform(0.3, 1, len(t)) * np.sign(rng.standard_normal(len(t)))
    crack = filt(imp, "high", 1800) * 3
    return hiss * 0.5 + crack

# ------------------------------------------------------------------ arrangement
CH = {"Dm": ([146.83, 174.61, 220.0], 73.42), "Bb": ([116.54, 146.83, 174.61], 58.27),
      "F": ([130.81, 174.61, 220.0], 87.31), "C": ([130.81, 164.81, 196.0], 65.41),
      "A": ([110.0, 138.59, 164.81], 55.0), "Gm": ([146.83, 196.0, 233.08], 98.0)}

# --- intro 0-4
add(impact(1.0), 0.0, 0.9)
add(rev_cymbal(0.9), 0.1, 0.25)
l, r = pad_note([36.71 * 2, 55.0 * 2, 73.42 * 2], 4.2, 600); add_st(l, r, 0.0, 0.55)
add(whoosh(0.6), 1.0 - 0.37, 0.5, -0.4)
add(impact(0.8), 1.0, 0.75)
add(taiko(), 1.5, 0.85, -0.2); add(bass_hit(), 1.5, 0.5)
add(taiko(), 2.0, 0.95, 0.2); add(bass_hit(), 2.0, 0.6)
for tk in (2.5, 2.75, 3.0, 3.25, 3.5, 3.625, 3.75, 3.875):
    add(taiko(), tk, 0.45 + 0.1 * (tk - 2.5), rng.uniform(-0.4, 0.4))
for i in range(16):
    add(hat(), 2.0 + i * 0.125, 0.12 + 0.012 * i, 0.3)
add(riser(1.5), 2.5, 0.45)
add(sparkle(1.0), 2.35, 0.25, 0.3)

# --- grooves
def groove(t0, t1, prog, energy=1.0, clap_on=False, arp=False, hats=True):
    bars = int(round((t1 - t0) / 2.0))
    for b in range(bars):
        chord = prog[b % len(prog)]; notes, root = CH[chord]; tb = t0 + b * 2.0
        l, r = pad_note([n * 2 for n in notes], 2.05, 1200 + 600 * energy); add_st(l, r, tb, 0.5)
        for q in range(4):
            tq = tb + q * BEAT
            add(kick(), tq, 0.72 * energy)
            if clap_on and q in (1, 3): add(clap(), tq, 0.5 * energy); add(snare(), tq, 0.25 * energy)
            if hats:
                add(hat(), tq + 0.25, 0.22 * energy, 0.25)
                add(hat(), tq + 0.125, 0.08 * energy, -0.3); add(hat(), tq + 0.375, 0.08 * energy, -0.3)
                if q == 3: add(hat(True), tq + 0.25, 0.12 * energy, 0.2)
            for e in (0.25,):  # off-beat bass (pumping)
                add(bass_note(root, 0.24), tq + e, 0.42 * energy)
            add(bass_note(root, 0.2), tq, 0.25 * energy)
        if arp:
            seq = [notes[0] * 4, notes[1] * 4, notes[2] * 4, notes[1] * 4 * 2, notes[2] * 4, notes[1] * 4, notes[0] * 4, notes[2] * 2]
            for s16 in range(16):
                add(pluck(seq[s16 % 8]), tb + s16 * 0.125, 0.10 * energy, 0.45 * np.sin(s16))
                add(pluck(seq[s16 % 8]), tb + s16 * 0.125 + 0.375, 0.035 * energy, -0.5)  # dotted-8th echo

groove(4.0, 10.0, ["Dm", "Bb", "C"], 0.8, clap_on=False, arp=False)
add(riser(1.0), 9.0, 0.4)
for i in range(8): add(snare(), 9.5 + i * 0.0625, 0.12 + 0.03 * i)
add(impact(0.9), 10.0, 0.7)
groove(10.0, 16.0, ["Dm", "Bb", "F"], 1.0, clap_on=True, arp=True)
add(riser(1.2), 14.8, 0.5)
for i in range(16): add(snare(), 15.0 + i * 0.0625, 0.08 + 0.02 * i)
add(impact(1.0), 16.0, 0.8)
# 16-17 breakdown: sub + pad only
l, r = pad_note([n * 2 for n in CH["Dm"][0]], 1.1, 900); add_st(l, r, 16.0, 0.6)
add(rev_cymbal(0.7), 16.3, 0.3)
# price reveal 17-23
add(bass_hit(), 17.0, 0.9); add(impact(0.7), 17.35, 0.55)
add(bass_hit(), 19.4, 0.9); add(impact(0.7), 19.75, 0.55)
groove(17.0, 23.0, ["Gm", "Dm", "A"], 0.85, clap_on=True, arp=False)
for ts in (20.3, 21.9, 20.6, 22.2): add(sparkle(0.8), ts, 0.18, rng.uniform(-0.5, 0.5))
add(riser(1.4), 21.6, 0.5); add(rev_cymbal(0.8), 22.2, 0.3)
# hero 23-28 + final
add(impact(1.0), 23.0, 0.85); add(sparkle(1.3), 23.15, 0.3, -0.2)
groove(23.0, 29.0, ["Dm", "Bb", "C"], 1.0, clap_on=True, arp=True)  # 3 bars (to 29), hard-cut at 28.5
lead = [(23.0, 587.33, 0.5), (23.5, 698.46, 0.5), (24.0, 880.0, 1.0), (25.0, 783.99, 0.5), (25.5, 698.46, 0.5),
        (26.0, 659.25, 1.0), (27.0, 698.46, 0.5), (27.5, 880.0, 0.5)]
for t0, f, d in lead: add(pluck(f, d + 0.2), t0, 0.06, 0.1)  # kept low: sits under the voice-over
add(sparkle(1.0), 26.2, 0.18, 0.4)
# hard stop at 28.5 -> final impact & tail
i28 = int(28.5 * SR)
fade = np.ones(N); fade[i28 - 240:i28] = np.linspace(1, 0, 240); fade[i28:] = 0
L *= fade; R *= fade
add(riser(0.6), 27.9, 0.4)
add(impact(1.15), 28.5, 1.0)
l, r = pad_note([n * 2 for n in CH["Dm"][0]] + [293.66], 2.0, 1500); add_st(l, r, 28.5, 0.5)
add(sparkle(1.2), 28.6, 0.22)

# whooshes on cuts
for tc, pan in ((4.0, 0.6), (6.0, -0.5), (8.0, 0.4), (10.0, -0.4), (12.0, 0.5), (14.0, -0.5), (16.0, 0.5), (17.0, -0.6), (19.4, 0.6), (23.0, 0.0)):
    add(whoosh(0.55), tc - 0.34, 0.42, pan)

# sizzle under food scenes
for t0, d, g in ((1.0, 3.0, 0.05), (4.0, 6.0, 0.05), (10.0, 2.0, 0.11), (12.0, 4.0, 0.06)):
    s = sizzle(d); env = np.minimum(1, tt(d) / 0.3) * np.minimum(1, (d - tt(d)) / 0.3)
    add(s * env, t0, g, rng.uniform(-0.3, 0.3))

# ------------------------------------------------------------------ original foley
def load_audio(path, ss, d):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(ss), "-t", str(d), "-i", path,
                                   "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"])
    x = np.frombuffer(raw, np.float32).astype(np.float64)
    x = filt(filt(x, "high", 250), "low", 9000)
    x /= (np.sqrt(np.mean(x ** 2)) + 1e-9)
    n = len(x); env = np.minimum(1, np.arange(n) / (0.15 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.15 * SR))
    return x * env
add(load_audio(f"{SRC}/vid_a_0323.mp4", 3.0, 2.0), 10.0, 0.06, -0.2)  # sambal frying
add(load_audio(f"{SRC}/vid_b_0325.mp4", 2.0, 2.0), 6.0, 0.035, 0.2)   # stirring
add(load_audio(f"{SRC}/vid_b_0325.mp4", 15.0, 2.0), 12.0, 0.04, 0.2)
add(load_audio(f"{SRC}/vid_b_0325.mp4", 9.0, 2.0), 14.0, 0.035, -0.2)

# ------------------------------------------------------------------ reverb + master
def reverb(x, d=2.2, mix=0.18, seed=0):
    g = np.random.default_rng(seed); t = tt(d)
    ir = g.standard_normal(len(t)) * np.exp(-t * 3.2); ir = filt(ir, "low", 6000); ir /= np.sqrt(np.sum(ir ** 2))
    pre = int(0.02 * SR)
    wet = signal.fftconvolve(x, ir)[:len(x)]
    wet = np.concatenate([np.zeros(pre), wet[:-pre]])
    return x + mix * wet
L = reverb(L, seed=1); R = reverb(R, seed=2)
# gentle glue: soft clip
peak = max(np.abs(L).max(), np.abs(R).max())
L = np.tanh(L / peak * 1.15) / np.tanh(1.15); R = np.tanh(R / peak * 1.15) / np.tanh(1.15)
end = np.ones(N); i0 = int(29.4 * SR); end[i0:] = np.linspace(1, 0, N - i0) ** 2
L *= end * 0.89; R *= end * 0.89
st = np.stack([L, R], 1).astype(np.float32)
import wave
pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
with wave.open(f"{WORK}/soundtrack_raw.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("ok")
