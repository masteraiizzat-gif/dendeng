"""Mix a female voice-over into the soundtrack with broadcast-style processing and ducking.

usage:
  python3 mix_vo.py <work_dir> <vo_dir> <out_wav>

<vo_dir> holds one recording per line, named after the ids in vo_script.json:
  v1.* v2.* ... v8.*   (wav / mp3 / m4a / ogg / opus — WhatsApp voice notes are fine)
or a single file full.* recorded in one take while watching the guide video
(it is then placed from 0.0 s as-is).

Each line is: silence-trimmed -> high-pass 80 Hz -> de-mud -> presence lift -> compressed ->
level-matched -> gently sped up (max 1.12x) only if it overruns its slot -> placed at its cue.
The music is ducked ~7 dB under the voice (30 ms attack, 300 ms release).
"""
import sys, os, glob, json, subprocess, wave
import numpy as np
from scipy import signal

WORK, VODIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000
SCRIPT = json.load(open(f"{HERE}/vo_script.json"))

def read_wav(p):
    with wave.open(p) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
        return x.reshape(-1, w.getnchannels())

CHAIN = ("highpass=f=80,equalizer=f=300:t=q:w=1.2:g=-2.5,equalizer=f=3500:t=q:w=1.0:g=3,"
         "equalizer=f=7500:t=q:w=2:g=-2,"
         "acompressor=threshold=-20dB:ratio=3.5:attack=8:release=120:makeup=3,"
         "loudnorm=I=-16:TP=-2:LRA=7")

def ff_f32(args, data=None):
    return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error"] + args, input=data, capture_output=True,
                                        check=True).stdout, np.float32).astype(np.float64)

def trim(x, rel_db=-38, pre=0.04, post=0.12):
    """cut leading/trailing silence (relative to the take's own peak level)"""
    win = int(0.02 * SR)
    if len(x) < win * 3: return x
    r = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, "same"))
    on = np.where(20 * np.log10(r + 1e-9) > 20 * np.log10(r.max() + 1e-9) + rel_db)[0]
    if len(on) == 0: return x
    return x[max(0, on[0] - int(pre * SR)): on[-1] + int(post * SR)]

def load_vo(path, tempo=1.0):
    x = trim(ff_f32(["-i", path, "-af", CHAIN, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]))
    if abs(tempo - 1) > 1e-3:
        x = ff_f32(["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", f"atempo={tempo:.4f}",
                    "-f", "f32le", "-"], x.astype(np.float32).tobytes())
    return x

music = read_wav(f"{WORK}/soundtrack_raw.wav")
N = music.shape[0]
vo = np.zeros(N)

full = glob.glob(f"{VODIR}/full.*")
if full:
    x = load_vo(full[0]); vo[:min(N, len(x))] = x[:N]
else:
    for ln in SCRIPT["lines"]:
        fs = sorted(glob.glob(f"{VODIR}/{ln['id']}.*"))
        if not fs:
            print("missing", ln["id"]); continue
        slot = ln["end"] - ln["start"] + 0.35
        x = load_vo(fs[0])
        dur = len(x) / SR
        if dur > slot:
            tempo = min(1.12, dur / slot)
            x = load_vo(fs[0], tempo)
            print(f"{ln['id']}: {dur:.2f}s > slot {slot:.2f}s -> atempo {tempo:.3f}")
        i = int(ln["start"] * SR)
        n = min(len(x), N - i)
        fade = np.minimum(1, np.arange(n) / (0.01 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.03 * SR))
        vo[i:i + n] += x[:n] * fade
        print(f"{ln['id']}: placed {ln['start']:.2f}-{ln['start'] + n / SR:.2f}s")

# short warm room so the voice sits in the same space as the music
t = np.arange(int(0.6 * SR)) / SR
ir = np.random.default_rng(5).standard_normal(len(t)) * np.exp(-t * 9)
ir = signal.sosfilt(signal.butter(2, 5000, "low", fs=SR, output="sos"), ir); ir /= np.sqrt(np.sum(ir ** 2))
vo = vo + 0.06 * signal.fftconvolve(vo, ir)[:N]

# ducking envelope
env = np.abs(vo)
att, rel = np.exp(-1 / (0.03 * SR)), np.exp(-1 / (0.30 * SR))
e = np.zeros(N); acc = 0.0
for k in range(0, N, 64):  # block-rate follower (fast enough, smooth after interpolation)
    v = env[k:k + 64].max()
    acc = att * acc + (1 - att) * v if v > acc else rel ** 64 * acc + (1 - rel ** 64) * v
    e[k:k + 64] = acc
e = np.clip(e / 0.08, 0, 1)
gain = 1 - 0.55 * e  # about -7 dB when the voice is present
mix = music * gain[:, None] + vo[:, None] * 0.95
peak = np.abs(mix).max()
if peak > 0.98: mix *= 0.98 / peak
with wave.open(OUT, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
print("wrote", OUT)
