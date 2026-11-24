"""Objective QA for generated voice-over takes (we cannot listen, so we measure):
  * Whisper large-v3-turbo transcription (Malay) -> character error rate vs the script
  * median pitch (F0, pYIN) -> confirms a female voice (~165-300 Hz)
  * duration vs. the slot in vo_script.json, clipping, long internal silences
Picks the best take per line and copies it to <pick_dir>/<id>.wav

usage: python3 qa_vo.py <takes_dir> <pick_dir>
"""
import sys, os, re, json, glob, shutil
import warnings; warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, torch
from transformers import pipeline

TAKES, PICK = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(PICK, exist_ok=True)
SC = {l["id"]: l for l in json.load(open(f"{HERE}/vo_script.json"))["lines"]}
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cpu",
               torch_dtype=torch.float32)

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))

# lines with numbers: Whisper writes digits, so check the digit sequence instead of the words
DIGITS = {"v5": ["300", "27"], "v6": ["500", "45"], "v8": ["0187884575"]}
SPOKEN = {"oda": "order", "oder": "order", "odah": "order"}

def text_error(vid, ref, hyp):
    if vid in DIGITS:
        d = re.sub(r"\D", "", hyp)
        ok = all(x in d for x in DIGITS[vid]) and len(d) == sum(len(x) for x in DIGITS[vid])
        return 0.0 if ok else 1.0
    h = " ".join(SPOKEN.get(w, w) for w in norm(hyp).split())
    return cer(ref, h)

report = []
for vid, ln in SC.items():
    best = None
    for f in sorted(glob.glob(f"{TAKES}/{vid}_take*.wav")):
        y, sr = sf.read(f)
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
        # trim for metrics
        yt, _ = librosa.effects.trim(y16, top_db=35)
        dur = len(yt) / 16000
        txt = asr({"raw": y16.astype(np.float32), "sampling_rate": 16000},
                  generate_kwargs={"language": "ms", "task": "transcribe"})["text"]
        c = text_error(vid, ln["text"], txt)
        f0, vf, _ = librosa.pyin(yt, fmin=70, fmax=500, sr=16000)
        f0m = float(np.nanmedian(f0)) if np.any(vf) else 0.0
        # longest internal silence
        iv = librosa.effects.split(yt, top_db=35)
        gaps = [(iv[i + 1][0] - iv[i][1]) / 16000 for i in range(len(iv) - 1)]
        gap = max(gaps) if gaps else 0.0
        clip = float(np.mean(np.abs(y) > 0.99))
        slot = ln["end"] - ln["start"] + 0.35
        over = max(0.0, dur / 1.12 - slot)        # what's left after max allowed speed-up
        score = c * 10 + over * 4 + max(0, gap - 0.8) * 3 + (5 if f0m < 160 else 0) + clip * 50
        row = dict(id=vid, file=os.path.basename(f), text=txt.strip(), cer=round(c, 3), f0=round(f0m),
                   dur=round(dur, 2), slot=round(slot, 2), gap=round(gap, 2), score=round(score, 2))
        report.append(row)
        print(row, flush=True)
        if best is None or score < best[0]: best = (score, f, row)
    if best:
        shutil.copy(best[1], f"{PICK}/{vid}.wav")
        print(f">>> {vid}: picked {os.path.basename(best[1])}  '{best[2]['text']}'", flush=True)
json.dump(report, open(f"{PICK}/qa_report.json", "w"), indent=1, ensure_ascii=False)
