"""Extract the selected frame ranges from the stabilised + motion-interpolated (60 fps)
clips into .npy arrays the renderer can memory-map.

usage: python3 prep_clips.py <work_dir>
  expects <work_dir>/slow60_{a_0323,b_0325,c_1201}.mp4 (see README for the ffmpeg prep steps)
"""
import subprocess, sys, json
import numpy as np

WORK = sys.argv[1]
# (name, file, start_sec, end_sec)  -- source seconds at 60 fps
RANGES = [
    ("a_colander", "slow60_a_0323.mp4", 0.00, 0.98),
    ("a_sambal",   "slow60_a_0323.mp4", 2.90, 7.05),
    ("b_stir1",    "slow60_b_0325.mp4", 1.90, 3.40),
    ("b_stir2",    "slow60_b_0325.mp4", 14.9, 17.6),
    ("b_stir3",    "slow60_b_0325.mp4", 8.9, 10.3),
    ("c_packs",    "slow60_c_1201.mp4", 7.90, 9.55),
]

def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height", "-of", "json", path])
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]

meta = {}
for name, f, a, b in RANGES:
    path = f"{WORK}/{f}"
    w, h = probe(path)
    i0, i1 = int(round(a * 60)), int(round(b * 60))
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vf",
                                   f"select='between(n\\,{i0}\\,{i1})'", "-vsync", "0",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    np.save(f"{WORK}/{name}.npy", arr)
    meta[name] = {"start": i0 / 60.0, "frames": int(arr.shape[0]), "w": w, "h": h}
    print(name, arr.shape)
json.dump(meta, open(f"{WORK}/clips.json", "w"), indent=1)
