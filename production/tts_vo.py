"""Generate the female Malay voice-over lines locally (no API key, no cloud TTS):
mesolitica/malaysian-parler-tts-mini-v1 (Parler-TTS mini, Apache-2.0 base) speaker 'Shafiqah Idayu',
DAC 44 kHz codec (MIT), all weights in safetensors.

Needs:  pip install parler-tts   (pulls transformers 4.46 — use a separate venv)
usage:  python tts_vo.py <out_dir> [takes_per_line] [ids...]
   ->   <out_dir>/<id>_take<k>.wav   (44.1 kHz mono)
"""
import sys, os, time
import warnings; warnings.filterwarnings("ignore")
import torch, soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

OUT = sys.argv[1]; TAKES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ONLY = set()
OVERRIDE = {}
for a in sys.argv[3:]:  # "v7" or "v7=Custom text"
    k, _, v = a.partition("=")
    ONLY.add(k)
    if v: OVERRIDE[k] = v
K0 = int(os.environ.get("TAKE0", "0"))  # take-number offset, to add takes without overwriting
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count())
MODEL = "mesolitica/malaysian-parler-tts-mini-v1"
SPEAKER = "Shafiqah Idayu"   # female Malaysian speaker of the model
# text sent to the model (numbers spelled out, punctuation tuned for pacing); ids match vo_script.json
TTS_TEXT = {
    "v1": "Rasa pedas, yang padu...",
    "v2": "Dendeng Berapi!",
    "v3": "Setiap suapan... pedasnya menggoda, rasanya memikat.",
    "v4": "Rasa yang buat nak lagi... dan lagi!",
    "v5": "Tiga ratus gram, dua puluh tujuh ringgit.",
    "v6": "Lima ratus gram, empat puluh lima ringgit.",
    "v7": "Oder sekarang!",  # phonetic spelling: model reads "Order" weakly
    "v8": "Hubungi, kosong satu lapan, tujuh lapan lapan, empat lima tujuh lima.",
}

t0 = time.time()
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
SR = model.config.sampling_rate
print(f"loaded in {time.time() - t0:.0f}s, sr={SR}", flush=True)
desc = tok(SPEAKER, return_tensors="pt")

for vid, text in TTS_TEXT.items():
    if ONLY and vid not in ONLY: continue
    text = OVERRIDE.get(vid, text)
    p = tok(text, return_tensors="pt")
    for k in range(K0, K0 + TAKES):
        torch.manual_seed(4242 + 31 * k + int(vid[1:]))
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                                 prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                                 do_sample=True, temperature=0.9)
        wav = gen.cpu().numpy().squeeze()
        sf.write(f"{OUT}/{vid}_take{k}.wav", wav, SR)
        print(f"{vid} take{k}: {len(wav) / SR:.2f}s audio in {time.time() - t1:.0f}s", flush=True)
