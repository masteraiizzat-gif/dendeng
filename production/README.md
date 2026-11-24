# DENDENG PADU — 30 s cinematic vertical commercial

Final deliverables are in `../exports/`:

| File | Use |
|---|---|
| `DENDENG_BERAPI_30s_tanpaVO_1080x1920_master.mp4` | v2 master (new hook, brand, phone number). Music only, voice-over pending. TikTok / Reels (H.264 High, 1080×1920, 30 fps, AAC 320 kbps, −14 LUFS) |
| `DENDENG_BERAPI_30s_tanpaVO_1080x1920_whatsapp.mp4` | Same, WhatsApp Status size (under 16 MB) |
| `PANDUAN_RAKAMAN_VO.mp4` | Recording guide: the ad with each voice-over line captioned at the moment it must be spoken |

v1 (`DENDENG_PADU_30s_*`) is in git history.

## v2 changes
* Opening hook on screen: **RASA PEDAS YANG PADU / DENDENG BERAPI!** (replaces "INI BARU DENDENG PADU!")
* Price scene header: **DENDENG BERAPI**. The 500G card moved to 19.4 s to match the voice-over.
* Final scene: **HUBUNGI +6018-788 4575** added below ORDER SEKARANG!, and the final impact moved to 28.5 s.
* Voice-over script and timings: `vo_script.json` / `SKRIP_VOICE_OVER.md`. Mixer: `mix_vo.py`. Export: `export.sh`.

### Adding the voice-over
```bash
python3 mix_vo.py $WORK /path/to/vo_files $WORK/mix_with_vo.wav   # v1.* … v8.*  or  full.*
./export.sh $WORK $WORK/mix_with_vo.wav ../exports DENDENG_BERAPI_30s_VO
```

## Source material (Google Drive folder, all 6 files reviewed)

| File | Used | Where |
|---|---|---|
| WhatsApp Image 2026-11-23 at 3.22.36 PM (finished dendeng, chili garnish) | ✅ | 1–4 s opening title push-in, 8–10 s chili macro / rack focus |
| WhatsApp Image 2026-11-23 at 3.25.47 PM (beef in gravy) | ✅ | 4–6 s macro with tilt-shift depth of field |
| WhatsApp Image 2026-11-24 at 12.01.20 PM (packaging, dark bokeh) | ✅ | 16–30 s product reveal, price backdrop, final hero |
| WhatsApp Video 2026-11-23 at 3.23.22 PM (colander beef + sambal frying) | ✅ | 0–1 s opening macro; 10–12 s sizzle with speed ramp; original sizzle audio |
| WhatsApp Video 2026-11-23 at 3.25.16 PM (stirring finished dendeng) | ✅ | 6–8 s, 12–14 s, 14–16 s slow motion; original stirring audio |
| WhatsApp Video 2026-11-24 at 12.01.20 PM (packs in freezer) | ❌ | Reviewed and dropped. Mostly shaky or blurred through frosted glass, and the steady section still graded flat grey. The packaging photo shows the same real packs much better. |

## Edit decision list (120 BPM, every cut lands on a beat)

| Time | Shot | Effects | Text |
|---|---|---|---|
| 0–1 s | Video A colander beef | flash, deep hit, fast push-in, zoom blur | — |
| 1–4 s | Finished dendeng photo | push-in, embers, chili flakes, light sweep, slams, camera shake | INI BARU / DENDENG / PADU! |
| 4–6 s | Cooking photo | macro drift, orbit keystone, tilt-shift DOF | — |
| 6–8 s | Video B | 0.5× motion-interpolated slow motion, DOF | — |
| 8–10 s | Finished photo (chili) | rack focus + pull-out | — |
| 10–12 s | Video A sambal | speed ramp 2.4× → 0.45×, sizzle | RASA YANG BUAT / NAK LAGI! |
| 12–14 s | Video B | speed ramp 1.6× → 0.5×, spice particles | (held) |
| 14–16 s | Video B (sharpest section) | 0.5× slow motion, push | (held) |
| 16–17 s | Packaging plate | product reveal pull-back, rack to defocus | — |
| 17–23 s | Packaging plate (defocused, orbit) | gold glass price cards, bass impacts, light sweeps | DENDENG PADU · 300G RM27 · 500G RM45 |
| 23–30 s | Packaging hero | rack into focus, specular light sweep, final impact | DENDENG PADU, / RASA MEMIKAT! / 300G RM27 \| 500G RM45 / ORDER SEKARANG! |

No website, promotion, certification or ingredient claim was added. The phone number (+6018-788 4575) is shown in v2 at the client's request, and it matches the packaging label.

## Rebuild from scratch

Requirements: ffmpeg (with vid.stab), Python 3 with `numpy scipy opencv-python-headless pillow`.

```bash
SRC=/path/to/src   # Drive files renamed: img_finished.jpg img_cooking.jpg img_packaging.jpg vid_a_0323.mp4 vid_b_0325.mp4 vid_c_1201.mp4
WORK=/path/to/work
cd $WORK
for f in a_0323 b_0325 c_1201; do
  ffmpeg -i $SRC/vid_$f.mp4 -vf vidstabdetect=shakiness=6:accuracy=15:result=$f.trf -f null -
  ffmpeg -i $SRC/vid_$f.mp4 -vf "vidstabtransform=input=$f.trf:smoothing=20:zoom=4:interpol=bicubic,hqdn3d=1.5:1.5:4:4" -an -c:v libx264 -crf 12 stab_$f.mp4
  ffmpeg -i stab_$f.mp4 -vf "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" -c:v libx264 -crf 12 slow60_$f.mp4
done
python3 prep_clips.py $WORK              # extract selected 60 fps ranges
python3 render.py $SRC $WORK render 4    # picture -> $WORK/video_master.mkv
python3 music.py  $SRC $WORK             # soundtrack -> $WORK/soundtrack_raw.wav
# mux + loudness normalise (see export commands in the session / below)
ffmpeg -i $WORK/video_master.mkv -i $WORK/soundtrack_raw.wav -af loudnorm=I=-14:TP=-1:LRA=11 \
  -c:v libx264 -profile:v high -preset slow -crf 16 -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:a aac -b:a 320k -ar 48000 -movflags +faststart -shortest DENDENG_PADU_30s_1080x1920_master.mp4
```

To preview any moment as stills: `python3 render.py $SRC $WORK preview 2.6 18.0 27.0`

Text, timings and prices are in `render.py` → `build_text()` / `text_layer()`. The music arrangement is in `music.py`.

## Notes / limitations

* The source footage is WhatsApp-compressed (videos 464×832 and 576×1024, photos about 960×1280). The final video is 1080×1920. It was upscaled with Lanczos, light sharpening and film grain. AI upscaling was avoided because the brief rules out artificial textures. A 4K export would add no real detail, so none was made. Re-shooting in 4K, or sending the original phone files instead of WhatsApp copies, would sharpen the result noticeably.
* The music and sound effects are original. They were synthesised in `music.py`, so there are no licensing issues for paid ads. If you prefer a licensed track, replace `soundtrack_raw.wav` and keep the hits at 0, 1, 1.5, 2, 10, 16, 17, 18.5, 23 and 28 s.
* Fonts: Anton and Cinzel (SIL Open Font License, `fonts/OFL-LICENSE.txt`), free for commercial use.
