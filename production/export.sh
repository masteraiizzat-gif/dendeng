#!/usr/bin/env bash
# usage: export.sh <work_dir> <audio_wav> <out_dir> <basename>
# -> <basename>_1080x1920_master.mp4 (14 Mbps, AAC 320k)  and  <basename>_1080x1920_whatsapp.mp4 (<16 MB)
set -euo pipefail
W=$1; A=$2; O=$3; B=$4
J=$(ffmpeg -hide_banner -i "$A" -af loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
MI=$(echo "$J" | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}\")")
ffmpeg -v error -y -i "$A" -af "loudnorm=I=-14:TP=-1.0:LRA=11:$MI:linear=true,aresample=48000" -c:a pcm_s16le "$W/audio_final.wav"
COMMON="-map 0:v -map 1:a -c:v libx264 -profile:v high -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv -r 30 -g 30 -movflags +faststart -shortest"
enc() { # bitrate maxrate abitrate outfile
  ffmpeg -v error -y -i "$W/video_master.mkv" -an -c:v libx264 -profile:v high -pix_fmt yuv420p -r 30 -g 30 -preset slow -b:v $1 -pass 1 -passlogfile "$W/p_$4" -f null /dev/null
  ffmpeg -v error -y -i "$W/video_master.mkv" -i "$W/audio_final.wav" $COMMON -preset slow -b:v $1 -maxrate $2 -bufsize $2 -pass 2 -passlogfile "$W/p_$4" -c:a aac -b:a $3 -ar 48000 "$O/$4"
}
enc 14M 20M 320k "${B}_1080x1920_master.mp4" &
enc 3700k 6M 160k "${B}_1080x1920_whatsapp.mp4" &
wait
ls -la "$O"/"${B}"_*
