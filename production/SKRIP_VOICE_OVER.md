# Skrip Voice-Over — Iklan Dendeng Berapi (30 saat)

**Suara:** perempuan, Bahasa Melayu Malaysia. Nada mesra, yakin dan sedikit menggoda, tanpa menjerit. **Senyum semasa bercakap**, kerana senyuman memang kedengaran dalam suara.

| Fail | Masa dalam video | Baca | Arahan |
|---|---|---|---|
| `v1` | 0.3 – 2.2 s | **Rasa pedas yang padu...** | HOOK. Perlahan, dekat dengan mic, hampir berbisik. Tekan perkataan "padu". |
| `v2` | 2.3 – 3.7 s | **Dendeng Berapi!** | Terang dan bertenaga. Sebut nama jenama dengan bangga. |
| `v3` | 4.6 – 9.3 s | **Setiap suapan... pedasnya menggoda, rasanya memikat.** | Lembut, perlahan, berselera. Jeda kecil selepas "suapan". |
| `v4` | 10.6 – 14.2 s | **Rasa yang buat nak lagi... dan lagi!** | Main-main dan ceria. "dan lagi!" sambil senyum. |
| `v5` | 17.0 – 19.1 s | **Tiga ratus gram, dua puluh tujuh ringgit.** | Jelas dan yakin. |
| `v6` | 19.4 – 21.7 s | **Lima ratus gram, empat puluh lima ringgit.** | Sama nada dengan v5. |
| `v7` | 23.5 – 24.6 s | **Order sekarang!** | Ajakan, bertenaga. |
| `v8` | 24.8 – 28.3 s | **Hubungi kosong satu lapan, tujuh lapan lapan, empat lima tujuh lima.** | Perlahan dan sangat jelas. Jeda kecil antara kumpulan nombor. |

## Cara rakam (guna telefon pun boleh)

1. Pilih bilik yang senyap dan berlangsir atau bertilam, bukan bilik air atau dapur. Matikan kipas dan aircond.
2. Pegang telefon kira-kira sejengkal dari mulut, sedikit ke tepi supaya bunyi "p" dan "b" tidak meletup.
3. Rakam **setiap baris sebagai fail berasingan**, dan namakan fail `v1`, `v2` … `v8`. Voice note WhatsApp atau Voice Memos pun boleh. Rakam 2–3 kali bagi setiap baris dan hantar yang terbaik.
4. Jika mahu rakam sekali jalan, mainkan `exports/PANDUAN_RAKAMAN_VO.mp4` pada telefon lain dengan headphone, kemudian baca teks kuning ketika ia muncul. Hantar sebagai satu fail bernama `full`.
5. Muat naik ke folder Google Drive yang sama.

Selepas itu, `production/mix_vo.py` akan proses suara secara automatik: buang bunyi senyap, EQ, kompresi dan reverb ringan. Ia juga letakkan setiap baris pada masanya dan rendahkan muzik (ducking) semasa suara masuk.
