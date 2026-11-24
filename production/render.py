"""DENDENG PADU — 30 s vertical cinematic commercial renderer.

Composites the ORIGINAL footage/photos (graded, stabilised, slow-mo, camera moves,
DOF, particles, light sweeps, typography) frame by frame at 1080x1920 / 30 fps.

usage:
  python3 render.py <src_dir> <work_dir> preview t1 t2 ...   -> PNG stills in work_dir/preview
  python3 render.py <src_dir> <work_dir> render  [workers]   -> work_dir/video_noaudio.mp4
"""
import sys, os, json, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)
SRC, WORK, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = {
    "anton": f"{HERE}/fonts/anton.woff",
    "cinzel": f"{HERE}/fonts/cinzel900.woff",
    "cinzel7": f"{HERE}/fonts/cinzel700.woff",
    "inter": "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf",
}
W, H, FPS, DUR = 1080, 1920, 30, 30.0
NF = int(DUR * FPS)

# ----------------------------------------------------------------- helpers
def clamp01(x): return np.clip(x, 0.0, 1.0)
def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0); return t * t * (3 - 2 * t)
def ease_out_expo(t): t = min(max(t, 0.0), 1.0); return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)
def ease_out_cubic(t): t = min(max(t, 0.0), 1.0); return 1 - (1 - t) ** 3
def ease_io(t): t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)
def ease_out_back(t, s=1.6):
    t = min(max(t, 0.0), 1.0) - 1; return 1 + t * t * ((s + 1) * t + s)
def lerp(a, b, t): return a + (b - a) * t

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
RR = np.sqrt(((XX - W / 2) / (W / 2)) ** 2 * 0.85 + ((YY - H * 0.48) / (H / 2)) ** 2)

def vignette(strength):
    return (1.0 - strength * sstep(0.35, 1.25, RR)).astype(np.float32)[..., None]

def load_img(name):
    im = cv2.imread(f"{SRC}/{name}", cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

CLIPMETA = json.load(open(f"{WORK}/clips.json"))
_clip_cache = {}
def clip(name):
    if name not in _clip_cache:
        _clip_cache[name] = np.load(f"{WORK}/{name}.npy", mmap_mode="r")
    return _clip_cache[name]

def clip_frame(name, src_t):
    """Frame-blended sample of a 60 fps clip at source time src_t (seconds)."""
    m = CLIPMETA[name]; arr = clip(name)
    f = (src_t - m["start"]) * 60.0
    f = min(max(f, 0.0), m["frames"] - 1.001)
    i0 = int(f); a = f - i0
    if a < 0.02: return np.asarray(arr[i0])
    return cv2.addWeighted(np.asarray(arr[i0]), 1 - a, np.asarray(arr[i0 + 1]), a, 0)

# ----------------------------------------------------------------- camera
def camera(src, zoom, cx, cy, rot=0.0, persp=(0.0, 0.0)):
    """Virtual camera over a source image. zoom is relative to 'cover' fit.
    cx, cy: centre in source pixels. persp: (horizontal orbit, vertical tilt) keystone amounts."""
    h, w = src.shape[:2]
    s = max(W / w, H / h) * zoom
    hw, hh = W / s / 2, H / s / 2
    px, py = persp
    # keystone: shrink one side to simulate orbiting around the subject
    corners = np.array([[-hw * (1 - py), -hh * (1 + px)], [hw * (1 - py), -hh * (1 - px)],
                        [hw * (1 + py), hh * (1 - px)], [-hw * (1 + py), hh * (1 + px)]], np.float32)
    c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    R = np.array([[c, -sn], [sn, c]], np.float32)
    quad = corners @ R.T + np.array([cx, cy], np.float32)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    out = cv2.warpPerspective(src, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)
    return out

# ----------------------------------------------------------------- grade
def grade(img_u8, exposure=1.0, crush=0.0, contrast=1.25, sat=1.12, warm=1.0, vig=0.55,
          sharpen=0.5, black=0.03):
    x = img_u8.astype(np.float32) / 255.0
    if sharpen > 0:  # restore micro-contrast lost in WhatsApp compression / upscaling
        bl = cv2.GaussianBlur(x, (0, 0), 1.6)
        x = x + sharpen * (x - bl)
    x = x * exposure
    lum = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
    if crush > 0:  # pull bright, neutral areas (white pan, wall tiles) down -> dark premium backdrop
        mx, mn = x.max(-1), x.min(-1)
        satm = (mx - mn) / (mx + 1e-4)
        m = sstep(0.42, 0.85, lum) * (1 - sstep(0.10, 0.38, satm))
        x = x * (1 - crush * m)[..., None]
        lum = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
    x = clamp01(x)
    p = contrast
    xp = x ** p; x = xp / (xp + (1 - x) ** p + 1e-6)
    x = clamp01((x - black) / (1 - black))
    lum = (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722)[..., None]
    x = lum + (x - lum) * sat
    # warm golden split-tone
    sh = (1 - lum) ** 2
    x = x + warm * sh * np.array([0.022, 0.009, -0.012], np.float32)
    x = x * (1 + warm * lum * np.array([0.05, 0.005, -0.07], np.float32))
    x = x * vignette(vig)
    return clamp01(x)

def bloom(x, amt=0.28, thr=0.68):
    small = cv2.resize(x, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hi = np.maximum(small - thr, 0)
    hi = cv2.GaussianBlur(hi, (0, 0), 9)
    hi = cv2.resize(hi, (W, H), interpolation=cv2.INTER_LINEAR)
    return x + amt * hi * np.array([1.0, 0.85, 0.6], np.float32)

def blur_frame(x, sigma):
    if sigma <= 0.3: return x
    small = cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 2)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

def tilt_shift(x, focus_y, band, sigma):
    bl = blur_frame(x, sigma)
    m = sstep(band, band + 520, np.abs(YY - focus_y))[..., None]
    return x * (1 - m) + bl * m

# ----------------------------------------------------------------- particles
rng = np.random.default_rng(7)
EMB_N = 90
EMB = dict(x=rng.uniform(0, W, EMB_N), y=rng.uniform(0, H + 300, EMB_N), v=rng.uniform(60, 210, EMB_N),
           sz=rng.uniform(1.2, 4.5, EMB_N), ph=rng.uniform(0, 6.28, EMB_N), fr=rng.uniform(0.4, 1.3, EMB_N),
           fl=rng.uniform(3, 11, EMB_N), big=rng.uniform(0, 1, EMB_N) < 0.16)

def embers(x, t, intensity):
    if intensity <= 0: return x
    lay_s = np.zeros((H // 2, W // 2, 3), np.float32); lay_b = np.zeros_like(lay_s)
    for i in range(EMB_N):
        yy = (EMB["y"][i] - EMB["v"][i] * t) % (H + 300) - 150
        xx = EMB["x"][i] + 40 * math.sin(EMB["ph"][i] + t * EMB["fr"][i]) + 12 * math.sin(t * 3.1 + i)
        br = 0.55 + 0.45 * math.sin(EMB["fl"][i] * t + EMB["ph"][i])
        fade = float(sstep(-120, 260, yy))  # fade out as they rise off the top
        col = (1.0 * br, 0.50 * br, 0.12 * br)
        if EMB["big"][i]:
            cv2.circle(lay_b, (int(xx / 2), int(yy / 2)), int(EMB["sz"][i] * 5), tuple(c * 0.35 * fade for c in col), -1, cv2.LINE_AA)
        else:
            cv2.circle(lay_s, (int(xx / 2), int(yy / 2)), max(1, int(EMB["sz"][i] / 2 + 0.5)), tuple(c * fade for c in col), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay_s, (0, 0), 1.2) * 1.6 + cv2.GaussianBlur(lay_s, (0, 0), 5) * 1.2 + cv2.GaussianBlur(lay_b, (0, 0), 7)
    lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
    return x + intensity * lay

# spice / chili flakes (real dendeng is garnished with chili) — subtle, depth-layered
SP_N = 46
def _flake_shape(r):
    k = rng.integers(5, 8); ang = np.sort(rng.uniform(0, 6.28, k))
    rad = rng.uniform(0.55, 1.0, k)
    return np.stack([np.cos(ang) * rad, np.sin(ang) * rad * rng.uniform(0.45, 0.9)], 1)
SP = []
for i in range(SP_N):
    kind = rng.choice(["chili", "chili", "chili", "dark", "seed"])
    depth = rng.choice([0, 0, 0, 1])  # 1 = out-of-focus foreground
    size = (rng.uniform(5, 13) if depth == 0 else rng.uniform(22, 40))
    col = {"chili": (0.78, 0.11, 0.04), "dark": (0.38, 0.07, 0.03), "seed": (0.93, 0.78, 0.45)}[kind]
    col = tuple(np.clip(np.array(col) * rng.uniform(0.8, 1.15), 0, 1))
    SP.append(dict(kind=kind, depth=depth, size=size * (0.6 if kind == "seed" else 1), col=col,
                   x=rng.uniform(-50, W + 50), y=rng.uniform(-200, H), vx=rng.uniform(-35, 35),
                   vy=rng.uniform(45, 130) * (1.6 if depth else 1), rot=rng.uniform(0, 6.28),
                   w=rng.uniform(-2.5, 2.5), shape=_flake_shape(1), ph=rng.uniform(0, 6.28)))

def spices(x, t, intensity, t0):
    if intensity <= 0: return x
    lt = t - t0
    for dep, scale, blur in ((0, 1, 0.8), (1, 2, 9)):
        col = np.zeros((H // scale, W // scale, 3), np.float32); a = np.zeros((H // scale, W // scale), np.float32)
        for p in SP:
            if p["depth"] != dep: continue
            yy = (p["y"] + p["vy"] * lt) % (H + 300) - 150
            xx = p["x"] + p["vx"] * lt + 25 * math.sin(lt * 1.3 + p["ph"])
            r = p["rot"] + p["w"] * lt
            c, s = math.cos(r), math.sin(r)
            pts = p["shape"] @ np.array([[c, s], [-s, c]]) * p["size"] + [xx, yy]
            pts = (pts / scale).astype(np.int32)
            cv2.fillPoly(col, [pts], p["col"], cv2.LINE_AA)
            cv2.fillPoly(a, [pts], 1.0, cv2.LINE_AA)
        col = cv2.GaussianBlur(col, (0, 0), blur); a = cv2.GaussianBlur(a, (0, 0), blur)
        if scale != 1:
            col = cv2.resize(col, (W, H)); a = cv2.resize(a, (W, H))
        k = intensity * (0.75 if dep else 0.95)
        x = x * (1 - k * a[..., None]) + k * col
    return x

def light_sweep(x, t, t0, dur, angle=-28, width=150, amt=0.55, lummask=True):
    if not (t0 <= t <= t0 + dur): return x
    k = ease_io((t - t0) / dur)
    a = math.radians(angle)
    d = XX * math.cos(a) + YY * math.sin(a)
    lo, hi = d.min(), d.max()
    pos = lerp(lo - 400, hi + 400, k)
    band = np.exp(-((d - pos) / width) ** 2)
    if lummask:
        lum = x.mean(-1)
        band = band * sstep(0.18, 0.75, lum)
    return x + amt * band[..., None] * np.array([1.0, 0.92, 0.75], np.float32)

GRAIN = None
def grain(x, fi, amt=0.035):
    global GRAIN
    if GRAIN is None:
        g = np.random.default_rng(3)
        GRAIN = [cv2.GaussianBlur(g.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7) for _ in range(8)]
    lum = x.mean(-1, keepdims=True)
    return x + amt * GRAIN[fi % 8][..., None] * (0.35 + 0.65 * (1 - np.abs(lum - 0.45) * 1.6).clip(0.2, 1))

# ----------------------------------------------------------------- transitions
def zoom_blur(x, k, cx=W / 2, cy=H / 2):
    if k <= 0.01: return x
    small = cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small); n = 9
    for i in range(n):
        s = 1 + 0.32 * k * i / (n - 1)
        M = np.array([[s, 0, (1 - s) * cx / 2], [0, s, (1 - s) * cy / 2]], np.float32)
        acc += cv2.warpAffine(small, M, (W // 2, H // 2), borderMode=cv2.BORDER_REFLECT101)
    acc /= n
    return cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)

def whip(x, k, dx, dy):
    if k <= 0.01: return x
    small = cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    L = int(2 + 150 * k)
    ker = np.zeros((L * 2 + 1, L * 2 + 1), np.float32)
    cv2.line(ker, (L - int(L * dx), L - int(L * dy)), (L + int(L * dx), L + int(L * dy)), 1.0, 1)
    ker /= ker.sum()
    small = cv2.filter2D(small, -1, ker, borderType=cv2.BORDER_REFLECT101)
    M = np.array([[1, 0, dx * 260 * k], [0, 1, dy * 260 * k]], np.float32)
    small = cv2.warpAffine(small, M, (W // 2, H // 2), borderMode=cv2.BORDER_REFLECT101)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

# ----------------------------------------------------------------- text
_font_cache = {}
def font(name, size):
    k = (name, size)
    if k not in _font_cache: _font_cache[k] = ImageFont.truetype(FONT[name], size)
    return _font_cache[k]

def gold_grad(h):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    stops = [(0.0, (1.00, 0.95, 0.74)), (0.40, (0.99, 0.80, 0.36)), (0.52, (0.80, 0.50, 0.14)),
             (0.70, (0.95, 0.70, 0.28)), (1.0, (1.00, 0.88, 0.52))]
    out = np.zeros((h, 3), np.float32)
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        m = ((t[:, 0] >= a) & (t[:, 0] <= b))
        u = ((t[:, 0] - a) / (b - a))[m][:, None]
        out[m] = np.array(ca) * (1 - u) + np.array(cb) * u
    return out[:, None, :]

FILLS = {"white": (0.98, 0.96, 0.92), "dark": (0.10, 0.05, 0.02), "cream": (1.0, 0.92, 0.78)}

class Elem:
    """Baked text/graphic layer: premultiplied colour, alpha, shadow alpha, glow alpha."""
    def __init__(self, col, a, glow_col=(1.0, 0.42, 0.08), glow=0.45, shadow=0.75):
        self.col, self.a = col, a
        self.sh = cv2.GaussianBlur(a, (0, 0), 13)
        self.gl = cv2.GaussianBlur(a, (0, 0), 30)
        self.glow_col = np.array(glow_col, np.float32); self.glow, self.shadow = glow, shadow
        self.h, self.w = a.shape

def text_elem(segments, pad=90, gap=0, **kw):
    """segments: list of (text, fontname, size, fill, tracking)"""
    parts = []
    for text, fn, size, fill, tr in segments:
        f = font(fn, size)
        asc, desc = f.getmetrics()
        width = int(sum(f.getlength(ch) + tr for ch in text) - tr) + 4
        im = Image.new("L", (width, asc + desc), 0); d = ImageDraw.Draw(im)
        xx = 0
        for ch in text:
            d.text((xx, 0), ch, font=f, fill=255); xx += f.getlength(ch) + tr
        a = np.asarray(im, np.float32) / 255.0
        ys = np.where(a.max(1) > 0.02)[0]
        a = a[max(ys[0] - 2, 0): ys[-1] + 3]
        parts.append((a, fill))
    hmax = max(p[0].shape[0] for p in parts)
    wtot = sum(p[0].shape[1] for p in parts) + gap * (len(parts) - 1)
    A = np.zeros((hmax + 2 * pad, wtot + 2 * pad), np.float32); C = np.zeros(A.shape + (3,), np.float32)
    xx = pad
    for a, fill in parts:
        h, w = a.shape; y0 = pad + (hmax - h)  # baseline-align on bottom
        if fill == "gold": col = np.broadcast_to(gold_grad(h), (h, w, 3))
        else: col = np.broadcast_to(np.array(FILLS[fill], np.float32), (h, w, 3))
        A[y0:y0 + h, xx:xx + w] = np.maximum(A[y0:y0 + h, xx:xx + w], a)
        C[y0:y0 + h, xx:xx + w] += col * a[..., None]
        xx += w + gap
    return Elem(C, A, **kw)

def rrect_mask(w, h, r, pad):
    m = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.rectangle(m, (pad + r, pad), (pad + w - r, pad + h), 255, -1)
    cv2.rectangle(m, (pad, pad + r), (pad + w, pad + h - r), 255, -1)
    for cx, cy in ((pad + r, pad + r), (pad + w - r, pad + r), (pad + r, pad + h - r), (pad + w - r, pad + h - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 0.8)

def comp(x, e, cx, cy, scale=1.0, op=1.0, blur=0.0, sweep=None, reveal=None, glow_boost=0.0):
    """Composite element e centred at (cx, cy)."""
    if op <= 0.003 or scale <= 0.01: return x
    col, a, sh, gl = e.col, e.a, e.sh, e.gl
    if abs(scale - 1) > 1e-3:
        sz = (max(2, int(e.w * scale)), max(2, int(e.h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        col, a, sh, gl = [cv2.resize(v, sz, interpolation=interp) for v in (col, a, sh, gl)]
    if blur > 0.3:
        col, a = cv2.GaussianBlur(col, (0, 0), blur), cv2.GaussianBlur(a, (0, 0), blur)
    h, w = a.shape
    if reveal is not None:  # soft left->right wipe
        ramp = sstep(reveal * (w + 200) - 200, reveal * (w + 200), np.arange(w, dtype=np.float32))
        m = (1 - ramp)[None, :]
        col, a, sh, gl = col * m[..., None], a * m, sh * m, gl * m
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0: return x
    sx, sy = X0 - x0, Y0 - y0
    sl = (slice(sy, sy + Y1 - Y0), slice(sx, sx + X1 - X0))
    reg = x[Y0:Y1, X0:X1]
    # shadow (offset down)
    shs = np.roll(sh, int(12 * scale), axis=0)[sl]
    reg *= (1 - e.shadow * op * np.clip(shs * 1.15, 0, 1))[..., None]
    reg += (e.glow + glow_boost) * op * gl[sl][..., None] * e.glow_col
    aa = a[sl][..., None] * op
    cc = col[sl] * op
    if sweep is not None:
        xs = np.arange(X1 - X0, dtype=np.float32)[None, :] + sx
        ys = np.arange(Y1 - Y0, dtype=np.float32)[:, None] + sy
        d = xs + ys * 0.45
        pos = lerp(-300, w + h * 0.45 + 300, sweep)
        band = np.exp(-((d - pos) / 55.0) ** 2)
        cc = cc + (band[..., None] * 0.85) * aa * np.array([1, 0.97, 0.88], np.float32)
    reg[:] = reg * (1 - aa) + cc
    return x

def card_elem(w, h, r=34, pad=60):
    m = rrect_mask(w, h, r, pad)
    inner = rrect_mask(w - 6, h - 6, r - 3, pad + 3)
    border = np.clip(m - inner, 0, 1)
    gg = np.broadcast_to(gold_grad(m.shape[0]), m.shape + (3,))
    col = border[..., None] * gg + inner[..., None] * np.array([0.045, 0.028, 0.018], np.float32) * 0.0
    e = Elem(col.astype(np.float32), border, glow=0.30, shadow=0.0)
    e.fill = inner  # glass area
    e.sh = cv2.GaussianBlur(m, (0, 0), 22)
    return e

def comp_card(x, e, cx, cy, op=1.0, dx=0.0, blurk=0.0):
    """frosted dark glass + gold border card"""
    if op <= 0.003: return x
    h, w = e.fill.shape
    x0, y0 = int(cx - w / 2 + dx), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0: return x
    sl = (slice(Y0 - y0, Y1 - y0), slice(X0 - x0, X1 - x0))
    reg = x[Y0:Y1, X0:X1]
    fill = e.fill[sl][..., None] * op
    sh = e.sh[sl][..., None] * op
    frost = cv2.GaussianBlur(reg, (0, 0), 16) * 0.32 + np.array([0.03, 0.017, 0.01], np.float32)
    reg *= (1 - 0.55 * sh)
    reg[:] = reg * (1 - fill * 0.94) + frost * fill * 0.94
    # top inner highlight
    hl = np.exp(-((np.arange(Y1 - Y0)[:, None] + (Y0 - y0) - 60 - 8) / 40.0) ** 2)
    reg += 0.05 * hl[..., None] * fill
    return comp(x, e, cx + dx, cy, op=op, blur=blurk)

# ----------------------------------------------------------------- assets
IMG_FIN = load_img("img_finished.jpg")
IMG_COOK = load_img("img_cooking.jpg")
IMG_PACK = load_img("img_packaging.jpg")

def build_plate():
    """1080x1920 hero plate from the packaging photo: full photo (contain) over its own
    blurred extension so nothing of the real product is cropped."""
    h, w = IMG_PACK.shape[:2]
    s = max(W / w, H / h)
    bg = cv2.resize(IMG_PACK, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    ox = (bg.shape[1] - W) // 2; bg = bg[:H, ox:ox + W].astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), 38) * 0.55
    s2 = W / w
    fg = cv2.resize(IMG_PACK, (W, int(h * s2)), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    top = 205
    m = np.ones(fg.shape[:2], np.float32)
    ramp = 140
    m[:ramp] = np.linspace(0, 1, ramp)[:, None] ** 1.5
    m[-ramp:] = np.linspace(1, 0, ramp)[:, None] ** 1.5
    plate = bg.copy()
    y1 = min(top + fg.shape[0], H)
    plate[top:y1] = bg[top:y1] * (1 - m[:y1 - top, :, None]) + fg[:y1 - top] * m[:y1 - top, :, None]
    return np.clip(plate, 0, 255).astype(np.uint8)
PLATE = build_plate()

# ----------------------------------------------------------------- shots
def ramp_time(lt, segs):
    """integrate piecewise-linear speed curve. segs: [(t, speed), ...]"""
    tt = np.linspace(0, lt, 64); sp = np.interp(tt, [s[0] for s in segs], [s[1] for s in segs])
    return float(np.trapezoid(sp, tt)) if lt > 0 else 0.0

def shot_colander_open(t):  # 0-1.0
    lt = t - 0.0
    src = clip_frame("a_colander", 0.04 + 0.42 * min(max(lt, 0), 1.2))
    z = 1.12 + 0.30 * ease_out_cubic(lt / 1.0)
    img = camera(src, z, 220, 430, rot=lerp(-2.5, 0.5, lt))
    return grade(img, exposure=1.02, crush=0.2, contrast=1.3, sat=1.08, vig=0.7, sharpen=0.7)

def shot_finished_title(t):  # 1-4
    lt = t - 1.0
    z = 1.34 + 0.16 * ease_out_expo(lt / 0.7) + 0.07 * (lt / 3.0)
    img = camera(IMG_FIN, z, 480, 800 - 30 * lt / 3, rot=lerp(1.2, -0.6, lt / 3))
    x = grade(img, exposure=1.0, crush=0.55, contrast=1.3, sat=1.12, vig=0.78, sharpen=0.55)
    return x

def shot_cook_macro(t):  # 4-6
    lt = t - 4.0
    z = 1.9 + 0.22 * ease_io(lt / 2)
    img = camera(IMG_COOK, z, lerp(470, 560, ease_io(lt / 2)), 640, rot=lerp(3, 0.5, lt / 2), persp=(lerp(0.05, -0.03, lt / 2), 0))
    x = grade(img, exposure=1.05, crush=0.4, contrast=1.28, sat=1.12, vig=0.65, sharpen=0.6)
    return tilt_shift(x, H * 0.50, 330, 14)

def shot_stir_slow(t):  # 6-8   true slow motion 0.5x
    lt = t - 6.0
    src = clip_frame("b_stir1", 2.0 + 0.5 * lt)
    z = 1.35 + 0.15 * ease_io(lt / 2)
    img = camera(src, z, 232, 400, rot=lerp(-1.5, 1.0, lt / 2))
    x = grade(img, exposure=0.95, crush=0.62, contrast=1.32, sat=1.15, vig=0.8, sharpen=0.75)
    return tilt_shift(x, H * 0.47, 420, 10)

def shot_fin_macro(t):  # 8-10 rack focus pull-out on chili garnish
    lt = t - 8.0
    z = 2.35 - 0.35 * ease_out_cubic(lt / 2)
    img = camera(IMG_FIN, z, lerp(330, 380, lt / 2), lerp(850, 820, lt / 2), rot=lerp(-3, -1, lt / 2))
    x = grade(img, exposure=1.03, crush=0.45, contrast=1.3, sat=1.14, vig=0.7, sharpen=0.55)
    rack = 10 * (1 - ease_out_cubic(lt / 0.9))
    x = blur_frame(x, rack) if rack > 0.3 else x
    return tilt_shift(x, H * 0.52, 380, 12)

def shot_sambal(t):  # 10-12 speed ramp 2.4x -> 0.45x
    lt = t - 10.0
    st = 3.0 + ramp_time(lt, [(0, 2.4), (0.35, 2.4), (0.85, 0.45), (2.5, 0.45)])
    src = clip_frame("a_sambal", st)
    z = 1.22 + 0.18 * ease_out_cubic(lt / 2)
    img = camera(src, z, 232, 470, rot=lerp(2, -0.5, lt / 2))
    x = grade(img, exposure=0.92, crush=0.35, contrast=1.38, sat=1.15, vig=0.75, sharpen=0.7)
    return x

def shot_stir_ramp(t):  # 12-14
    lt = t - 12.0
    st = 15.0 + ramp_time(lt, [(0, 1.6), (0.4, 1.6), (0.9, 0.5), (2.5, 0.5)])
    src = clip_frame("b_stir2", st)
    z = 1.42 - 0.1 * ease_io(lt / 2)
    img = camera(src, z, 240, 420, rot=lerp(1.5, -1.5, lt / 2), persp=(lerp(-0.04, 0.04, lt / 2), 0))
    x = grade(img, exposure=0.95, crush=0.62, contrast=1.32, sat=1.15, vig=0.8, sharpen=0.75)
    return tilt_shift(x, H * 0.45, 430, 9)

def shot_colander_slow(t):  # 14-16 super slow pull-out
    lt = t - 14.0
    src = clip_frame("b_stir3", 9.0 + 0.5 * lt)
    z = 1.75 - 0.2 * ease_out_cubic(lt / 2)
    img = camera(src, z, 215, 380, rot=lerp(-2, 1.0, lt / 2), persp=(lerp(0.03, -0.02, lt / 2), 0))
    x = grade(img, exposure=0.97, crush=0.62, contrast=1.32, sat=1.15, vig=0.8, sharpen=0.75)
    return tilt_shift(x, H * 0.48, 380, 10)

def shot_packs(t):  # 16-17 real packed stock
    lt = t - 16.0
    src = clip_frame("c_packs", 8.0 + 0.8 * lt)
    z = 1.12 + 0.1 * ease_out_cubic(lt)
    img = camera(src, z, 330, 470, rot=lerp(-1.5, 0, lt))
    x = grade(img, exposure=0.8, crush=0.42, contrast=1.42, sat=1.1, warm=1.8, vig=0.8, sharpen=0.6)
    return x

def shot_plate(t):  # 16-30: product reveal -> defocused price backdrop -> hero
    lt = t - 16.0
    if t < 17.0: z = lerp(1.38, 1.10, ease_out_expo((t - 16.0) / 0.9))
    elif t < 22.4: z = 1.10 + 0.055 * (t - 17.0) / 5.4
    else: z = lerp(1.155, 1.0, ease_io((t - 22.4) / 1.4)) + 0.05 * ease_out_cubic(max(0.0, t - 23.8) / 6.2)
    if t < 16.8: f = 0.0
    elif t < 17.3: f = ease_io((t - 16.8) / 0.5)
    elif t < 22.4: f = 1.0
    else: f = 1 - ease_io((t - 22.4) / 1.0)
    img = camera(PLATE, z, 540 + 22 * math.sin(lt * 0.5), 968, persp=(0.03 * math.sin(lt * 0.55), 0.012))
    x = grade(img, exposure=0.98 - 0.08 * f, crush=0.0, contrast=1.18, sat=1.06, vig=0.62 + 0.18 * f,
              sharpen=0.45 * (1 - f), warm=0.6)
    if f > 0.01:
        x = blur_frame(x, 11 * f) * (1 - 0.38 * f)
    return x

SHOTS = [(0.0, 1.0, shot_colander_open), (1.0, 4.0, shot_finished_title), (4.0, 6.0, shot_cook_macro),
         (6.0, 8.0, shot_stir_slow), (8.0, 10.0, shot_fin_macro), (10.0, 12.0, shot_sambal),
         (12.0, 14.0, shot_stir_ramp), (14.0, 16.0, shot_colander_slow), (16.0, 30.01, shot_plate)]
# cut time -> (type, half-width, dir)
CUTS = {1.0: ("zoom", 0.17, None), 4.0: ("whip", 0.2, (1, 0)), 6.0: ("zoom", 0.17, None),
        8.0: ("whip", 0.2, (0, 1)), 10.0: ("zoom", 0.2, None), 12.0: ("whip", 0.2, (-1, 0)),
        14.0: ("zoom", 0.17, None), 16.0: ("whip", 0.2, (1, 0))}
FLASHES = [(0.0, 0.5, 0.12), (1.0, 0.33, 0.1), (4.0, 0.2, 0.1), (10.0, 0.36, 0.12), (16.0, 0.4, 0.12),
           (17.0, 0.45, 0.12), (17.35, 0.25, 0.1), (18.85, 0.25, 0.1), (23.0, 0.25, 0.2), (28.0, 0.34, 0.22)]
SHAKES = [(0.0, 16), (1.5, 18), (2.0, 20), (10.0, 14), (11.0, 12), (16.0, 10), (17.35, 14), (18.85, 14), (28.0, 16)]

def base_frame(t):
    for T, (kind, hw, d) in CUTS.items():
        if T - hw <= t < T + hw:
            before = t < T
            fn = [s for s in SHOTS if (s[1] == T if before else s[0] == T)][0][2]
            x = fn(t)
            k = (t - (T - hw)) / hw if before else 1 - (t - T) / hw
            k = k * k
            if kind == "zoom": return zoom_blur(x, k)
            return whip(x, k, d[0] * (1 if before else 1), d[1])
    for a, b, fn in SHOTS:
        if a <= t < b: return fn(t)
    return SHOTS[-1][2](t)

# ----------------------------------------------------------------- typography
E = {}
def build_text():
    E["ini"] = text_elem([("INI BARU", "cinzel", 92, "cream", 22)], glow=0.25)
    E["dendeng"] = text_elem([("DENDENG", "anton", 270, "white", 6)], glow=0.30)
    E["padu"] = text_elem([("PADU!", "anton", 300, "gold", 8)], glow=0.55)
    E["rasa1"] = text_elem([("RASA YANG BUAT", "cinzel", 80, "cream", 10)], glow=0.25)
    E["rasa2"] = text_elem([("NAK LAGI!", "anton", 250, "gold", 8)], glow=0.5)
    E["hdr"] = text_elem([("DENDENG PADU", "cinzel", 62, "cream", 16)], glow=0.2)
    E["w300"] = text_elem([("300G", "anton", 120, "white", 4)], glow=0.15)
    E["p27"] = text_elem([("RM27", "anton", 205, "gold", 4)], glow=0.45)
    E["w500"] = text_elem([("500G", "anton", 120, "white", 4)], glow=0.15)
    E["p45"] = text_elem([("RM45", "anton", 205, "gold", 4)], glow=0.45)
    E["card"] = card_elem(900, 330)
    vd = np.zeros((230, 40), np.float32); vd[:, 19:21] = 1
    vd *= np.minimum(1, np.minimum(np.arange(230), 229 - np.arange(230)) / 60.0)[:, None]
    E["vdiv"] = Elem(np.ascontiguousarray(np.broadcast_to(gold_grad(230), (230, 40, 3)) * vd[..., None]), vd, glow=0.4, shadow=0)
    E["fin1"] = text_elem([("DENDENG PADU,", "anton", 150, "white", 4)], glow=0.3)
    E["fin2"] = text_elem([("RASA MEMIKAT!", "cinzel", 92, "gold", 8)], glow=0.45)
    E["finp"] = text_elem([("300G", "anton", 92, "white", 3), ("RM27", "anton", 92, "gold", 3),
                           ("|", "anton", 92, "cream", 0),
                           ("500G", "anton", 92, "white", 3), ("RM45", "anton", 92, "gold", 3)], gap=26, glow=0.3)
    # CTA pill
    t = text_elem([("ORDER SEKARANG!", "anton", 96, "dark", 4)], pad=0)
    pw, ph = t.w + 130, t.h + 64
    pill = rrect_mask(pw, ph, ph // 2, 60)
    gg = np.broadcast_to(gold_grad(pill.shape[0]), pill.shape + (3,))
    col = pill[..., None] * gg
    y0 = (pill.shape[0] - t.h) // 2; x0 = (pill.shape[1] - t.w) // 2
    sub = col[y0:y0 + t.h, x0:x0 + t.w]
    sub[:] = sub * (1 - t.a[..., None]) + t.col
    E["cta"] = Elem(col.astype(np.float32), pill, glow=0.55, shadow=0.6)
    # divider line
    ln = np.zeros((40, 760), np.float32); ln[19:21, :] = 1
    ln *= np.minimum(1, np.minimum(np.arange(760), 759 - np.arange(760)) / 200.0)[None, :]
    E["line"] = Elem(np.broadcast_to(gold_grad(40), (40, 760, 3)) * ln[..., None], ln, glow=0.4, shadow=0)

def scrim(x, y0, y1, amt):
    """soft dark gradient band behind text for legibility"""
    m = sstep(y0 - 240, y0, YY) * (1 - sstep(y1, y1 + 240, YY))
    return x * (1 - amt * m[..., None])

def slam(t, t0, dur=0.28):
    """scale/opacity/blur for an impact slam-in"""
    k = (t - t0) / dur
    if k < 0: return None
    e = ease_out_expo(k)
    return dict(scale=lerp(1.9, 1.0, e), op=min(1, k * 2.2), blur=max(0, 14 * (1 - e)))

def text_layer(x, t):
    # --- scene 1 title 1.1 - 4.0
    if 1.05 <= t < 4.1:
        out = sstep(3.78, 4.0, t)
        x = scrim(x, 700, 1250, 0.55 * sstep(1.0, 1.3, t) * (1 - out))
        k = ease_out_cubic((t - 1.1) / 0.45)
        x = comp(x, E["ini"], W / 2, 735 - 30 * (1 - k), op=k * (1 - out), blur=6 * (1 - k) + 10 * out, scale=1 + 0.15 * out)
        s = slam(t, 1.5)
        if s: x = comp(x, E["dendeng"], W / 2, 925, scale=s["scale"] * (1 + 0.25 * out), op=s["op"] * (1 - out), blur=s["blur"] + 14 * out,
                       sweep=((t - 2.35) / 0.8) if 2.35 < t < 3.15 else None)
        s = slam(t, 2.0)
        if s: x = comp(x, E["padu"], W / 2, 1180, scale=s["scale"] * (1 + 0.25 * out), op=s["op"] * (1 - out), blur=s["blur"] + 14 * out,
                       sweep=((t - 2.5) / 0.8) if 2.5 < t < 3.3 else None, glow_boost=0.4 * math.exp(-(t - 2.0) * 4))
    # --- scene 3 tagline 10.45 - 16
    if 10.4 <= t < 16.05:
        out = sstep(15.72, 15.98, t)
        x = scrim(x, 1180, 1560, 0.5 * sstep(10.4, 10.8, t) * (1 - out))
        k = ease_out_cubic((t - 10.45) / 0.5)
        x = comp(x, E["rasa1"], W / 2, 1215 + 40 * (1 - k), op=k * (1 - out), blur=8 * (1 - k) + 12 * out, reveal=None)
        s = slam(t, 11.0)
        if s: x = comp(x, E["rasa2"], W / 2, 1400, scale=s["scale"] * (1 + 0.2 * out) * (1 + 0.012 * math.sin((t - 11) * 6.283 * 2) * (t < 15.7)),
                       op=s["op"] * (1 - out), blur=s["blur"] + 12 * out,
                       sweep=((t - 12.0) / 0.9) if 12.0 < t < 12.9 else (((t - 14.0) / 0.9) if 14.0 < t < 14.9 else None),
                       glow_boost=0.4 * math.exp(-(t - 11.0) * 4))
    # --- scene 4 price reveal 17 - 23
    if 17.0 <= t < 23.1:
        out = sstep(22.75, 23.0, t)
        k = ease_out_cubic((t - 17.05) / 0.5)
        x = comp(x, E["hdr"], W / 2, 455 - 20 * (1 - k), op=k * (1 - out))
        x = comp(x, E["line"], W / 2, 520, op=k * (1 - out), reveal=None, scale=0.4 + 0.6 * k)
        flo = 6 * math.sin(t * 1.6)
        for (cy, t0, frm, wkey, pkey, sw) in ((800, 17.0, -1, "w300", "p27", (19.4, 21.4)),
                                               (1215, 18.5, 1, "w500", "p45", (19.7, 21.7))):
            kc = ease_out_expo((t - t0) / 0.45) if t >= t0 else 0
            if kc <= 0: continue
            dx = frm * 1100 * (1 - kc)
            x = comp_card(x, E["card"], W / 2, cy + flo, op=min(1, kc * 1.5) * (1 - out), dx=dx, blurk=10 * (1 - kc))
            kw = ease_out_cubic((t - t0 - 0.12) / 0.35)
            cw1, cw2 = E[wkey].w - 180, E[pkey].w - 180
            free = 900 - cw1 - cw2
            lx = W / 2 - 450 + free / 3 + cw1 / 2; rx = W / 2 + 450 - free / 3 - cw2 / 2
            dvx = (lx + cw1 / 2 + rx - cw2 / 2) / 2
            x = comp(x, E["vdiv"], dvx + dx, cy + flo, op=max(0, kw) * (1 - out))
            x = comp(x, E[wkey], lx + dx, cy + flo + 6, op=max(0, kw) * (1 - out), blur=8 * (1 - max(kw, 0)))
            s = slam(t, t0 + 0.35)
            if s:
                swp = None
                for s0 in sw:
                    if s0 < t < s0 + 0.85: swp = (t - s0) / 0.85
                x = comp(x, E[pkey], rx + dx, cy + flo, scale=s["scale"], op=s["op"] * (1 - out), blur=s["blur"],
                         sweep=swp, glow_boost=0.5 * math.exp(-(t - t0 - 0.35) * 4))
    # --- scene 5 hero 23.3 - 30
    if t >= 23.3:
        x = scrim(x, 150, 360, 0.55 * sstep(23.3, 23.7, t))
        x = scrim(x, 1500, 1760, 0.75 * sstep(24.4, 24.9, t))
        k = ease_out_cubic((t - 23.35) / 0.5)
        pulse_t = t - 28.0
        gb = 0.5 * math.exp(-pulse_t * 3) if pulse_t > 0 else 0
        x = comp(x, E["fin1"], W / 2, 205 + 40 * (1 - k), op=k, blur=10 * (1 - k), glow_boost=gb,
                 sweep=((t - 26.2) / 0.9) if 26.2 < t < 27.1 else None)
        s = slam(t, 23.75, 0.3)
        if s: x = comp(x, E["fin2"], W / 2, 330, scale=s["scale"] * 0.9 + 0.1, op=s["op"], blur=s["blur"],
                       glow_boost=gb + 0.4 * math.exp(-(t - 23.75) * 4),
                       sweep=((t - 26.5) / 0.9) if 26.5 < t < 27.4 else None)
        kp = ease_out_cubic((t - 24.75) / 0.45)
        if kp > 0: x = comp(x, E["finp"], W / 2, 1565 + 30 * (1 - kp), op=kp, blur=8 * (1 - kp), glow_boost=gb,
                            sweep=((t - 27.0) / 0.9) if 27.0 < t < 27.9 else None)
        s = slam(t, 25.5, 0.3)
        if s:
            beat = (t - 25.5) % 0.5
            pulse = 1 + 0.035 * math.exp(-beat * 9) if t > 25.9 else 1
            x = comp(x, E["cta"], W / 2, 1705, scale=s["scale"] * 0.85 * pulse + 0.15 * pulse - 0.0, op=s["op"], blur=s["blur"],
                     glow_boost=gb + 0.25 * math.exp(-beat * 6) * (t > 25.9),
                     sweep=((t - 26.0) / 0.8) if 26.0 < t < 26.8 else (((t - 28.1) / 0.8) if 28.1 < t < 28.9 else None))
    return x

# ----------------------------------------------------------------- frame
def particles_intensity(t):
    if t < 1.0: return 0.35
    if t < 4.0: return 0.95
    if t < 10.0: return 0.45
    if t < 16.0: return 0.55
    if t < 17.0: return 0.25
    if t < 23.0: return 0.75
    return 0.85

def render_frame(fi):
    t = fi / FPS
    x = base_frame(t)
    if 10.0 <= t < 16.0:
        x = spices(x, t, sstep(10.0, 10.3, t) * (1 - sstep(15.8, 16.0, t)), 10.0)
    if 1.0 <= t < 4.0:
        x = spices(x, t, 0.6 * sstep(1.0, 1.4, t) * (1 - sstep(3.8, 4.0, t)), 1.0)
    x = embers(x, t, particles_intensity(t))
    x = light_sweep(x, t, 1.05, 0.9, amt=0.35)
    x = light_sweep(x, t, 23.15, 1.3, amt=0.75, width=180)
    x = light_sweep(x, t, 26.9, 1.2, amt=0.5, width=160)
    x = bloom(x)
    x = text_layer(x, t)
    for t0, amt, dur in FLASHES:
        if t0 <= t < t0 + dur * 3:
            x = x + amt * math.exp(-(t - t0) / dur) * np.array([1.0, 0.86, 0.62], np.float32)
    # camera shake on impacts
    dxs = dys = 0.0
    for t0, a in SHAKES:
        if t0 <= t < t0 + 0.35:
            d = math.exp(-(t - t0) * 12) * a
            dxs += d * math.sin((t - t0) * 90); dys += d * math.cos((t - t0) * 77)
    if abs(dxs) + abs(dys) > 0.5:
        s = 1.02
        M = np.array([[s, 0, dxs - (s - 1) * W / 2], [0, s, dys - (s - 1) * H / 2]], np.float32)
        x = cv2.warpAffine(x, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    if t < 0.12: x = x + 0.3 * (1 - t / 0.12) * np.array([1.0, 0.86, 0.62], np.float32)  # punch in from flash
    x = grain(x, fi)
    x = x * (1 - sstep(29.55, 30.0, t))
    return (clamp01(x) * 255 + 0.5).astype(np.uint8)

build_text()

if MODE == "preview":
    os.makedirs(f"{WORK}/preview", exist_ok=True)
    for ts in sys.argv[4:]:
        fi = int(round(float(ts) * FPS))
        cv2.imwrite(f"{WORK}/preview/f_{float(ts):05.2f}.jpg", cv2.cvtColor(render_frame(fi), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("ok")
elif MODE == "chunk":
    a, b, out = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "10",
                          "-pix_fmt", "yuv444p", out], stdin=subprocess.PIPE)
    for fi in range(a, b):
        p.stdin.write(render_frame(fi).tobytes())
    p.stdin.close(); p.wait()
elif MODE == "render":
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    step = math.ceil(NF / nw); procs = []; parts = []
    for i in range(nw):
        a, b = i * step, min(NF, (i + 1) * step)
        out = f"{WORK}/chunk_{i}.mkv"; parts.append(out)
        procs.append(subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), SRC, WORK, "chunk", str(a), str(b), out]))
    for p in procs: p.wait()
    with open(f"{WORK}/chunks.txt", "w") as f:
        for o in parts: f.write(f"file '{o}'\n")
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{WORK}/chunks.txt",
                           "-c", "copy", f"{WORK}/video_master.mkv"])
    print("rendered", f"{WORK}/video_master.mkv")
