# -*- coding: utf-8 -*-
"""
SnapBench: image perturbation generator (sev 1/3/5)
====================================================
Apply 15 perturbation types × 3 severity levels to 1,145 query images.
Output: bench_images/perturbed/{type}/sev{1,3,5}/{query_id}.jpg
Total: 1,145 × 15 × 3 = 51,675 images (not shipped; generate locally).

Usage:
    python benchmark/gen_image_perturbations.py [--benchmark BENCHMARK_JSON] [--dry-run]
"""

import os, sys, io, json, math, random, argparse, datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT   = os.path.join(BASE, "bench_images", "perturbed")
BENCHMARK  = os.path.join(BASE, "benchmark", "snap_bench.json")


def query_image_dir():
    root = os.environ.get("BENCH_IMAGES_DIR", os.path.join(BASE, "bench_images"))
    return os.path.join(root, "query")


def resolve_query_image(record):
    fname = record.get("query_image_local", "")
    if not fname:
        raise ValueError(f"missing query_image_local for {record.get('query_id')}")
    if os.path.isabs(fname):
        return fname
    return os.path.join(query_image_dir(), fname)

SEVERITY_LEVELS = [1, 3, 5]

# ── Image conversion helpers ────────────────────────────────────────────────
def p2c(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def c2p(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY PARAMETER TABLE
# From apply_perturbations.py, taking sev 1/3/5 three levels
# ═══════════════════════════════════════════════════════════════════════════════

SEV_PARAMS = {
    "defocus_blur": {
        1: {"sigma": 1.0},
        3: {"sigma": 3.5},
        5: {"sigma": 10.0},
    },
    "motion_blur": {
        1: {"ks": 5},
        3: {"ks": 13},
        5: {"ks": 25},
    },
    "low_light": {
        1: {"brightness": 0.70, "noise_std": 3},
        3: {"brightness": 0.40, "noise_std": 10},
        5: {"brightness": 0.15, "noise_std": 22},
    },
    "overexposure": {
        1: {"factor": 1.4},
        3: {"factor": 2.2},
        5: {"factor": 3.0},
    },
    "low_resolution": {
        1: {"scale": 0.50},
        3: {"scale": 0.25},
        5: {"scale": 0.08},
    },
    "compression": {
        1: {"quality": 50},
        3: {"quality": 18},
        5: {"quality": 5},
    },
    "rotation": {
        1: {"angle": 10},
        3: {"angle": 45},
        5: {"angle": 180},
    },
    "perspective": {
        1: {"d_ratio": 0.10},
        3: {"d_ratio": 0.20},
        5: {"d_ratio": 0.30},
    },
    "lens_distortion": {
        1: {"k1": -0.10},
        3: {"k1": -0.20},
        5: {"k1": -0.30},
    },
    "cropping": {
        1: {"keep_ratio": 0.90},
        3: {"keep_ratio": 0.65},
        5: {"keep_ratio": 0.40},
    },
    "downscale": {
        1: {"scale": 0.20},
        3: {"scale": 0.10},
        5: {"scale": 0.04},
    },
    "watermark": {
        1: {"opacity": 80,  "mode": "single"},      # 1 watermark text
        3: {"opacity": 160, "mode": "half_tile"},    # tiled 50% area
        5: {"opacity": 220, "mode": "full_tile"},    # full image tiled
    },
    "mosaic": {
        1: {"bs": 42, "n_blocks": 1, "block_ratio": 0.20},
        3: {"bs": 42, "n_blocks": 2, "block_ratio": 0.35},
        5: {"bs": 42, "n_blocks": 4, "block_ratio": 0.55},
    },
    "scribble": {
        1: {"n_lines": 3, "thickness": 2},
        3: {"n_lines": 6, "thickness": 4},
        5: {"n_lines": 9, "thickness": 5},
    },
    "ui_elements": {
        1: {"elements": ["top_bar"]},                       # top bar only ~5%
        3: {"elements": ["top_bar", "bottom_bar"]},         # top + bottom
        5: {"elements": ["top_bar", "bottom_bar", "address_bar", "fab_button"]},  # full UI
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 15 IMAGE PERTURBATION FUNCTIONS (parameterized by severity)
# ═══════════════════════════════════════════════════════════════════════════════

def perturb_low_light(img, sev=5, **kw):
    params = SEV_PARAMS["low_light"][sev]
    arr = np.array(img).astype(np.float32) * params["brightness"]
    noise = np.random.randn(*arr.shape) * params["noise_std"]
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def perturb_overexposure(img, sev=5, **kw):
    params = SEV_PARAMS["overexposure"][sev]
    return Image.fromarray(np.clip(np.array(img).astype(np.float32) * params["factor"], 0, 255).astype(np.uint8))

def perturb_defocus_blur(img, sev=5, **kw):
    params = SEV_PARAMS["defocus_blur"][sev]
    return img.filter(ImageFilter.GaussianBlur(radius=params["sigma"]))

def perturb_motion_blur(img, sev=5, **kw):
    params = SEV_PARAMS["motion_blur"][sev]
    ks = params["ks"]
    arr = p2c(img)
    k = np.zeros((ks, ks))
    k[ks // 2, :] = 1.0 / ks
    return c2p(cv2.filter2D(arr, -1, k))

def perturb_compression(img, sev=5, **kw):
    params = SEV_PARAMS["compression"][sev]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=params["quality"])
    buf.seek(0)
    return Image.open(buf).copy()

def perturb_low_resolution(img, sev=5, **kw):
    params = SEV_PARAMS["low_resolution"][sev]
    w, h = img.size
    sw, sh = max(4, int(w * params["scale"])), max(4, int(h * params["scale"]))
    return img.resize((sw, sh), Image.NEAREST).resize((w, h), Image.NEAREST)

def perturb_rotation(img, sev=5, **kw):
    params = SEV_PARAMS["rotation"][sev]
    angle = kw.get("angle", params["angle"])
    w, h = img.size
    if angle >= 85:
        return img.rotate(angle, resample=Image.BICUBIC, expand=True).resize((w, h), Image.LANCZOS)
    rad = math.radians(abs(angle) % 90)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    scale = max(cos_a + (h / w) * sin_a, cos_a + (w / h) * sin_a) + 0.02
    ew, eh = int(w * scale) + 4, int(h * scale) + 4
    enlarged = img.resize((ew, eh), Image.LANCZOS)
    rotated = enlarged.rotate(angle, resample=Image.BICUBIC, expand=False)
    cx, cy = ew // 2, eh // 2
    return rotated.crop((cx - w // 2, cy - h // 2, cx - w // 2 + w, cy - h // 2 + h))

def perturb_perspective(img, sev=5, **kw):
    params = SEV_PARAMS["perspective"][sev]
    arr = p2c(img)
    h, w = arr.shape[:2]
    d = int(min(w, h) * params["d_ratio"])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, d // 2], [w - d, d // 2], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mx, my = d + 2, d // 2 + 2
    cropped = warped[my:h, mx:w - mx]
    if cropped.shape[0] < 8 or cropped.shape[1] < 8:
        cropped = warped
    return c2p(cv2.resize(cropped, (w, h)))

def perturb_lens_distortion(img, sev=5, **kw):
    params = SEV_PARAMS["lens_distortion"][sev]
    arr = p2c(img)
    h, w = arr.shape[:2]
    k1 = params["k1"]
    cx, cy = w / 2.0, h / 2.0
    y_c, x_c = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (x_c - cx) / cx, (y_c - cy) / cy
    factor = 1.0 + k1 * (xn ** 2 + yn ** 2)
    map_x = (xn * factor * cx + cx).astype(np.float32)
    map_y = (yn * factor * cy + cy).astype(np.float32)
    dist = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cf = min(0.18, abs(k1) * 0.12)
    mx, my = max(2, int(w * cf)), max(2, int(h * cf))
    cropped = dist[my:h - my, mx:w - mx]
    return c2p(cv2.resize(cropped, (w, h)))

def perturb_cropping(img, sev=5, **kw):
    params = SEV_PARAMS["cropping"][sev]
    keep_ratio = kw.get("keep_ratio", params["keep_ratio"])
    w, h = img.size
    cw, ch = int(w * keep_ratio), int(h * keep_ratio)
    x, y = random.randint(0, max(0, w - cw)), random.randint(0, max(0, h - ch))
    return img.crop((x, y, x + cw, y + ch))

def perturb_downscale(img, sev=5, **kw):
    params = SEV_PARAMS["downscale"][sev]
    w, h = img.size
    return img.resize((max(4, int(w * params["scale"])), max(4, int(h * params["scale"]))), Image.LANCZOS)

def perturb_watermark(img, sev=5, **kw):
    params = SEV_PARAMS["watermark"][sev]
    result = img.copy().convert("RGBA")
    w, h = result.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    opacity = params["opacity"]
    mode = params["mode"]

    fs = max(16, int(w * 0.16))
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", fs)
    except:
        font = ImageFont.load_default()

    if mode == "single":
        # single centered watermark
        tx, ty = w // 4, h // 2 - fs // 2
        draw.text((tx, ty), "SAMPLE", font=font, fill=(180, 180, 180, opacity))
    elif mode == "half_tile":
        # tile ~50% of area
        sp = max(1, int(w * 0.32))
        y_start = h // 4
        y_end = 3 * h // 4
        for ys in range(y_start, y_end, sp):
            for xs in range(-w // 4, w, sp):
                draw.text((xs, ys), "SAMPLE", font=font, fill=(180, 180, 180, opacity))
    else:  # full_tile
        sp = max(1, int(w * 0.16))
        for ys in range(-h // 2, h, sp):
            for xs in range(-w // 2, w, sp):
                draw.text((xs, ys), "SAMPLE", font=font, fill=(180, 180, 180, opacity))

    return Image.alpha_composite(result, overlay).convert("RGB")

def perturb_mosaic(img, sev=5, **kw):
    params = SEV_PARAMS["mosaic"][sev]
    result = img.copy()
    w, h = img.size
    n_blocks = params["n_blocks"]
    block_ratio = params["block_ratio"]
    bs = params["bs"]

    cx_lo, cx_hi = int(w * 0.25), int(w * 0.75)
    cy_lo, cy_hi = int(h * 0.25), int(h * 0.75)

    for _ in range(n_blocks):
        rw, rh = max(80, int(w * block_ratio)), max(80, int(h * block_ratio))
        rw, rh = min(rw, w), min(rh, h)
        rx, ry = 0, 0
        for _ in range(40):
            rx = random.randint(0, max(0, w - rw))
            ry = random.randint(0, max(0, h - rh))
            if rx + rw <= cx_lo or rx >= cx_hi or ry + rh <= cy_lo or ry >= cy_hi:
                break
        region = result.crop((rx, ry, rx + rw, ry + rh))
        small = region.resize((max(1, rw // bs), max(1, rh // bs)), Image.NEAREST)
        result.paste(small.resize((rw, rh), Image.NEAREST), (rx, ry))
    return result

def perturb_scribble(img, sev=5, **kw):
    params = SEV_PARAMS["scribble"][sev]
    arr = p2c(img)
    h, w = arr.shape[:2]
    n_lines = params["n_lines"]
    thickness = params["thickness"]

    COLORS = [(0, 0, 0), (220, 30, 30), (30, 30, 220), (30, 200, 30), (200, 150, 0)]
    for _ in range(n_lines):
        color = random.choice(COLORS)
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        pts = [(x, y)]
        for _ in range(random.randint(4, 10)):
            x = max(0, min(w - 1, x + random.randint(-w // 5, w // 5)))
            y = max(0, min(h - 1, y + random.randint(-h // 5, h // 5)))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(arr, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)
    return c2p(arr)

def perturb_ui_elements(img, sev=5, **kw):
    params = SEV_PARAMS["ui_elements"][sev]
    elements = params["elements"]

    result = img.copy().convert("RGBA")
    w, h = result.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        f_main = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", max(10, int(h * 0.035)))
    except:
        f_main = ImageFont.load_default()
    try:
        f_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", max(9, int(h * 0.025)))
    except:
        f_small = f_main

    if "top_bar" in elements:
        bt = int(h * 0.05)  # sev1: ~5% height
        if sev >= 3:
            bt = int(h * 0.09)
        draw.rectangle([0, 0, w, bt], fill=(30, 30, 30, 245))
        draw.text((8, 2), "9:41", font=f_main, fill=(255, 255, 255, 230))
        draw.text((w - 60, 2), "100%\u25a0", font=f_main, fill=(255, 255, 255, 230))

    if "bottom_bar" in elements:
        bb = int(h * 0.08)
        draw.rectangle([0, h - bb, w, h], fill=(30, 30, 30, 245))
        for i, item in enumerate(["\u25c1", "\u25cb", "\u25a1"]):
            draw.text(((i + 1) * w // 4 - 6, h - bb + 4), item, font=f_main, fill=(255, 255, 255, 200))

    if "address_bar" in elements:
        bt = int(h * 0.09) if sev >= 3 else int(h * 0.05)
        ab_h = int(h * 0.05)
        draw.rectangle([0, bt, w, bt + ab_h], fill=(240, 240, 240, 210))
        draw.text((8, bt + 4), "www.example.com/photo", font=f_small, fill=(80, 80, 80, 200))

    if "fab_button" in elements:
        bx, by, br = w - 50, h // 2, 22
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(66, 133, 244, 200))
        draw.text((bx - 6, by - 8), "+", font=f_main, fill=(255, 255, 255, 230))

    return Image.alpha_composite(result, overlay).convert("RGB")


# ── Perturbation registry ────────────────────────────────────────────────────
PERTURB_FNS = {
    "low_light":       perturb_low_light,
    "overexposure":    perturb_overexposure,
    "defocus_blur":    perturb_defocus_blur,
    "motion_blur":     perturb_motion_blur,
    "compression":     perturb_compression,
    "low_resolution":  perturb_low_resolution,
    "rotation":        perturb_rotation,
    "perspective":     perturb_perspective,
    "lens_distortion": perturb_lens_distortion,
    "cropping":        perturb_cropping,
    "downscale":       perturb_downscale,
    "watermark":       perturb_watermark,
    "mosaic":          perturb_mosaic,
    "scribble":        perturb_scribble,
    "ui_elements":     perturb_ui_elements,
}


def run(benchmark_json: str, dry_run: bool = False):
    data = json.load(open(benchmark_json, encoding="utf-8"))
    print(f"Loaded {len(data)} queries from {benchmark_json}")
    print(f"Severity levels: {SEVERITY_LEVELS}")
    print(f"Total images to generate: {len(data)} × {len(PERTURB_FNS)} × {len(SEVERITY_LEVELS)} = {len(data) * len(PERTURB_FNS) * len(SEVERITY_LEVELS)}")

    # Create output directories: perturbed/{type}/sev{1,3,5}/
    for pt_name in PERTURB_FNS:
        for sev in SEVERITY_LEVELS:
            out_dir = os.path.join(OUT_ROOT, pt_name, f"sev{sev}")
            os.makedirs(out_dir, exist_ok=True)

    total = len(data) * len(PERTURB_FNS) * len(SEVERITY_LEVELS)
    done = 0
    skipped = 0
    errors = 0

    for record in tqdm(data, desc="Queries"):
        qid = record["query_id"]
        img_path = resolve_query_image(record)

        # Extract per-perturbation random params from benchmark
        img_perturbs = record.get("image_perturbations", {})

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ERROR loading {img_path}: {e}")
            errors += len(PERTURB_FNS) * len(SEVERITY_LEVELS)
            continue

        for pt_name, fn in PERTURB_FNS.items():
            for sev in SEVERITY_LEVELS:
                out_path = os.path.join(OUT_ROOT, pt_name, f"sev{sev}", f"{qid}.jpg")

                # Skip if already exists
                if os.path.exists(out_path):
                    skipped += 1
                    done += 1
                    continue

                if dry_run:
                    done += 1
                    continue

                # Get per-query random params from benchmark record (for rotation/cropping)
                pt_params = img_perturbs.get(pt_name, {})
                kwargs = {"sev": sev}
                # For rotation, use per-query angle from benchmark if available
                if "angle" in pt_params and pt_name == "rotation":
                    # Scale angle proportionally to severity
                    base_angle = pt_params["angle"]
                    sev_angle = SEV_PARAMS["rotation"][sev]["angle"]
                    # For sev1/3, use severity-defined angle; for sev5 (180°), keep 180
                    kwargs["angle"] = sev_angle

                # Set numpy/random seed for reproducibility
                seed = (hash(qid + pt_name) + sev * 7919) & 0xFFFFFFFF
                random.seed(seed)
                np.random.seed(seed)

                try:
                    perturbed = fn(img, **kwargs)
                    perturbed.save(out_path, format="JPEG", quality=92)
                    done += 1
                except Exception as e:
                    print(f"  ERROR {pt_name}/sev{sev} on {qid}: {e}")
                    errors += 1
                    done += 1

    print(f"\nDone: {done}/{total} | Skipped (existing): {skipped} | Errors: {errors}")
    if not dry_run:
        # Count generated files
        total_files = 0
        for pt in PERTURB_FNS:
            for sev in SEVERITY_LEVELS:
                sev_dir = os.path.join(OUT_ROOT, pt, f"sev{sev}")
                if os.path.isdir(sev_dir):
                    total_files += len(os.listdir(sev_dir))
        print(f"Total files in bench_images/perturbed/*/sev*/: {total_files}")
        print(f"Expected: {len(data) * len(PERTURB_FNS) * len(SEVERITY_LEVELS)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate perturbed images for SnapBench (sev 1/3/5)")
    parser.add_argument("--benchmark", default=BENCHMARK,
                        help="Path to benchmark JSON (default: latest)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just count, don't generate")
    args = parser.parse_args()

    run(args.benchmark, dry_run=args.dry_run)
