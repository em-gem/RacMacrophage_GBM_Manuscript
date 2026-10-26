# INPUT_DIRS = [
#     "C:/Users/980 Analysis/Documents/AlyseGBMCellPose/F480_Data/Fixed F480 D7 ADCPs/MAXs/08192026",
#     "C:/Users/980 Analysis/Documents/AlyseGBMCellPose/F480_Data/Fixed F480 D7 ADCPs/MAXs/08272026"
# ]

# OUTPUT_ROOT = "C:/Users/980 Analysis/Documents/AlyseGBMCellPose/F480_Data/V1_FirstRunthroughFullDataset"

# -*- coding: utf-8 -*-
"""
Manual-Macrophage-Mask Pipeline: Macrophages vs Glioblastoma
=============================================================
Macrophage masks come from ONE hand-tuned CellPose GUI mask per field of view,
drawn on EITHER the F4/80 image OR the CellTracker Green image:

    <base>_C3_seg.npy  (F4/80)              -> used as the macrophage masks
    <base>_C1_seg.npy  (CellTracker Green)  -> used as the macrophage masks
    <base>_C2.tif      (DAPI)               -> nuclei (manual _seg.npy if present,
                                               otherwise auto-segmented)

File suffixes are matched at the END of the file stem (case-insensitive),
e.g. "Well3_Pos2_C3.tif" + "Well3_Pos2_C3_seg.npy". The CellPose GUI writes
<image name>_seg.npy next to the image when you save, so no renaming is needed.

If both a C3 and a C1 mask exist for the same field, MAC_MASK_PRIORITY decides
which one is used (a warning is printed). If neither exists, IF_NO_MANUAL_MAC
decides what happens (skip the field, or auto-segment F4/80).

Manual macrophage masks are used exactly as drawn: no size or signal filters.

For every field of view:
  1. Macrophages = objects in the chosen manual mask.
  2. A nucleus is inside a macrophage if >= NUC_OVERLAP_FRAC of its area lies
     inside the macrophage masks.
       glioblastoma cells = nuclei NOT inside macrophage masks
       (every nucleus of a multinucleated macrophage is excluded)
  3. Nuclei per macrophage (multinucleation).

Outputs (in OUTPUT_ROOT):
  per_image_summary_<ts>.csv   one row per field; mac_mask_source says which
                               channel the macrophage mask came from
  per_cell_detail_<ts>.csv     one row per macrophage
  per_nucleus_detail_<ts>.csv  one row per DAPI nucleus
  run_settings_<ts>.txt        every parameter used
  <folder>/{dapi,macrophage}/  overlays / outlines (/ masks)
  <folder>/classification/     composite + classified-nuclei figure

Usage:
    python cellpose_manual_macrophage_gbm.py
    python cellpose_manual_macrophage_gbm.py --max_fields 3
    python cellpose_manual_macrophage_gbm.py --name_filter Scene-22
"""

import argparse, re, sys, warnings
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

INPUT_DIRS = [
    "C:/Users/980 Analysis/Documents/AlyseGBMCellPose/F480_Data/Fixed_F480_D7_ADCPs/MAXs/08272026",
    "C:/Users/980 Analysis/Documents/AlyseGBMCellPose/F480_Data/Fixed_F480_D7_ADCPs/MAXs/08192026",
    "C:/Users/980 Analysis/Documents/AlyseGBMCellPose/F480_Data/Fixed_F480_D7_ADCPs/MAXs/10022026"
]

OUTPUT_ROOT = "C:/Users/980 Analysis/Documents/AlyseGBMCellPose/F480_Data/V7_FullData"

PIXEL_SIZE_UM = 1.3811        # um per pixel (area conversion = this squared)

# Quick-test options (None = process everything)
MAX_FIELDS_PER_FOLDER = None
NAME_FILTER           = None

# ── Channels and file suffixes ─────────────────────────────────
CHANNEL_SUFFIX = {
    "ctg":  "C1",     # CellTracker Green
    "dapi": "C2",     # DAPI
    "f480": "C3",     # F4/80
}
SKIP_SUFFIXES = ["C4"]          # far red: ignored without a warning
CHANNEL_LABEL = {"ctg": "CellTracker Green", "dapi": "DAPI", "f480": "F4/80"}

# ── Macrophage mask source ─────────────────────────────────────
MAC_MASK_PRIORITY = ["f480", "ctg"]   # channels that can supply macrophage masks
WHEN_BOTH_MASKS   = "newest"          # if BOTH have a _seg.npy:
                                      #   "newest"   -> the most recently saved file wins
                                      #   "priority" -> first channel in MAC_MASK_PRIORITY wins
IF_NO_MANUAL_MAC  = "skip"            # "skip"      -> field left out of the counts
                                      # "auto_f480" -> auto-segment F4/80 (filters applied)

# ── DAPI ───────────────────────────────────────────────────────
DAPI_USE_MANUAL = True        # use <base>_C2_seg.npy when present, else auto-segment

# ── Background / normalisation (DAPI auto-segmentation + display) ──
BG_BLOCK_PX    = 128
BG_PERCENTILE  = 10
NORM_UPPER_PCT = 99.9
MIN_RANGE_SD   = 15

# ── CellPose settings for AUTO segmentation only ───────────────
#    Diameters in PIXELS. Copy the values that worked for you in the GUI.
#    Cellpose 4.x (Cellpose-SAM) ignores model_type.
#    MODELS: either {"pretrained_model": <name>} (Cellpose 4.x built-ins such as
#    "cpdino", "cpdino-vitb", "cpsam_v2", "cpsam") or {"model_type": <name>}
#    (Cellpose 3.x names such as "nuclei", "cyto3").
MODELS = {
    "dapi": {"pretrained_model": "cpdino"},   # CellposeDINO - best for your DAPI
    "f480": {"model_type": "cyto3"},          # fallback only (Cellpose 4 uses its default)
}
#    What image CellPose sees:
#      "raw"       -> the raw image, CellPose normalises it itself (same as the GUI)
#      "bgsub"     -> this script's background-subtracted 0-1 image (normalize=False)
SEG_INPUT = {"dapi": "raw", "f480": "bgsub"}
SEG_PARAMS = {
    "dapi": {
        "diameter":            None,   # <-- put your GUI DAPI diameter here (None = model default)
        "flow_threshold":      0.4,    # <-- and your GUI flow threshold
        "cellprob_threshold":  0.0,    # <-- and your GUI cellprob threshold
        "min_size":            15,
        "augment":             False,
        "resample":            True,
        "batch_size":          8,
    },
    "f480": {                          # only used when IF_NO_MANUAL_MAC = "auto_f480"
        "diameter":            40.0,   # your tuned GUI settings
        "flow_threshold":      0.8,
        "cellprob_threshold":  0.0,
        "min_size":            30,
        "channels":            [0, 0],
        "augment":             True,
        "resample":            True,
        "batch_size":          8,
    },
}

# ── Post-segmentation filters (AUTO masks only; manual masks are kept as drawn) ──
FILTERS = {
    # DAPI signal-to-noise filter is OFF: in dense fields it was removing real nuclei
    "dapi": {"min_area_um2": 20,  "max_area_um2": 1500, "min_snr": None, "min_solidity": None},
    "f480": {"min_area_um2": 100, "max_area_um2": 7000, "min_snr": 3.0,  "min_solidity": None},
}

# ── Macrophage counting ────────────────────────────────────────
#    Masks with no nucleus are treated as over-segmentation fragments or dead
#    cells / debris and are NOT counted as macrophages. They still count as
#    macrophage area when deciding whether a nucleus is inside a macrophage, so
#    a nucleus split across a cell and its fragment stays a macrophage nucleus.
REQUIRE_NUCLEUS = True

# ── Classification threshold ───────────────────────────────────
NUC_OVERLAP_FRAC = 0.50    # nucleus inside a macrophage when >= this fraction
                           # of its area lies inside the macrophage masks

# ── CellTracker check on nuclei ────────────────────────────────
#    F4/80 outlines (membrane) can sprawl over neighbouring GBM cells. Every
#    macrophage carries CellTracker Green, so a nucleus that sits inside a
#    macrophage mask but has no CellTracker signal over it is counted as a GBM
#    cell instead. Value = mean background-subtracted, normalised CellTracker
#    intensity over the nucleus (the "mean_ctg" column of per_nucleus_detail).
#    In your data: macrophage nuclei >= ~0.09 (10th pct), GBM nuclei <= ~0.03 (90th pct).
#    OFF by default: nuclei inside a macrophage mask always stay with the
#    macrophage (e.g. engulfed GBM cells). This can never overcount either cell
#    type - each macrophage is counted once, each nucleus at most once.
CTG_CHECK            = False
CTG_CHECK_THRESHOLD  = 0.05
CTG_CHECK_MASK_TYPES = ["f480"]   # apply to fields whose macrophage mask came from these
                                  # channels (CellTracker masks already are CellTracker+)

# ── Outputs ────────────────────────────────────────────────────
SAVE_MASKS          = False
SAVE_OVERLAYS       = True
SAVE_OUTLINES       = True
SAVE_CLASSIFICATION = True
SAVE_SETTINGS       = True
OVERLAY_ALPHA       = 0.4
COLORMAP            = "tab20"

NUC_CLASSES     = ["MAC", "GBM", "GBM_CTGNEG"]
NUC_CLASS_LABEL = {"MAC": "nucleus in macrophage",
                   "GBM": "nucleus outside macrophages (GBM)",
                   "GBM_CTGNEG": "inside mask but CellTracker-neg (GBM)"}
NUC_CLASS_COLOR = {"MAC": (1.00, 1.00, 0.00),        # yellow
                   "GBM": (0.00, 0.85, 1.00),        # cyan
                   "GBM_CTGNEG": (1.00, 0.20, 1.00)} # magenta
MAC_OUTLINE_COLOR = {"f480": (1.00, 0.15, 0.15),  # red when drawn on F4/80
                     "ctg":  (0.10, 1.00, 0.10)}  # green when drawn on CellTracker
EXCLUDED_OUTLINE_COLOR = (0.55, 0.55, 0.55)       # grey: nucleus-free masks (not counted)

PX_TO_UM2 = None

# ═══════════════════════════════════════════════════════════════
#  IMAGE LOADING / BACKGROUND / NORMALISATION
# ═══════════════════════════════════════════════════════════════

def load_raw(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[0] <= 4:
        img = img[0]
    elif img.ndim == 3:
        img = img[:, :, 0]
    return img.squeeze().astype(np.float32)


def estimate_background(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    b = int(max(8, min(BG_BLOCK_PX, h, w)))
    H, W = -(-h // b) * b, -(-w // b) * b
    pad = np.pad(img, ((0, H - h), (0, W - w)), mode="edge")
    tiles = pad.reshape(H // b, b, W // b, b).transpose(0, 2, 1, 3).reshape(H // b, W // b, -1)
    grid = np.percentile(tiles, BG_PERCENTILE, axis=2)
    if min(grid.shape) >= 3:
        grid = ndi.median_filter(grid, size=3, mode="nearest")
    bg = ndi.zoom(grid, b, order=1, mode="nearest")[:h, :w]
    return ndi.gaussian_filter(bg, sigma=b / 4)


def prepare_channel(raw: np.ndarray) -> dict:
    bg    = estimate_background(raw)
    resid = raw - bg
    p10, p30 = np.percentile(resid, [10, 30])
    noise = (p30 - p10) / 0.7572
    if not noise > 0:
        low = resid[resid <= np.percentile(resid, 50)]
        noise = float(low.std()) if low.size else 0.0
    rng   = float(np.percentile(raw, 99.9) - np.percentile(raw, 1))
    noise = max(float(noise), 0.005 * rng, 1e-6)
    resid = resid - (p10 + 1.2816 * noise)
    p_hi  = float(np.percentile(resid, NORM_UPPER_PCT))
    scale = max(p_hi, MIN_RANGE_SD * noise)
    norm  = np.clip(resid / scale, 0, 1).astype(np.float32)
    return {"norm": norm, "resid": resid.astype(np.float32), "noise_sd": noise,
            "bg_median": float(np.median(bg)), "signal_snr": round(p_hi / noise, 2)}


def seg_path_for(tif_path: Path) -> Path:
    return tif_path.parent / (tif_path.stem + "_seg.npy")


def load_manual_mask(tif_path: Path):
    seg_path = seg_path_for(tif_path)
    if not seg_path.exists():
        return None
    try:
        seg   = np.load(str(seg_path), allow_pickle=True).item()
        masks = seg.get("masks", seg.get("mask", None))
        if masks is None:
            print(f"    [warn] {seg_path.name}: no 'masks' key found")
            return None
        return np.asarray(masks).astype(np.int32)
    except Exception as e:
        print(f"    [warn] Could not load {seg_path.name}: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
#  FILE GROUPING
# ═══════════════════════════════════════════════════════════════

def _suffix_match(stem, suffix):
    return re.match(rf"^(.*?)[ _\-]*{re.escape(suffix)}$", stem, flags=re.IGNORECASE)


def find_fields(folder: Path) -> dict:
    """Group TIFs into {base_name: {"ctg": Path, "dapi": Path, "f480": Path}}."""
    fields, ignored = {}, []
    for p in sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))):
        matched = False
        for channel, suffix in CHANNEL_SUFFIX.items():
            m = _suffix_match(p.stem, suffix)
            if m:
                fields.setdefault(m.group(1).rstrip(" _-"), {})[channel] = p
                matched = True
                break
        if not matched and not any(_suffix_match(p.stem, s) for s in SKIP_SUFFIXES):
            ignored.append(p.name)
    if ignored:
        print(f"  [note] {len(ignored)} file(s) ignored (no "
              f"{'/'.join(CHANNEL_SUFFIX.values())} suffix), e.g. {ignored[0]}")
    fields = dict(sorted(fields.items()))
    if NAME_FILTER:
        fields = {k: v for k, v in fields.items() if NAME_FILTER.lower() in k.lower()}
    if MAX_FIELDS_PER_FOLDER:
        fields = dict(list(fields.items())[:int(MAX_FIELDS_PER_FOLDER)])
    return fields

# ═══════════════════════════════════════════════════════════════
#  LABEL HELPERS
# ═══════════════════════════════════════════════════════════════

def labels_of(mask):
    lbl = np.unique(mask)
    return lbl[lbl > 0]


def label_means(mask, values, labels):
    if len(labels) == 0:
        return np.array([], dtype=float)
    return np.atleast_1d(ndi.mean(values, labels=mask, index=labels)).astype(float)


def majority_overlap(obj_mask, cell_mask):
    sel = (obj_mask > 0) & (cell_mask > 0)
    if not sel.any():
        return {}
    df = pd.DataFrame({"o": obj_mask[sel], "c": cell_mask[sel]})
    counts = df.value_counts().reset_index(name="px")
    best = counts.sort_values("px", ascending=False).drop_duplicates("o")
    return dict(zip(best["o"].astype(int), best["c"].astype(int)))


def outline(mask):
    return find_boundaries(mask, mode="inner")

# ═══════════════════════════════════════════════════════════════
#  MEASUREMENT / FILTERING
# ═══════════════════════════════════════════════════════════════

def measure_and_filter(prep, mask, filt, apply_filters=True):
    """Returns (relabelled_mask, stats). Manual masks keep every object."""
    mask   = np.asarray(mask).astype(np.int32)
    labels = labels_of(mask)
    n      = len(labels)

    areas_px = np.bincount(mask.ravel())[labels] if n else np.array([], dtype=np.int64)
    areas_um = areas_px * PX_TO_UM2
    means    = label_means(mask, prep["norm"], labels)
    snr      = label_means(mask, prep["resid"], labels) / prep["noise_sd"]
    solidity = np.full(n, np.nan)
    if apply_filters and filt.get("min_solidity") is not None and n:
        sol = {rp.label: rp.solidity for rp in regionprops(mask)}
        solidity = np.array([sol.get(int(l), np.nan) for l in labels])

    rm_size = rm_snr = rm_shape = np.zeros(n, dtype=bool)
    if apply_filters and n:
        rm_size = np.zeros(n, dtype=bool)
        if filt.get("min_area_um2") is not None:
            rm_size |= areas_um < filt["min_area_um2"]
        if filt.get("max_area_um2") is not None:
            rm_size |= areas_um > filt["max_area_um2"]
        rm_snr = ~rm_size & (snr < filt["min_snr"]) if filt.get("min_snr") is not None \
                 else np.zeros(n, dtype=bool)
        rm_shape = ~rm_size & ~rm_snr & (solidity < filt["min_solidity"]) \
                   if filt.get("min_solidity") is not None else np.zeros(n, dtype=bool)
    keep = ~(rm_size | rm_snr | rm_shape)

    kept = labels[keep]
    lut = np.zeros(int(mask.max()) + 1, dtype=np.int32)
    lut[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)
    new_mask = lut[mask]

    new_labels = np.arange(1, len(kept) + 1)
    cents = ndi.center_of_mass(new_mask > 0, new_mask, new_labels) if len(kept) else []

    records = []
    for i, j in enumerate(np.flatnonzero(keep)):
        cy, cx = cents[i]
        records.append({
            "cell_label":     i + 1,
            "centroid_y":     round(float(cy), 1),
            "centroid_x":     round(float(cx), 1),
            "area_px2":       int(areas_px[j]),
            "area_um2":       round(float(areas_um[j]), 2),
            "eq_diam_px":     round(float(2 * np.sqrt(areas_px[j] / np.pi)), 1),
            "snr":            round(float(snr[j]), 2),
            "mean_intensity": round(float(means[j]), 4),
        })

    return new_mask, {"n_removed_size": int(rm_size.sum()), "n_removed_snr": int(rm_snr.sum()),
                      "n_removed_shape": int(rm_shape.sum()), "cell_records": records}


def summarise_area(cell_records: list) -> dict:
    if not cell_records:
        return {"total_area_um2": 0.0, "mean_area_um2": 0.0, "median_diam_px": np.nan}
    um2s = [r["area_um2"] for r in cell_records]
    return {
        "total_area_um2": round(sum(um2s), 2),
        "mean_area_um2":  round(sum(um2s) / len(um2s), 2),
        "median_diam_px": round(float(np.median([r["eq_diam_px"] for r in cell_records])), 1),
    }

# ═══════════════════════════════════════════════════════════════
#  MODEL (only needed for auto segmentation)
# ═══════════════════════════════════════════════════════════════

_MODELS = {}

def _gpu_available():
    try:
        from cellpose import core; return core.use_gpu()
    except Exception:
        return False


def _cellpose_version():
    try:
        import cellpose
        return getattr(cellpose, "version", None) or getattr(cellpose, "__version__", "unknown")
    except Exception:
        return "not installed"


def _model_name(key):
    spec = MODELS[key]
    return spec.get("pretrained_model") or spec.get("model_type") or "default"


def get_model(key):
    name = _model_name(key)
    if name not in _MODELS:
        from cellpose import models as cp_models
        gpu  = _gpu_available()
        spec = MODELS[key]
        if "pretrained_model" in spec:
            try:
                _MODELS[name] = cp_models.CellposeModel(gpu=gpu, pretrained_model=spec["pretrained_model"])
            except Exception as e:
                raise RuntimeError(
                    f"Could not load CellPose model '{name}' (cellpose {_cellpose_version()}). "
                    f"Built-in models like cpdino need a recent Cellpose 4 - in the same Python "
                    f"environment VS Code uses, run:  pip install -U cellpose   ({e})") from e
        else:
            try:
                _MODELS[name] = cp_models.CellposeModel(model_type=spec["model_type"], gpu=gpu)
            except TypeError:
                _MODELS[name] = cp_models.CellposeModel(gpu=gpu)
        print(f"[model] loaded '{name}' for {key}  |  GPU={gpu}  |  cellpose {_cellpose_version()}")
    return _MODELS[name]


def auto_segment(raw, prep, key):
    """Run CellPose on either the raw image (CellPose normalises, like the GUI)
    or the background-subtracted image (normalize=False)."""
    model  = get_model(key)
    params = SEG_PARAMS[key].copy()
    if SEG_INPUT.get(key, "bgsub") == "raw":
        img, params["normalize"] = raw, True
    else:
        img, params["normalize"] = prep["norm"], False
    try:
        return model.eval(img, **params)[0]
    except TypeError:
        # older/newer CellPose versions differ in which keywords eval() accepts
        for k in ("channels", "resample", "min_size", "augment"):
            params.pop(k, None)
        return model.eval(img, **params)[0]

# ═══════════════════════════════════════════════════════════════
#  PER-FIELD PROCESSING
# ═══════════════════════════════════════════════════════════════

def get_dapi(files):
    path = files["dapi"]
    raw  = load_raw(path)
    prep = prepare_channel(raw)
    masks, source = None, "auto"
    if DAPI_USE_MANUAL:
        masks = load_manual_mask(path)
        if masks is not None:
            source = "manual"
            if masks.shape != prep["norm"].shape:
                print(f"    [warn] {seg_path_for(path).name} shape differs from image; re-segmenting")
                masks, source = None, "auto"
    if masks is None:
        masks = auto_segment(raw, prep, "dapi")
        source = f"auto {_model_name('dapi')}"
    mask, fstats = measure_and_filter(prep, masks, FILTERS["dapi"], apply_filters=source.startswith("auto"))
    return {"prep": prep, "img": prep["norm"], "mask": mask, "fstats": fstats,
            "source": source, "filename": path.name}


def get_macrophages(files):
    """Pick the manual macrophage mask (F4/80 or CellTracker), or fall back."""
    available = [ch for ch in MAC_MASK_PRIORITY
                 if files.get(ch) is not None and seg_path_for(files[ch]).exists()]
    if len(available) > 1:
        if WHEN_BOTH_MASKS == "newest":
            available.sort(key=lambda c: seg_path_for(files[c]).stat().st_mtime, reverse=True)
            why = "most recently saved"
        else:
            why = "MAC_MASK_PRIORITY"
        print(f"    [note] manual masks found for BOTH "
              f"{' and '.join(CHANNEL_LABEL[c] for c in available)}; "
              f"using {CHANNEL_LABEL[available[0]]} ({why})")

    for ch in available:
        path  = files[ch]
        prep  = prepare_channel(load_raw(path))
        masks = load_manual_mask(path)
        if masks is None:
            continue
        if masks.shape != prep["norm"].shape:
            print(f"    [warn] {seg_path_for(path).name} shape {masks.shape} != image "
                  f"{prep['norm'].shape}; skipping this mask")
            continue
        mask, fstats = measure_and_filter(prep, masks, {}, apply_filters=False)
        return {"prep": prep, "img": prep["norm"], "mask": mask, "fstats": fstats,
                "channel": ch, "source": f"manual {CHANNEL_LABEL[ch]}",
                "filename": seg_path_for(path).name, "also_found": available[1:]}

    if IF_NO_MANUAL_MAC == "auto_f480" and files.get("f480") is not None:
        path = files["f480"]
        raw  = load_raw(path)
        prep = prepare_channel(raw)
        mask, fstats = measure_and_filter(prep, auto_segment(raw, prep, "f480"),
                                          FILTERS["f480"], apply_filters=True)
        return {"prep": prep, "img": prep["norm"], "mask": mask, "fstats": fstats,
                "channel": "f480", "source": "auto F4/80", "filename": path.name,
                "also_found": []}
    return None


def classify_field(dapi, mac, marker_imgs):
    """Returns (summary_dict, per_cell_rows, per_nucleus_rows, plot_info)."""
    nm, mm = dapi["mask"], mac["mask"]
    mac_bin = (mm > 0).astype(np.float32)
    m_lbls, n_lbls = labels_of(mm), labels_of(nm)

    n_frac   = label_means(nm, mac_bin, n_lbls)
    n_geom   = n_frac >= NUC_OVERLAP_FRAC          # inside by position only
    ctg_img  = marker_imgs.get("ctg")
    n_ctg    = label_means(nm, ctg_img, n_lbls) if ctg_img is not None else np.full(len(n_lbls), np.nan)
    check_on = (CTG_CHECK and ctg_img is not None and mac["channel"] in CTG_CHECK_MASK_TYPES)
    ctg_neg  = n_geom & (n_ctg < CTG_CHECK_THRESHOLD) if check_on else np.zeros(len(n_lbls), bool)
    n_in     = n_geom & ~ctg_neg
    nuc_cls  = np.where(n_in, "MAC", np.where(ctg_neg, "GBM_CTGNEG", "GBM"))
    nuc_to_mac = majority_overlap(nm, mm)

    per = Counter(nuc_to_mac.get(int(l), 0) for l, ins in zip(n_lbls, n_in) if ins)
    per.pop(0, None)
    m_nnuc = np.array([per.get(int(l), 0) for l in m_lbls], dtype=int)
    counted = (m_nnuc >= 1) if REQUIRE_NUCLEUS else np.ones(len(m_lbls), dtype=bool)
    cnt_nnuc = m_nnuc[counted]

    n_mac, n_nuc = int(counted.sum()), len(n_lbls)
    n_in_mac     = int(n_in.sum())
    n_gbm        = n_nuc - n_in_mac
    excl_area    = [r["area_um2"] for r, c in
                    zip(sorted(mac["fstats"]["cell_records"], key=lambda r: r["cell_label"]), counted)
                    if not c]

    summary = {
        "n_macrophages":                  n_mac,
        "n_total_nuclei":                 n_nuc,
        "n_nuclei_in_macrophages":        n_in_mac,
        "n_GBM_cells_nuclei_outside_mac": n_gbm,
        "pct_cells_macrophages":          round(100.0 * n_mac / (n_mac + n_gbm), 2) if (n_mac + n_gbm) else np.nan,
        "ratio_GBM_to_macrophage":        round(n_gbm / n_mac, 3) if n_mac else np.nan,
        "mean_nuclei_per_macrophage":     round(float(cnt_nnuc.mean()), 3) if n_mac else np.nan,
        "n_macrophages_multinucleated":   int((cnt_nnuc >= 2).sum()),
        "ctg_check_applied":              bool(check_on),
        "n_nuclei_reassigned_ctg_neg":    int(ctg_neg.sum()),
        "n_GBM_without_ctg_check":        int(n_nuc - n_geom.sum()),
        "n_mac_masks_total":              len(m_lbls),
        "n_mac_masks_excluded_no_nucleus": int((~counted).sum()),
        "pct_mac_masks_excluded":         round(100.0 * (~counted).sum() / len(m_lbls), 1) if len(m_lbls) else np.nan,
        "median_area_excluded_um2":       round(float(np.median(excl_area)), 1) if excl_area else np.nan,
    }

    cell_rows = []
    rec = {r["cell_label"]: r for r in mac["fstats"]["cell_records"]}
    m_means = {c: label_means(mm, im, m_lbls) for c, im in marker_imgs.items()}
    for i, l in enumerate(m_lbls):
        r = rec.get(int(l), {})
        cell_rows.append({
            "cell_label": int(l),
            "centroid_y": r.get("centroid_y"), "centroid_x": r.get("centroid_x"),
            "area_px2": r.get("area_px2"), "area_um2": r.get("area_um2"),
            "eq_diam_px": r.get("eq_diam_px"),
            "n_nuclei": int(m_nnuc[i]),
            "counted_as_macrophage": bool(counted[i]),
            **{f"mean_{c}": round(float(v[i]), 4) for c, v in m_means.items()},
        })

    nuc_rows = []
    n_rec   = {r["cell_label"]: r for r in dapi["fstats"]["cell_records"]}
    n_means = {c: label_means(nm, im, n_lbls) for c, im in marker_imgs.items()}
    for i, l in enumerate(n_lbls):
        r = n_rec.get(int(l), {})
        nuc_rows.append({
            "nucleus_label": int(l),
            "class": NUC_CLASS_LABEL[nuc_cls[i]],
            "counted_as_GBM": bool(not n_in[i]),
            "centroid_y": r.get("centroid_y"), "centroid_x": r.get("centroid_x"),
            "area_px2": r.get("area_px2"), "area_um2": r.get("area_um2"),
            "snr": r.get("snr"),
            "frac_in_macrophage": round(float(n_frac[i]), 3),
            "reassigned_ctg_neg": bool(ctg_neg[i]),
            "macrophage_label": nuc_to_mac.get(int(l), 0) if n_in[i] else 0,
            **{f"mean_{c}": round(float(v[i]), 4) for c, v in n_means.items()},
        })

    return summary, cell_rows, nuc_rows, {"n_lbls": n_lbls, "nuc_class": nuc_cls,
                                          "excluded_lbls": m_lbls[~counted]}

# ═══════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════

def save_overlay_png(img, mask, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Background-subtracted"); axes[0].axis("off")
    cmap    = plt.get_cmap(COLORMAP)
    colored = cmap(mask % cmap.N).astype(float)
    colored[mask == 0] = 0
    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(colored, alpha=OVERLAY_ALPHA)
    axes[1].set_title(title, fontsize=9); axes[1].axis("off")
    plt.tight_layout(); fig.savefig(str(out_path), bbox_inches="tight"); plt.close(fig)


def save_outlines_png(img, mask, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ov = np.zeros((*img.shape, 4), dtype=float)
    ov[outline(mask)] = [1, 0.2, 0.2, 0.9]
    ax.imshow(ov)
    ax.set_title(title, fontsize=9); ax.axis("off")
    plt.tight_layout(); fig.savefig(str(out_path), bbox_inches="tight"); plt.close(fig)


def save_classification_png(base, dapi, mac, marker_imgs, summary, plot_info, out_path):
    """Left: composite (R = F4/80, G = CellTracker, B = DAPI).
       Right: nuclei filled by class + macrophage outlines."""
    shape = dapi["img"].shape
    zero  = np.zeros(shape, dtype=np.float32)
    rgb   = np.dstack([marker_imgs.get("f480", zero), marker_imgs.get("ctg", zero),
                       dapi["img"]]).clip(0, 1)

    nm, mm = dapi["mask"], mac["mask"]
    cls_img = rgb * 0.45
    lut = np.zeros(int(nm.max()) + 1, dtype=np.int8)
    for l, c in zip(plot_info["n_lbls"], plot_info["nuc_class"]):
        lut[l] = NUC_CLASSES.index(c) + 1
    codes = lut[nm]
    for i, c in enumerate(NUC_CLASSES, start=1):
        sel = codes == i
        cls_img[sel] = 0.3 * cls_img[sel] + 0.7 * np.array(NUC_CLASS_COLOR[c])
    out_color = MAC_OUTLINE_COLOR[mac["channel"]]
    excl = np.isin(mm, plot_info["excluded_lbls"])
    edges = outline(mm)
    cls_img[edges & ~excl] = out_color
    cls_img[edges & excl]  = EXCLUDED_OUTLINE_COLOR

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), dpi=150)
    axes[0].imshow(rgb); axes[0].axis("off")
    axes[0].set_title("Composite  (R = F4/80, G = CellTracker Green, B = DAPI)", fontsize=9)
    axes[1].imshow(cls_img); axes[1].axis("off")
    s = summary
    axes[1].set_title(
        f"macrophages {s['n_macrophages']} [{mac['source']}] | "
        f"multinucleated {s['n_macrophages_multinucleated']} | "
        f"excluded (no nucleus) {s['n_mac_masks_excluded_no_nucleus']}\n"
        f"nuclei {s['n_total_nuclei']} | in macrophages {s['n_nuclei_in_macrophages']} | "
        f"GBM {s['n_GBM_cells_nuclei_outside_mac']}"
        + (f" (incl. {s['n_nuclei_reassigned_ctg_neg']} CellTracker-neg under mask)"
           if s["ctg_check_applied"] else ""),
        fontsize=9)
    handles = [Patch(color=out_color, label=f"macrophage outline ({mac['source']})"),
               Patch(color=EXCLUDED_OUTLINE_COLOR, label="excluded: no nucleus")]
    shown = [c for c in NUC_CLASSES if c != "GBM_CTGNEG" or s["ctg_check_applied"]]
    handles += [Patch(color=NUC_CLASS_COLOR[c], label=NUC_CLASS_LABEL[c]) for c in shown]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(base, fontsize=10)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(str(out_path), bbox_inches="tight"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════
#  RUN SETTINGS LOG
# ═══════════════════════════════════════════════════════════════

def save_run_settings(output_root: Path, timestamp: str, input_dirs) -> Path:
    def fmt(v):
        return "None (disabled)" if v is None else v

    L = []; A = L.append
    A("=" * 70)
    A(" MANUAL-MACROPHAGE-MASK / GBM CLASSIFICATION - RUN SETTINGS")
    A("=" * 70)
    A(f"Run timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"File tag        : {timestamp}")
    A(f"Script          : {Path(__file__).name if '__file__' in globals() else 'interactive'}")
    A(f"CellPose version: {_cellpose_version()}")
    A(f"Python          : {sys.version.split()[0]}")
    A("")
    A("-" * 70); A(" PATHS"); A("-" * 70)
    for d in input_dirs:
        A(f"  input  : {d}{'' if Path(d).exists() else '   [NOT FOUND]'}")
    A(f"  output : {output_root}")
    if NAME_FILTER or MAX_FIELDS_PER_FOLDER:
        A(f"  TEST MODE: name_filter={NAME_FILTER}  max_fields_per_folder={MAX_FIELDS_PER_FOLDER}")
    A("")
    A("-" * 70); A(" CHANNELS / MASK SOURCES"); A("-" * 70)
    for ch, sfx in CHANNEL_SUFFIX.items():
        A(f"  {ch:<5} ({CHANNEL_LABEL[ch]:<17}) : files ending in {sfx}.tif")
    A(f"  skipped suffixes   : {', '.join(SKIP_SUFFIXES)}")
    A(f"  macrophage masks   : manual _seg.npy on "
      f"{' or '.join(CHANNEL_LABEL[c] for c in MAC_MASK_PRIORITY)} "
      f"(priority if both: {CHANNEL_LABEL[MAC_MASK_PRIORITY[0]]}); used as drawn, no filters")
    A(f"  if no manual mask  : {IF_NO_MANUAL_MAC}")
    A(f"  DAPI manual masks  : {DAPI_USE_MANUAL} (otherwise auto-segmented)")
    A(f"  pixel_size_um      : {PIXEL_SIZE_UM} um/px   (um2_per_px2 = {PX_TO_UM2})")
    A("")
    A("-" * 70); A(" AUTO SEGMENTATION (DAPI without manual mask"
                  + (", F4/80 fallback" if IF_NO_MANUAL_MAC == "auto_f480" else "") + ")"); A("-" * 70)
    A(f"  background: block {BG_BLOCK_PX} px, percentile {BG_PERCENTILE}, "
      f"upper pct {NORM_UPPER_PCT}, min range {MIN_RANGE_SD} SD")
    for ch, params in SEG_PARAMS.items():
        A(f"  [{ch}]  model={_model_name(ch)}  input={SEG_INPUT.get(ch, 'bgsub')}")
        for k, v in params.items():
            A(f"      {k:<20} : {fmt(v)}")
        for k, v in FILTERS[ch].items():
            A(f"      filter {k:<13} : {fmt(v)}")
        A("")
    A("-" * 70); A(" CLASSIFICATION"); A("-" * 70)
    A(f"  nuc_overlap_frac : {NUC_OVERLAP_FRAC}")
    A(f"  require_nucleus  : {REQUIRE_NUCLEUS}")
    A(f"  ctg_check        : {CTG_CHECK}  threshold {CTG_CHECK_THRESHOLD}  "
      f"applied to masks from: {', '.join(CHANNEL_LABEL[c] for c in CTG_CHECK_MASK_TYPES)}")
    A("      nucleus inside a macrophage mask but mean CellTracker < threshold -> GBM")
    A(f"  when_both_masks  : {WHEN_BOTH_MASKS}")
    A("  macrophages = objects in the manual macrophage mask"
      + (" containing >= 1 nucleus (nucleus-free masks excluded from counts,"
         " but still treated as macrophage area for nucleus classification)"
         if REQUIRE_NUCLEUS else ""))
    A("  GBM cells   = nuclei with < nuc_overlap_frac of their area inside macrophage masks")
    A("=" * 70)

    out = Path(output_root); out.mkdir(parents=True, exist_ok=True)
    out = out / f"run_settings_{timestamp}.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    return out

# ═══════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════

HEADLINE_COLS = [
    "n_macrophages", "n_total_nuclei", "n_nuclei_in_macrophages",
    "n_GBM_cells_nuclei_outside_mac", "n_macrophages_multinucleated",
    "n_mac_masks_excluded_no_nucleus", "n_nuclei_reassigned_ctg_neg",
]


def run_pipeline(input_dirs, output_root):
    global PX_TO_UM2
    PX_TO_UM2 = (PIXEL_SIZE_UM ** 2) if PIXEL_SIZE_UM else 1.0

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if SAVE_SETTINGS:
        print(f"[settings] {save_run_settings(output_root, ts, input_dirs)}")

    rows, cell_rows, nuc_rows, skipped = [], [], [], []

    for folder in input_dirs:
        folder = Path(folder)
        if not folder.exists():
            print(f"[skip] Not found: {folder}"); continue
        fields = find_fields(folder)
        if not fields:
            print(f"[skip] No matching TIFs in {folder}"); continue

        out_folder = output_root / folder.name
        for sub in ("dapi", "macrophage"):
            for kind in ("masks", "overlays", "outlines"):
                (out_folder / sub / kind).mkdir(parents=True, exist_ok=True)
        if SAVE_CLASSIFICATION:
            (out_folder / "classification").mkdir(parents=True, exist_ok=True)

        print(f"\n{'-'*78}\n Folder : {folder.name}  ({len(fields)} field(s) of view)\n{'-'*78}")
        counts = Counter()

        for base, files in fields.items():
            print(f"  {base}")
            row = {"folder": folder.name, "base_name": base}
            try:
                if files.get("dapi") is None:
                    print(f"    [skip] missing {CHANNEL_SUFFIX['dapi']} (DAPI) file")
                    skipped.append((folder.name, base, "no DAPI image")); continue
                mac = get_macrophages(files)
                if mac is None:
                    print("    [skip] no manual macrophage mask (_seg.npy on "
                          f"{' or '.join(CHANNEL_SUFFIX[c] for c in MAC_MASK_PRIORITY)})")
                    skipped.append((folder.name, base, "no manual macrophage mask")); continue
                dapi = get_dapi(files)
                if dapi["mask"].shape != mac["mask"].shape:
                    print("    [skip] DAPI and macrophage images differ in size")
                    skipped.append((folder.name, base, "image size mismatch")); continue
            except Exception:
                import traceback; traceback.print_exc()
                skipped.append((folder.name, base, "error")); continue

            counts[mac["source"]] += 1
            # marker images for the composite and per-cell mean intensities
            marker_imgs = {"dapi": dapi["img"], mac["channel"]: mac["img"]}
            for ch in ("ctg", "f480"):
                if ch not in marker_imgs and files.get(ch) is not None:
                    try:
                        im = prepare_channel(load_raw(files[ch]))["norm"]
                        if im.shape == dapi["img"].shape:
                            marker_imgs[ch] = im
                    except Exception:
                        pass

            summary, c_rows, n_rows, plot_info = classify_field(dapi, mac, marker_imgs)
            row.update(summary)
            row.update({
                "mac_mask_source":      mac["source"],
                "mac_mask_file":        mac["filename"],
                "mac_other_mask_found": ", ".join(CHANNEL_LABEL[c] for c in mac["also_found"]),
                "dapi_source":          dapi["source"],
                "dapi_filename":        dapi["filename"],
                "dapi_n_removed_size":  dapi["fstats"]["n_removed_size"],
                "dapi_n_removed_snr":   dapi["fstats"]["n_removed_snr"],
                "mac_total_area_um2":   summarise_area(mac["fstats"]["cell_records"])["total_area_um2"],
                "mac_mean_area_um2":    summarise_area(mac["fstats"]["cell_records"])["mean_area_um2"],
                "pixel_size_um":        PIXEL_SIZE_UM,
                "nuc_overlap_frac":     NUC_OVERLAP_FRAC,
            })
            rows.append(row)
            meta = {"folder": folder.name, "base_name": base, "mac_mask_source": mac["source"]}
            cell_rows += [{**meta, **r} for r in c_rows]
            nuc_rows  += [{**meta, **r} for r in n_rows]

            print(f"    macrophages {summary['n_macrophages']:>5} [{mac['source']}, "
                  f"{summary['n_mac_masks_excluded_no_nucleus']} nucleus-free masks excluded] | "
                  f"nuclei {summary['n_total_nuclei']} [{dapi['source']}"
                  + ("" if dapi["source"] == "manual" else
                     f", removed size {dapi['fstats']['n_removed_size']} "
                     f"low-SNR {dapi['fstats']['n_removed_snr']}") + "]")
            print(f"    -> in macrophages {summary['n_nuclei_in_macrophages']} | "
                  f"GBM {summary['n_GBM_cells_nuclei_outside_mac']}"
                  + (f" (incl. {summary['n_nuclei_reassigned_ctg_neg']} CellTracker-neg under mask)"
                     if summary["ctg_check_applied"] else " (CellTracker check off)") + " | "
                  f"multinucleated {summary['n_macrophages_multinucleated']}")

            stem = files["dapi"].stem
            mstem = base + "_macrophage"
            if SAVE_MASKS:
                tifffile.imwrite(str(out_folder / "dapi" / "masks" / f"{stem}_mask.tif"),
                                 dapi["mask"].astype(np.uint16), compression="zlib")
                tifffile.imwrite(str(out_folder / "macrophage" / "masks" / f"{mstem}_mask.tif"),
                                 mac["mask"].astype(np.uint16), compression="zlib")
            if SAVE_OVERLAYS:
                save_overlay_png(dapi["img"], dapi["mask"],
                                 f"{summary['n_total_nuclei']} nuclei [{dapi['source']}]",
                                 out_folder / "dapi" / "overlays" / f"{stem}_overlay.png")
                save_overlay_png(mac["img"], mac["mask"],
                                 f"{summary['n_macrophages']} macrophages [{mac['source']}]",
                                 out_folder / "macrophage" / "overlays" / f"{mstem}_overlay.png")
            if SAVE_OUTLINES:
                save_outlines_png(dapi["img"], dapi["mask"], f"{summary['n_total_nuclei']} nuclei",
                                  out_folder / "dapi" / "outlines" / f"{stem}_outlines.png")
                save_outlines_png(mac["img"], mac["mask"], f"{summary['n_macrophages']} macrophages",
                                  out_folder / "macrophage" / "outlines" / f"{mstem}_outlines.png")
            if SAVE_CLASSIFICATION:
                try:
                    save_classification_png(base, dapi, mac, marker_imgs, summary, plot_info,
                                            out_folder / "classification" / f"{base}_classification.png")
                except Exception as e:
                    print(f"    [classification figure failed: {e}]")

        print(f"  -> macrophage mask sources in this folder: "
              + (", ".join(f"{k}: {v}" for k, v in counts.items()) or "none"))

    df_img  = pd.DataFrame(rows)
    df_cell = pd.DataFrame(cell_rows)
    df_nuc  = pd.DataFrame(nuc_rows)
    csv_img  = output_root / f"per_image_summary_{ts}.csv"
    csv_cell = output_root / f"per_cell_detail_{ts}.csv"
    csv_nuc  = output_root / f"per_nucleus_detail_{ts}.csv"
    df_img.to_csv(csv_img,  index=False, encoding="utf-8-sig")
    df_cell.to_csv(csv_cell, index=False, encoding="utf-8-sig")
    df_nuc.to_csv(csv_nuc,  index=False, encoding="utf-8-sig")
    if skipped:
        csv_skip = output_root / f"skipped_fields_{ts}.csv"
        pd.DataFrame(skipped, columns=["folder", "base_name", "reason"]).to_csv(
            csv_skip, index=False, encoding="utf-8-sig")

    print(f"\n{'='*78}\n SUMMARY (sum per folder)\n{'='*78}")
    if not df_img.empty:
        cols = [c for c in HEADLINE_COLS if c in df_img.columns]
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(df_img.groupby("folder")[cols].sum(min_count=1).to_string())
    if skipped:
        print(f"\n{len(skipped)} field(s) skipped -> {csv_skip}")
    print(f"\nPer-image CSV   -> {csv_img}")
    print(f"Per-cell CSV    -> {csv_cell}")
    print(f"Per-nucleus CSV -> {csv_nuc}")
    print(f"All outputs     -> {output_root}")
    return df_img, df_cell, df_nuc

# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs",    nargs="+", default=None)
    p.add_argument("--output_root",   default=None)
    p.add_argument("--pixel_size_um", type=float, default=None)
    p.add_argument("--nuc_overlap",   type=float, default=None)
    p.add_argument("--max_fields",    type=int,   default=None)
    p.add_argument("--name_filter",   default=None)
    p.add_argument("--prefer",        choices=["f480", "ctg"], default=None,
                   help="which manual mask wins if both exist")
    p.add_argument("--fallback",      choices=["skip", "auto_f480"], default=None,
                   help="what to do when a field has no manual macrophage mask")
    p.add_argument("--no_settings", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input_dirs    is not None: INPUT_DIRS[:] = args.input_dirs
    if args.output_root   is not None: OUTPUT_ROOT   = args.output_root
    if args.pixel_size_um is not None: PIXEL_SIZE_UM = args.pixel_size_um
    if args.nuc_overlap   is not None: NUC_OVERLAP_FRAC = args.nuc_overlap
    if args.max_fields    is not None: MAX_FIELDS_PER_FOLDER = args.max_fields
    if args.name_filter   is not None: NAME_FILTER = args.name_filter
    if args.prefer        is not None:
        MAC_MASK_PRIORITY = [args.prefer] + [c for c in ("f480", "ctg") if c != args.prefer]
        WHEN_BOTH_MASKS   = "priority"
    if args.fallback      is not None: IF_NO_MANUAL_MAC = args.fallback
    if args.no_settings: SAVE_SETTINGS = False

    run_pipeline(INPUT_DIRS, OUTPUT_ROOT)