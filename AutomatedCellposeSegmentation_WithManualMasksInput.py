
# -*- coding: utf-8 -*-
"""
Dual-Channel CellPose Segmentation Pipeline
===========================================
Targets     = files ending in C4.tif
Macrophages = files ending in C1.tif

Changes from the adaptive (density-tier) version:
  - No density classification / SPARSE-MEDIUM-DENSE tiers. One fixed
    parameter set per channel instead.
  - Images are matched into pairs by their base name, so each field of
    view produces ONE row containing both target and macrophage results.
  - Areas reported in both px^2 and um^2 for both channels.
  - No automated clump splitting (oversized objects are simply removed
    by the max-area filter, or fixed by hand in the CellPose GUI).
  - Manual masks (_seg.npy) are used exactly as saved: CellPose is not
    re-run on that file, and no filters are applied to it.
  - Every run writes run_settings_<timestamp>.txt next to the CSVs
    recording the exact parameters used.

Usage:
    python cellpose_dual_channel.py
    python cellpose_dual_channel.py --input_dirs D:/data/run1 --output_root D:/out
"""

import argparse, re, sys, warnings
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from skimage.measure import regionprops

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

INPUT_DIRS = [] # List folders here

OUTPUT_ROOT =  # List output folder here

MODEL_TYPE    = "cyto3"
USE_SAM       = True
MANUAL_MODE   = True          # prefer hand-corrected _seg.npy when present
PIXEL_SIZE_UM = 1.3811        # um per pixel (area conversion = this squared)

# ── Which filename suffix belongs to which channel ─────────────
#    Matched case-insensitively against the end of the file stem,
#    e.g. "Well3_Pos2_C4.tif" -> base "Well3_Pos2", channel "target".
CHANNEL_SUFFIX = {
    "target":     "C4",
    "macrophage": "C1",
}

# ── Segmentation settings, per channel ─────────────────────────
SEG_PARAMS = {
    "target": {
        "diameter":           60.0,
        "flow_threshold":      1.0,
        "cellprob_threshold": -3.0,
        "min_size":           30,
        "channels":          [0, 0],
        "normalize":          True,
        "augment":            False,
        "resample":           True,
        "batch_size":         8,
    },
    "macrophage": {
        "diameter":            70.0,
        "flow_threshold":      1.0,
        "cellprob_threshold":  0.0,
        "min_size":           30,
        "channels":          [0, 0],
        "normalize":          True,
        "augment":            True,
        "resample":           True,
        "batch_size":         8,
    },
}

# ── Post-segmentation filters, per channel ─────────────────────
#    Areas are in um^2 (i.e. the same units as the reported area_um2).
#    Set any value to None to disable that filter.
FILTERS = {
    "target": {
        "min_area_um2":   100,
        "max_area_um2":   7000,
        "min_brightness": 0.7,     # cell mean intensity / whole-image mean
    },
    "macrophage": {
        "min_area_um2":   100,
        "max_area_um2":   7000,
        "min_brightness": None,
    },
}

SAVE_MASKS    = True
SAVE_OVERLAYS = True
SAVE_OUTLINES = True
SAVE_COMBINED = True          # both channels outlined on one figure
SAVE_SETTINGS = True          # write run_settings_<timestamp>.txt
OVERLAY_ALPHA = 0.4
COLORMAP      = "tab20"

PX_TO_UM2 = None              # computed at runtime from PIXEL_SIZE_UM

# ═══════════════════════════════════════════════════════════════
#  IMAGE LOADING
# ═══════════════════════════════════════════════════════════════

def load_image(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[0] <= 4:
        img = img[0]
    elif img.ndim == 3:
        img = img[:, :, 0]
    img = img.squeeze().astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def load_manual_mask(tif_path: Path):
    seg_path = tif_path.parent / (tif_path.stem + "_seg.npy")
    if not seg_path.exists():
        return None
    try:
        seg   = np.load(str(seg_path), allow_pickle=True).item()
        masks = seg.get("masks", seg.get("mask", None))
        if masks is None:
            print(f"    [warn] {seg_path.name}: no 'masks' key found")
            return None
        return masks.astype(np.uint16)
    except Exception as e:
        print(f"    [warn] Could not load {seg_path.name}: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
#  FILE PAIRING
# ═══════════════════════════════════════════════════════════════

def find_pairs(folder: Path) -> dict:
    """
    Group TIFs into {base_name: {"target": Path, "macrophage": Path}}.
    Files whose stem does not end with a configured suffix are ignored.
    """
    pairs, ignored = {}, []
    for p in sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))):
        matched = False
        for channel, suffix in CHANNEL_SUFFIX.items():
            m = re.match(rf"^(.*?)[ _\-]*{suffix}$", p.stem, flags=re.IGNORECASE)
            if m:
                base = m.group(1).rstrip(" _-")
                pairs.setdefault(base, {})[channel] = p
                matched = True
                break
        if not matched:
            ignored.append(p.name)
    if ignored:
        print(f"  [note] {len(ignored)} file(s) ignored (no "
              f"{'/'.join(CHANNEL_SUFFIX.values())} suffix), e.g. {ignored[0]}")
    return dict(sorted(pairs.items()))

# ═══════════════════════════════════════════════════════════════
#  RUN SETTINGS LOG
# ═══════════════════════════════════════════════════════════════

def save_run_settings(output_root: Path, model_tag: str, timestamp: str,
                      input_dirs=None) -> Path:
    """
    Write a plain-text record of every parameter used for this run into
    the output directory, alongside the CSVs. Called before processing
    starts, so the record survives even if the run is interrupted.
    """
    def fmt(v):
        return "None (disabled)" if v is None else v

    try:
        import cellpose
        cp_version = getattr(cellpose, "version", None) or \
                     getattr(cellpose, "__version__", "unknown")
    except Exception:
        cp_version = "unknown"

    L = []
    A = L.append

    A("=" * 70)
    A(" CELLPOSE DUAL-CHANNEL RUN SETTINGS")
    A("=" * 70)
    A(f"Run timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"File tag        : {timestamp}")
    A(f"Script          : {Path(__file__).name if '__file__' in globals() else 'interactive'}")
    A(f"CellPose version: {cp_version}")
    A(f"Python          : {sys.version.split()[0]}")
    A("")

    A("-" * 70)
    A(" PATHS")
    A("-" * 70)
    for d in (input_dirs if input_dirs is not None else INPUT_DIRS):
        exists = "" if Path(d).exists() else "   [NOT FOUND]"
        A(f"  input  : {d}{exists}")
    A(f"  output : {output_root}")
    A("")

    A("-" * 70)
    A(" MODEL / GENERAL")
    A("-" * 70)
    A(f"  model_type    : {MODEL_TYPE}")
    A(f"  model_tag     : {model_tag}")
    A(f"  use_sam       : {USE_SAM}")
    A(f"  manual_mode   : {MANUAL_MODE}"
      + ("   (existing _seg.npy used as-is, no filters applied)"
         if MANUAL_MODE else "   (all images auto-segmented)"))
    A(f"  pixel_size_um : {PIXEL_SIZE_UM} um/px")
    A(f"  um2_per_px2   : {PX_TO_UM2}")
    A("")

    A("-" * 70)
    A(" CHANNEL ASSIGNMENT")
    A("-" * 70)
    for channel, suffix in CHANNEL_SUFFIX.items():
        A(f"  {channel:<12} : files ending in {suffix}.tif")
    A("")

    A("-" * 70)
    A(" SEGMENTATION PARAMETERS  (passed to model.eval)")
    A("-" * 70)
    for channel, params in SEG_PARAMS.items():
        A(f"  [{channel}]")
        for k, v in params.items():
            A(f"      {k:<20} : {fmt(v)}")
        A("")

    A("-" * 70)
    A(" POST-SEGMENTATION FILTERS  (auto masks only)")
    A("-" * 70)
    A("  Areas in um^2. min_brightness = cell mean intensity / image mean.")
    A("")
    for channel, f in FILTERS.items():
        A(f"  [{channel}]")
        for k, v in f.items():
            A(f"      {k:<20} : {fmt(v)}")
            if k.endswith("_um2") and v is not None and PX_TO_UM2:
                A(f"      {'':<20}   (= {v / PX_TO_UM2:.1f} px^2)")
        A("")

    A("-" * 70)
    A(" OUTPUTS")
    A("-" * 70)
    A(f"  save_masks    : {SAVE_MASKS}")
    A(f"  save_overlays : {SAVE_OVERLAYS}")
    A(f"  save_outlines : {SAVE_OUTLINES}")
    A(f"  save_combined : {SAVE_COMBINED}")
    A(f"  overlay_alpha : {OVERLAY_ALPHA}")
    A(f"  colormap      : {COLORMAP}")
    A("")
    A(f"  per_image_summary_{timestamp}.csv")
    A(f"  per_cell_detail_{timestamp}.csv")
    A("=" * 70)

    out_path = Path(output_root)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / f"run_settings_{timestamp}.txt"
    out_path.write_text("\n".join(L), encoding="utf-8")
    return out_path

# ═══════════════════════════════════════════════════════════════
#  MEASUREMENT / FILTERING
# ═══════════════════════════════════════════════════════════════

def measure_and_filter(img, mask, channel, apply_filters=True):
    """
    Returns (relabelled_mask, stats). When apply_filters is False
    (manual masks) every region is kept exactly as drawn.
    """
    f        = FILTERS[channel]
    img_mean = img.mean()
    kept, removed_size, removed_bright, cell_records = [], [], [], []

    for rp in regionprops(mask, intensity_image=img):
        area_um2 = rp.area * PX_TO_UM2
        mean_int = float(rp.mean_intensity)

        if apply_filters:
            if f["min_area_um2"] is not None and area_um2 < f["min_area_um2"]:
                removed_size.append(rp.label); continue
            if f["max_area_um2"] is not None and area_um2 > f["max_area_um2"]:
                removed_size.append(rp.label); continue
            if f["min_brightness"] is not None and img_mean > 0:
                if (mean_int / img_mean) < f["min_brightness"]:
                    removed_bright.append(rp.label); continue

        kept.append(rp.label)
        cell_records.append({
            "cell_label":     int(rp.label),
            "area_px2":       int(rp.area),
            "area_um2":       round(area_um2, 2),
            "mean_intensity": round(mean_int, 4),
        })

    if apply_filters and (removed_size or removed_bright):
        filtered = mask.copy()
        for lbl in removed_size + removed_bright:
            filtered[filtered == lbl] = 0
        new_mask = np.zeros_like(filtered)
        for new_id, old_lbl in enumerate(kept, start=1):
            new_mask[filtered == old_lbl] = new_id
        for new_id, rec in enumerate(cell_records, start=1):
            rec["cell_label"] = new_id
    else:
        new_mask = mask

    return new_mask, {
        "n_removed_size":       len(removed_size),
        "n_removed_brightness": len(removed_bright),
        "cell_records":         cell_records,
    }


def summarise_area(cell_records: list) -> dict:
    if not cell_records:
        return {"total_area_px2": 0, "total_area_um2": 0.0,
                "mean_area_px2": 0.0, "mean_area_um2": 0.0}
    pxs  = [r["area_px2"] for r in cell_records]
    um2s = [r["area_um2"] for r in cell_records]
    n    = len(cell_records)
    return {
        "total_area_px2": int(sum(pxs)),
        "total_area_um2": round(sum(um2s), 2),
        "mean_area_px2":  round(sum(pxs) / n, 2),
        "mean_area_um2":  round(sum(um2s) / n, 2),
    }

# ═══════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════

def save_overlay_png(img, mask, channel, n_cells, area_str, source_tag, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
    cmap    = plt.get_cmap(COLORMAP)
    colored = cmap(mask % cmap.N).astype(float)
    colored[mask == 0] = 0
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(colored, alpha=OVERLAY_ALPHA)
    axes[1].set_title(f"{n_cells} {channel}s  [{source_tag}]\n{area_str}", fontsize=10)
    axes[1].axis("off")
    plt.tight_layout(); fig.savefig(str(out_path), bbox_inches="tight"); plt.close(fig)


def save_outlines_png(img, mask, out_path):
    from cellpose.utils import masks_to_outlines
    outlines = masks_to_outlines(mask)
    fig, ax  = plt.subplots(figsize=(7, 7), dpi=150)
    ax.imshow(img, cmap="gray")
    ov = np.zeros((*img.shape, 4), dtype=float)
    ov[outlines] = [1, 0.2, 0.2, 0.9]
    ax.imshow(ov)
    ax.set_title(f"{int(mask.max())} cells"); ax.axis("off")
    plt.tight_layout(); fig.savefig(str(out_path), bbox_inches="tight"); plt.close(fig)


def save_combined_png(base, results, out_path):
    """Target outlines (red) + macrophage outlines (cyan) on one figure."""
    from cellpose.utils import masks_to_outlines
    ref = results.get("target") or results.get("macrophage")
    if ref is None:
        return
    img = ref["img"]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.imshow(img, cmap="gray")
    colors = {"target": [1, 0.2, 0.2, 0.9], "macrophage": [0.2, 0.9, 1.0, 0.9]}
    bits   = []
    for channel, res in results.items():
        if res is None or res["mask"].shape != img.shape:
            continue
        ov = np.zeros((*img.shape, 4), dtype=float)
        ov[masks_to_outlines(res["mask"])] = colors[channel]
        ax.imshow(ov)
        bits.append(f"{res['n_cells']} {channel}s")
    ax.set_title(f"{base}\n{'  |  '.join(bits)}", fontsize=9); ax.axis("off")
    plt.tight_layout(); fig.savefig(str(out_path), bbox_inches="tight"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════

def _gpu_available():
    try:
        from cellpose import core; return core.use_gpu()
    except Exception:
        return False


def build_model():
    from cellpose import models as cp_models
    gpu   = _gpu_available()
    model = cp_models.CellposeModel(model_type=MODEL_TYPE, gpu=gpu)
    tag   = f"cellpose_sam_{MODEL_TYPE}" if USE_SAM else f"cellpose_{MODEL_TYPE}"
    print(f"[model] {tag}  |  GPU={gpu}")
    return model, tag

# ═══════════════════════════════════════════════════════════════
#  PER-CHANNEL PROCESSING
# ═══════════════════════════════════════════════════════════════

def process_channel(tif_path, channel, model, out_folder, model_tag):
    stem = tif_path.stem
    img  = load_image(tif_path)

    masks_raw, source_tag = None, "auto"
    if MANUAL_MODE:
        masks_raw = load_manual_mask(tif_path)
        if masks_raw is not None:
            source_tag = "manual"
    if masks_raw is None:
        masks_raw, _, _ = model.eval(img, **SEG_PARAMS[channel].copy())

    # Manual masks are measured exactly as drawn; auto masks get the filters.
    mask, fstats = measure_and_filter(
        img, masks_raw, channel, apply_filters=(source_tag != "manual"))

    areas    = summarise_area(fstats["cell_records"])
    n_cells  = len(fstats["cell_records"])
    area_str = (f"total: {areas['total_area_um2']:,.0f} um2 "
                f"({areas['total_area_px2']:,} px2) | "
                f"mean: {areas['mean_area_um2']:,.1f} um2")

    if SAVE_MASKS:
        tifffile.imwrite(str(out_folder / channel / "masks" / f"{stem}_mask.tif"),
                         mask.astype(np.uint16), compression="zlib")
    if SAVE_OVERLAYS:
        save_overlay_png(img, mask, channel, n_cells, area_str, source_tag,
                         out_folder / channel / "overlays" / f"{stem}_overlay.png")
    if SAVE_OUTLINES:
        save_outlines_png(img, mask,
                          out_folder / channel / "outlines" / f"{stem}_outlines.png")

    print(f"    {channel:<11} {tif_path.name:<40} "
          f"{n_cells:>4} cells | {area_str}"
          + (" [manual - no filters]" if source_tag == "manual"
             else f" | rm size={fstats['n_removed_size']} "
                  f"bright={fstats['n_removed_brightness']}"))

    return {
        "img":        img,
        "mask":       mask,
        "n_cells":    n_cells,
        "areas":      areas,
        "source":     source_tag,
        "filename":   tif_path.name,
        "fstats":     fstats,
        "model_tag":  model_tag,
    }

# ═══════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(input_dirs, output_root):
    global PX_TO_UM2
    PX_TO_UM2 = (PIXEL_SIZE_UM ** 2) if PIXEL_SIZE_UM else 1.0

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, model_tag = build_model()

    if SAVE_SETTINGS:
        settings_path = save_run_settings(output_root, model_tag, ts, input_dirs)
        print(f"[settings] {settings_path}")

    rows, cell_rows = [], []

    for folder in input_dirs:
        folder = Path(folder)
        if not folder.exists():
            print(f"[skip] Not found: {folder}"); continue

        pairs = find_pairs(folder)
        if not pairs:
            print(f"[skip] No {'/'.join(CHANNEL_SUFFIX.values())} TIFs in {folder}")
            continue

        out_folder = output_root / folder.name
        for channel in CHANNEL_SUFFIX:
            for sub in ("masks", "overlays", "outlines"):
                (out_folder / channel / sub).mkdir(parents=True, exist_ok=True)
        if SAVE_COMBINED:
            (out_folder / "combined").mkdir(parents=True, exist_ok=True)

        print(f"\n{'-'*78}")
        print(f" Folder : {folder.name}  ({len(pairs)} field(s) of view)")
        print(f" Pixel  : {PIXEL_SIZE_UM} um/px  ->  {PX_TO_UM2:.4f} um2/px2")
        print(f"{'-'*78}")

        manual_used = 0
        for base, files in pairs.items():
            print(f"  {base}")
            results = {}
            for channel in CHANNEL_SUFFIX:
                path = files.get(channel)
                if path is None:
                    print(f"    {channel:<11} [missing {CHANNEL_SUFFIX[channel]} file]")
                    results[channel] = None
                    continue
                try:
                    results[channel] = process_channel(
                        path, channel, model, out_folder, model_tag)
                    if results[channel]["source"] == "manual":
                        manual_used += 1
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results[channel] = None

            if SAVE_COMBINED:
                try:
                    save_combined_png(base, results,
                                      out_folder / "combined" / f"{base}_combined.png")
                except Exception as e:
                    print(f"    [combined overlay failed: {e}]")

            row = {"folder": folder.name, "base_name": base}
            for channel in CHANNEL_SUFFIX:
                res = results.get(channel)
                pre = channel  # column prefix: target_* / macrophage_*
                if res is None:
                    row.update({
                        f"{pre}_filename":      files.get(channel).name
                                                if files.get(channel) else "",
                        f"{pre}_source":        "missing",
                        f"n_{pre}s":            np.nan,
                        f"{pre}_total_area_px2": np.nan,
                        f"{pre}_total_area_um2": np.nan,
                        f"{pre}_mean_area_px2":  np.nan,
                        f"{pre}_mean_area_um2":  np.nan,
                    })
                    continue
                a = res["areas"]
                row.update({
                    f"{pre}_filename":       res["filename"],
                    f"{pre}_source":         res["source"],
                    f"n_{pre}s":             res["n_cells"],
                    f"{pre}_total_area_px2": a["total_area_px2"],
                    f"{pre}_total_area_um2": a["total_area_um2"],
                    f"{pre}_mean_area_px2":  a["mean_area_px2"],
                    f"{pre}_mean_area_um2":  a["mean_area_um2"],
                })
                for cr in res["fstats"]["cell_records"]:
                    cell_rows.append({
                        "folder":    folder.name,
                        "base_name": base,
                        "filename":  res["filename"],
                        "channel":   channel,
                        "source":    res["source"],
                        **cr,
                    })
            row["pixel_size_um"] = PIXEL_SIZE_UM
            row["um2_per_px2"]   = round(PX_TO_UM2, 6)
            row["model"]         = model_tag
            rows.append(row)

        print(f"  -> manual masks used in this folder: {manual_used}")

    df_img   = pd.DataFrame(rows)
    df_cell  = pd.DataFrame(cell_rows)
    csv_img  = output_root / f"per_image_summary_{ts}.csv"
    csv_cell = output_root / f"per_cell_detail_{ts}.csv"
    df_img.to_csv(csv_img,  index=False, encoding="utf-8-sig")
    df_cell.to_csv(csv_cell, index=False, encoding="utf-8-sig")

    print(f"\n{'='*78}\n SUMMARY\n{'='*78}")
    if not df_img.empty:
        cols = [c for c in ("n_targets", "n_macrophages",
                            "target_total_area_um2", "macrophage_total_area_um2")
                if c in df_img.columns]
        if cols:
            print(df_img.groupby("folder")[cols].agg(["mean", "sum"]).round(1).to_string())

    print(f"\nPer-image CSV -> {csv_img}")
    print(f"Per-cell CSV  -> {csv_cell}")
    if SAVE_SETTINGS:
        print(f"Settings log  -> {output_root / f'run_settings_{ts}.txt'}")
    print(f"All outputs   -> {output_root}")
    return df_img, df_cell

# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs",    nargs="+", default=None)
    p.add_argument("--output_root",   default=None)
    p.add_argument("--pixel_size_um", type=float, default=None)
    p.add_argument("--model_type",    default=None)
    p.add_argument("--target_suffix",     default=None)
    p.add_argument("--macrophage_suffix", default=None)
    p.add_argument("--manual",     action="store_true", default=None)
    p.add_argument("--no_manual",  action="store_true", default=None)
    p.add_argument("--no_sam",     action="store_true", default=None)
    p.add_argument("--no_settings", action="store_true", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_dirs        is not None: INPUT_DIRS[:] = args.input_dirs
    if args.output_root       is not None: OUTPUT_ROOT   = args.output_root
    if args.pixel_size_um     is not None: PIXEL_SIZE_UM = args.pixel_size_um
    if args.model_type        is not None: MODEL_TYPE    = args.model_type
    if args.target_suffix     is not None: CHANNEL_SUFFIX["target"]     = args.target_suffix
    if args.macrophage_suffix is not None: CHANNEL_SUFFIX["macrophage"] = args.macrophage_suffix
    if args.manual:      MANUAL_MODE   = True
    if args.no_manual:   MANUAL_MODE   = False
    if args.no_sam:      USE_SAM       = False
    if args.no_settings: SAVE_SETTINGS = False

    run_pipeline(INPUT_DIRS, OUTPUT_ROOT)