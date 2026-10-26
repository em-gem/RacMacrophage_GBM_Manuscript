# -*- coding: utf-8 -*-
"""
GBM (target) nearest-neighbour analysis
=======================================
Companion to cellpose_dual_channel.py. Reads the target masks saved under
MASK_ROOT/<folder>/target/masks/*_mask.tif  and, for every field of view
(FOV):

  - finds the centroid of every target cell
  - measures each cell's distance to its nearest neighbouring target cell
  - summarises those distances per FOV and per folder
  - measures patchiness: how unevenly the cell masks cover the image
  - saves QC images so every step can be checked by eye

Patchiness (PATCHINESS = True): the image is cut into square tiles of
TILE_SIZE_UM, the fraction of each tile covered by cell masks is measured,
and patchiness_cv = SD / mean of those tile coverages.
  low  (~0.3) -> cells cover the field evenly (e.g. a continuous sheet)
  high (>1)   -> cells packed into islands with empty space between
This uses the masks, not the centroids. It is NOT density-corrected: at
low coverage it rises even if cells are scattered at random, so read it
next to coverage_frac and nnd_ratio_vs_random. Keep TILE_SIZE_UM the same
for every condition you compare.

Optional (COMPARE_TO_RANDOM = True): the same number of cells is placed at
random NSIM times in the same image frame, keeping them at least one cell
apart, and the observed mean NND is divided by the random mean NND.
  nnd_ratio < 1  -> cells closer together than chance (clustered)
  nnd_ratio ~ 1  -> random
  nnd_ratio > 1  -> more evenly spread than chance
Because the random cells sit in the same frame, edge effects cancel in
this ratio. The raw mean NND is NOT edge-corrected, and it grows whenever
there are fewer cells even if the arrangement is unchanged.

Outputs (in OUTPUT_ROOT):
  per_fov_nnd_<ts>.csv          FINAL MEASUREMENTS, one row per FOV
  per_folder_summary_<ts>.csv   folder means / medians of the FOV values
  per_cell_nnd_<ts>.csv         one row per target cell
  per_tile_coverage_<ts>.csv    one row per patchiness tile
  nn_settings_<ts>.txt          parameters used for this run
  <folder>/centroids_tif/<base>_centroids.tif   (QC) dot at each centroid,
                                pixel value = cell label; same size as the
                                mask, so it overlays directly in Fiji
  <folder>/centroids_png/<base>_centroids.png   (QC) outlines + centroids
                                + cell labels
  <folder>/nn_png/<base>_nn.png (QC) each cell linked to its nearest
                                neighbour, coloured by distance
  <folder>/nnd_hist_png/<base>_nnd_hist.png     (QC) distribution of cell
                                NNDs, and observed vs random mean NND
  <folder>/patchiness_png/<base>_patchiness.png (QC) tile grid over the
                                masks, each tile's % coverage written on it

cell_label matches cell_label in the pipeline's per_cell_detail CSV, so
the tables merge on folder + base_name + cell_label.

Usage:
    python gbm_nearest_neighbors.py
    python gbm_nearest_neighbors.py --mask_root D:/out/OutputV9 --no_random
"""

import argparse, re, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from skimage.draw import disk
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Folder that cellpose_dual_channel.py wrote to (its OUTPUT_ROOT).
MASK_ROOT   = "C:/Users/980 Analysis/Documents/AlyseGBMCellPose/DataToAnalyze_10142026_CorrectedImages/CellPoseOutput"

# Where this script writes everything. Built from MASK_ROOT by default;
# replace with any full path to put the results somewhere else.
OUTPUT_SUBFOLDER = "NearestNeighborsV5_targets"
OUTPUT_ROOT = MASK_ROOT + "/" + OUTPUT_SUBFOLDER

PIXEL_SIZE_UM = 1.3811        # must match the segmentation run
TARGET_SUFFIX = "C4"

# ── Random comparison ──────────────────────────────────────────
COMPARE_TO_RANDOM = True      # adds nnd_ratio_vs_random (density-corrected)
NSIM              = 199       # random placements per FOV
MIN_SPACING_UM    = "auto"    # random cells kept at least this far apart.
                              # "auto" = smallest observed NND in the FOV,
                              # capped at the median cell diameter.
                              # 0 = pure random points (ignores cell size).
RANDOM_SEED       = 12345
MIN_CELLS         = 5         # fewer cells -> ratio not computed

# ── Patchiness ─────────────────────────────────────────────────
PATCHINESS          = True
TILE_SIZE_UM        = 177     # tile side; 177 um = 128 px = 8 x 8 tiles on a
                              # 1024-px image. Partial tiles at the image
                              # edge are dropped (grid is centred).
MIN_TILES           = 4       # fewer whole tiles -> patchiness not computed
EMPTY_TILE_COVERAGE = 0.01    # tiles below this coverage count as "empty"

# ── QC outputs ─────────────────────────────────────────────────
SAVE_CENTROID_TIF = True
SAVE_CENTROID_PNG = True
SAVE_NN_PNG       = True
SAVE_HIST_PNG     = True
SAVE_PATCH_PNG    = True
CENTROID_DOT_PX   = 3         # radius of the dot drawn in the centroid TIF
LABEL_CELLS       = True      # write cell_label numbers on the centroid PNG
QC_DPI            = 150

# ═══════════════════════════════════════════════════════════════
#  MEASUREMENT
# ═══════════════════════════════════════════════════════════════

def _p(path):
    """
    Path string safe for long Windows paths. Windows refuses paths over
    260 characters unless they carry the \\\\?\\ prefix; long image names
    plus nested output folders easily exceed that.
    """
    path = Path(path).resolve()
    if sys.platform.startswith("win"):
        s = str(path)
        if not s.startswith("\\\\?\\"):
            s = ("\\\\?\\UNC\\" + s[2:]) if s.startswith("\\\\") else "\\\\?\\" + s
        return s
    return str(path)


def find_masks(mask_root: Path):
    for tdir in sorted(mask_root.glob("*/target/masks")):
        folder = tdir.parent.parent.name
        for p in sorted(tdir.glob("*_mask.tif*")):
            stem = re.sub(r"_mask$", "", p.stem, flags=re.IGNORECASE)
            m = re.match(rf"^(.*?)[ _\-]*{TARGET_SUFFIX}$", stem, flags=re.IGNORECASE)
            yield folder, (m.group(1).rstrip(" _-") if m else stem), p


def nearest(xy):
    if len(xy) < 2:
        return np.full(len(xy), np.nan), np.full(len(xy), -1)
    d, i = cKDTree(xy).query(xy, k=2)
    return d[:, 1], i[:, 1]


def random_mean_nnd(n, W, H, spacing, rng):
    """Mean NND of n random cells in a W x H frame (um), min spacing kept."""
    if spacing <= 0:
        return nearest(rng.uniform(size=(n, 2)) * [W, H])[0].mean()
    pts, k, tries, s2 = np.empty((n, 2)), 0, 0, spacing ** 2
    while k < n:
        for c in rng.uniform(size=(512, 2)) * [W, H]:
            tries += 1
            if k == 0 or ((pts[:k] - c) ** 2).sum(1).min() >= s2:
                pts[k] = c; k += 1
                if k == n:
                    break
        if tries > 2000 * n:
            return np.nan                      # too crowded to place
    return nearest(pts)[0].mean()


def tile_coverage(mask):
    """
    Fraction of each whole TILE_SIZE_UM tile covered by cell masks.
    Returns (coverage[ny, nx], tile_px, y0, x0) or None if too few tiles.
    """
    fg = mask > 0
    Hpx, Wpx = fg.shape
    t = int(round(TILE_SIZE_UM / PIXEL_SIZE_UM))
    ny, nx = Hpx // t, Wpx // t
    if t < 1 or ny * nx < MIN_TILES:
        return None
    y0, x0 = (Hpx - ny * t) // 2, (Wpx - nx * t) // 2
    cov = fg[y0:y0 + ny * t, x0:x0 + nx * t].reshape(ny, t, nx, t).mean(axis=(1, 3))
    return cov, t, y0, x0


def analyse(folder, base, path, rng):
    mask = tifffile.imread(_p(path)).astype(np.int32)
    H, W = np.array(mask.shape) * PIXEL_SIZE_UM
    props = regionprops(mask)
    n = len(props)
    labels = np.array([p.label for p in props], dtype=int)
    rc = np.array([p.centroid for p in props]) if n else np.zeros((0, 2))
    xy = rc[:, ::-1] * PIXEL_SIZE_UM                     # (x, y) in um
    area = np.array([p.area for p in props]) * PIXEL_SIZE_UM ** 2
    d, i = nearest(xy)
    mutual = np.array([j >= 0 and i[j] == k for k, j in enumerate(i)], bool)

    cells = pd.DataFrame({
        "folder": folder, "base_name": base, "cell_label": labels,
        "centroid_x_px": np.round(rc[:, 1], 2),
        "centroid_y_px": np.round(rc[:, 0], 2),
        "centroid_x_um": np.round(xy[:, 0], 2),
        "centroid_y_um": np.round(xy[:, 1], 2),
        "area_um2": np.round(area, 2),
        "nnd_um": np.round(d, 2),
        "nearest_neighbour_label": [labels[j] if j >= 0 else -1 for j in i],
        "mutual_nearest_neighbours": mutual,
    })

    row = {"folder": folder, "base_name": base, "mask_file": path.name,
           "n_targets": n,
           "image_w_um": round(W, 1), "image_h_um": round(H, 1),
           "density_per_mm2": round(n / (W * H) * 1e6, 2)}
    if n >= 2:
        row.update({"mean_nnd_um": round(d.mean(), 2),
                    "median_nnd_um": round(np.median(d), 2),
                    "sd_nnd_um": round(d.std(ddof=1), 2) if n > 2 else np.nan,
                    "min_nnd_um": round(d.min(), 2),
                    "max_nnd_um": round(d.max(), 2),
                    "frac_mutual_pairs": round(mutual.mean(), 4)})

    sims = None
    if COMPARE_TO_RANDOM and n >= MIN_CELLS:
        if MIN_SPACING_UM == "auto":
            diam = np.median(2 * np.sqrt(area / np.pi))
            spacing = min(d.min(), diam)
        else:
            spacing = float(MIN_SPACING_UM)
        sims = np.array([random_mean_nnd(n, W, H, spacing, rng) for _ in range(NSIM)])
        sims = sims[~np.isnan(sims)]
        if len(sims) >= 19:
            row.update({
                "min_spacing_um": round(spacing, 2),
                "n_random_sims": len(sims),
                "random_mean_nnd_um": round(sims.mean(), 2),
                "nnd_ratio_vs_random": round(d.mean() / sims.mean(), 4),
                "p_clustered": round((1 + np.sum(sims <= d.mean())) / (len(sims) + 1), 4),
            })
        else:
            row["note"] = f"random placement failed (spacing {spacing:.1f} um too crowded)"
            sims = None
    elif COMPARE_TO_RANDOM:
        row["note"] = f"n < {MIN_CELLS}: no random comparison"

    # ── patchiness (mask-based) ───────────────────────────────
    row["coverage_frac"] = round(float((mask > 0).mean()), 4)
    tiles, tiles_df = None, None
    if PATCHINESS:
        tiles = tile_coverage(mask)
        if tiles is None:
            row["note"] = (row.get("note", "") + "; " if "note" in row else "") + \
                          f"fewer than {MIN_TILES} whole tiles: no patchiness"
        else:
            cov, t, y0, x0 = tiles
            m = cov.mean()
            row.update({
                "tile_size_um": round(t * PIXEL_SIZE_UM, 1),
                "n_tiles": int(cov.size),
                "tile_coverage_mean": round(float(m), 4),
                "tile_coverage_sd": round(float(cov.std()), 4),
                "patchiness_cv": round(float(cov.std() / m), 4) if m > 0 else np.nan,
                "frac_tiles_empty": round(float((cov < EMPTY_TILE_COVERAGE).mean()), 4),
            })
            ty, tx = np.indices(cov.shape)
            tiles_df = pd.DataFrame({
                "folder": folder, "base_name": base,
                "tile_row": ty.ravel(), "tile_col": tx.ravel(),
                "x0_px": x0 + tx.ravel() * t, "y0_px": y0 + ty.ravel() * t,
                "tile_px": t, "coverage": np.round(cov.ravel(), 4),
            })

    qc = {"mask": mask, "rc": rc, "labels": labels, "nn_idx": i,
          "nnd": d, "mutual": mutual, "sims": sims,
          "tiles": tiles, "tiles_df": tiles_df}
    return cells, row, qc

# ═══════════════════════════════════════════════════════════════
#  QC OUTPUTS
# ═══════════════════════════════════════════════════════════════

def _background(ax, mask):
    """Grey cell fill + dark outlines, in pixel coordinates."""
    bg = np.ones((*mask.shape, 3))
    bg[mask > 0] = 0.85
    bg[find_boundaries(mask, mode="inner")] = 0.45
    ax.imshow(bg, interpolation="nearest")
    ax.set_xlim(0, mask.shape[1]); ax.set_ylim(mask.shape[0], 0)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")


def _fig_for(mask):
    h, w = mask.shape
    return plt.subplots(figsize=(9, 9 * h / w + 0.6), dpi=QC_DPI)


def save_centroid_tif(qc, out_path):
    img = np.zeros(qc["mask"].shape, np.uint16)
    for (r, c), lab in zip(qc["rc"], qc["labels"]):
        rr, cc = disk((r, c), CENTROID_DOT_PX, shape=img.shape)
        img[rr, cc] = lab
    tifffile.imwrite(_p(out_path), img, compression="zlib")


def save_centroid_png(base, row, qc, out_path):
    fig, ax = _fig_for(qc["mask"])
    _background(ax, qc["mask"])
    rc = qc["rc"]
    if len(rc):
        ax.scatter(rc[:, 1], rc[:, 0], s=10, c="crimson", zorder=3)
        if LABEL_CELLS:
            fs = max(3, min(7, 400 / np.sqrt(len(rc) + 1) / 4))
            for (r, c), lab in zip(rc, qc["labels"]):
                ax.text(c + 4, r - 4, str(lab), fontsize=fs, color="navy", zorder=4)
    ax.set_title(f"{base}  |  {row['n_targets']} cells  |  centroids (red), "
                 f"cell_label (blue)", fontsize=9)
    plt.tight_layout(); fig.savefig(_p(out_path), bbox_inches="tight"); plt.close(fig)


def save_nn_png(base, row, qc, out_path):
    fig, ax = _fig_for(qc["mask"])
    _background(ax, qc["mask"])
    rc, idx, d, mutual = qc["rc"], qc["nn_idx"], qc["nnd"], qc["mutual"]
    if len(rc) >= 2:
        vmin, vmax = np.nanpercentile(d, [2, 98])
        norm = plt.Normalize(vmin, vmax if vmax > vmin else vmin + 1)
        cmap = plt.get_cmap("plasma")
        # mutual pairs: one plain line per pair
        segs, cols = [], []
        for k, j in enumerate(idx):
            if mutual[k] and k < j:
                segs.append([(rc[k, 1], rc[k, 0]), (rc[j, 1], rc[j, 0])])
                cols.append(cmap(norm(d[k])))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.8, zorder=2))
        # one-way: arrow from the cell to its nearest neighbour
        for k, j in enumerate(idx):
            if not mutual[k]:
                ax.annotate("", xy=(rc[j, 1], rc[j, 0]), xytext=(rc[k, 1], rc[k, 0]),
                            arrowprops=dict(arrowstyle="-|>", color=cmap(norm(d[k])),
                                            lw=1.1, mutation_scale=8,
                                            shrinkA=0, shrinkB=3), zorder=2)
        ax.scatter(rc[:, 1], rc[:, 0], s=8, c="k", zorder=3)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02,
                     label="nearest-neighbour distance (um)")
        ax.legend(handles=[Line2D([], [], c="0.3", lw=1.8, label="mutual nearest neighbours"),
                           Line2D([], [], c="0.3", lw=1.1, marker=">", markevery=[1],
                                  label="one-way: arrow points to the neighbour")],
                  loc="upper right", fontsize=7, framealpha=0.85)
    ratio = row.get("nnd_ratio_vs_random")
    ax.set_title(f"{base}  |  n={row['n_targets']}  |  mean NND "
                 f"{row.get('mean_nnd_um', float('nan')):.1f} um"
                 + (f"  |  ratio vs random {ratio:.2f}" if ratio is not None else ""),
                 fontsize=9)
    plt.tight_layout(); fig.savefig(_p(out_path), bbox_inches="tight"); plt.close(fig)


def save_hist_png(base, row, qc, out_path):
    d, sims = qc["nnd"], qc["sims"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=QC_DPI)
    a = ax[0]
    if len(d) >= 2:
        a.hist(d, bins=min(40, max(8, len(d) // 4)), color="0.6")
        a.axvline(d.mean(), c="crimson", lw=2, label=f"mean {d.mean():.1f} um")
        a.axvline(np.median(d), c="navy", lw=2, ls="--",
                  label=f"median {np.median(d):.1f} um")
        a.legend(fontsize=8)
    a.set_xlabel("nearest-neighbour distance per cell (um)"); a.set_ylabel("cells")
    a.set_title(f"{base}: per-cell NND (n={row['n_targets']})", fontsize=9)

    a = ax[1]
    if sims is not None:
        a.hist(sims, bins=25, color="0.75", label=f"{len(sims)} random placements")
        a.axvline(d.mean(), c="crimson", lw=2, label="observed")
        a.set_title(f"ratio {row['nnd_ratio_vs_random']:.3f}  |  "
                    f"p(clustered) = {row['p_clustered']:.3f}", fontsize=9)
        a.set_xlabel("mean NND of the whole FOV (um)"); a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, row.get("note", "random comparison off"),
               ha="center", va="center", transform=a.transAxes, fontsize=9)
        a.set_axis_off()
    plt.tight_layout(); fig.savefig(_p(out_path), bbox_inches="tight"); plt.close(fig)


def save_patch_png(base, row, qc, out_path):
    fig, ax = _fig_for(qc["mask"])
    _background(ax, qc["mask"])
    if qc["tiles"] is not None:
        cov, t, y0, x0 = qc["tiles"]
        ny, nx = cov.shape
        im = ax.imshow(cov, cmap="viridis", vmin=0, vmax=1, alpha=0.45,
                       extent=[x0, x0 + nx * t, y0 + ny * t, y0],
                       interpolation="nearest", zorder=2)
        for k in range(ny + 1):
            ax.plot([x0, x0 + nx * t], [y0 + k * t] * 2, c="k", lw=0.6, zorder=3)
        for k in range(nx + 1):
            ax.plot([x0 + k * t] * 2, [y0, y0 + ny * t], c="k", lw=0.6, zorder=3)
        fs = max(4, min(9, 70 / max(ny, nx)))
        for r in range(ny):
            for c in range(nx):
                ax.text(x0 + (c + 0.5) * t, y0 + (r + 0.5) * t, f"{100 * cov[r, c]:.0f}",
                        ha="center", va="center", fontsize=fs, fontweight="bold",
                        color="white" if cov[r, c] < 0.5 else "k", zorder=4)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="tile coverage")
        ax.set_title(f"{base}  |  coverage {100 * row['coverage_frac']:.0f}%  |  "
                     f"patchiness CV {row['patchiness_cv']:.2f}  |  "
                     f"{100 * row['frac_tiles_empty']:.0f}% of tiles empty\n"
                     f"{ny} x {nx} tiles of {row['tile_size_um']:.0f} um; "
                     f"number = % of tile covered by cells", fontsize=9)
    else:
        ax.set_title(f"{base}  |  {row.get('note', 'no patchiness')}", fontsize=9)
    plt.tight_layout(); fig.savefig(_p(out_path), bbox_inches="tight"); plt.close(fig)


def write_settings(out, ts, mask_root, n_fov):
    keys = ["PIXEL_SIZE_UM", "TARGET_SUFFIX", "COMPARE_TO_RANDOM", "NSIM",
            "MIN_SPACING_UM", "RANDOM_SEED", "MIN_CELLS", "PATCHINESS",
            "TILE_SIZE_UM", "MIN_TILES", "EMPTY_TILE_COVERAGE", "CENTROID_DOT_PX"]
    g = globals()
    lines = ["NEAREST-NEIGHBOUR RUN SETTINGS",
             f"Run        : {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Python     : {sys.version.split()[0]}",
             f"MASK_ROOT  : {mask_root}",
             f"OUTPUT_ROOT: {out}",
             f"FOVs found : {n_fov}", ""]
    lines += [f"{k:<18}: {g[k]}" for k in keys]
    Path(_p(out / f"nn_settings_{ts}.txt")).write_text("\n".join(lines), encoding="utf-8")

# ═══════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════

def run(mask_root, output_root):
    mask_root, out = Path(mask_root), Path(output_root)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = np.random.default_rng(RANDOM_SEED)

    found = list(find_masks(mask_root))
    if not found:
        print(f"No target masks found under {mask_root}/*/target/masks/")
        return
    Path(_p(out)).mkdir(parents=True, exist_ok=True)
    write_settings(out, ts, mask_root, len(found))
    print(f"{len(found)} FOV(s) found. Output -> {out}\n")

    qc_dirs = {"centroids_tif": SAVE_CENTROID_TIF, "centroids_png": SAVE_CENTROID_PNG,
               "nn_png": SAVE_NN_PNG, "nnd_hist_png": SAVE_HIST_PNG,
               "patchiness_png": SAVE_PATCH_PNG and PATCHINESS}

    all_cells, rows, all_tiles = [], [], []
    for folder, base, path in found:
        try:
            cells, row, qc = analyse(folder, base, path, rng)
        except Exception:
            import traceback; traceback.print_exc(); continue
        all_cells.append(cells); rows.append(row)
        if qc["tiles_df"] is not None:
            all_tiles.append(qc["tiles_df"])
        print(f"  {folder:<28} {base:<34} n={row['n_targets']:>4}  "
              f"meanNND={row.get('mean_nnd_um', np.nan):>7}  "
              f"ratio={row.get('nnd_ratio_vs_random', np.nan)}  "
              f"patchCV={row.get('patchiness_cv', np.nan)}")

        for sub, on in qc_dirs.items():
            if on:
                Path(_p(out / folder / sub)).mkdir(parents=True, exist_ok=True)
        qc_jobs = [
            (SAVE_CENTROID_TIF, save_centroid_tif, (qc,), "centroids_tif", "_centroids.tif"),
            (SAVE_CENTROID_PNG, save_centroid_png, (base, row, qc), "centroids_png", "_centroids.png"),
            (SAVE_NN_PNG, save_nn_png, (base, row, qc), "nn_png", "_nn.png"),
            (SAVE_HIST_PNG, save_hist_png, (base, row, qc), "nnd_hist_png", "_nnd_hist.png"),
            (SAVE_PATCH_PNG and PATCHINESS, save_patch_png, (base, row, qc),
             "patchiness_png", "_patchiness.png"),
        ]
        for on, fn, args, sub, suffix in qc_jobs:
            if not on:
                continue
            try:
                fn(*args, out / folder / sub / f"{base}{suffix}")
            except Exception as e:
                print(f"    [{sub} failed: {e}]")

    df = pd.DataFrame(rows)
    df.to_csv(_p(out / f"per_fov_nnd_{ts}.csv"), index=False, encoding="utf-8-sig")
    pd.concat(all_cells, ignore_index=True).to_csv(
        _p(out / f"per_cell_nnd_{ts}.csv"), index=False, encoding="utf-8-sig")
    if all_tiles:
        pd.concat(all_tiles, ignore_index=True).to_csv(
            _p(out / f"per_tile_coverage_{ts}.csv"), index=False, encoding="utf-8-sig")

    cols = [c for c in ("n_targets", "density_per_mm2", "mean_nnd_um",
                        "median_nnd_um", "nnd_ratio_vs_random", "coverage_frac",
                        "patchiness_cv", "frac_tiles_empty") if c in df]
    summ = df.groupby("folder")[cols].agg(["mean", "median", "std"]).round(3)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.insert(0, "n_fovs", df.groupby("folder").size())
    summ.to_csv(_p(out / f"per_folder_summary_{ts}.csv"), encoding="utf-8-sig")

    print(f"\n{'=' * 70}\n FOLDER MEDIANS\n{'=' * 70}")
    print(df.groupby("folder")[cols].median().round(3).to_string())
    print(f"\nFinal measurements -> {out / f'per_fov_nnd_{ts}.csv'}")
    print(f"All outputs        -> {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mask_root", default=None)
    p.add_argument("--output_root", default=None,
                   help="default: <mask_root>/<OUTPUT_SUBFOLDER>")
    p.add_argument("--pixel_size_um", type=float, default=None)
    p.add_argument("--no_random", action="store_true",
                   help="skip the random comparison (centroids + NND only)")
    p.add_argument("--tile_size_um", type=float, default=None)
    p.add_argument("--no_patchiness", action="store_true")
    a = p.parse_args()
    if a.mask_root:
        MASK_ROOT = a.mask_root
        OUTPUT_ROOT = MASK_ROOT + "/" + OUTPUT_SUBFOLDER
    if a.output_root:   OUTPUT_ROOT = a.output_root
    if a.pixel_size_um: PIXEL_SIZE_UM = a.pixel_size_um
    if a.no_random:     COMPARE_TO_RANDOM = False
    if a.tile_size_um:  TILE_SIZE_UM = a.tile_size_um
    if a.no_patchiness: PATCHINESS = False
    run(MASK_ROOT, OUTPUT_ROOT)