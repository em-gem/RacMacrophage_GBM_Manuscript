# RacMacrophage_GBM_Manuscript
Integration of Cellpose to segment macrophages and GBM cells and nuclei using different channels. Tiling and nearest-neighbor analysis to quantify clustering.


# CellPose Image Analysis: Macrophages, Glioblastoma Targets & Spatial Patterning

Python pipelines for segmenting and quantifying fluorescence microscopy images with [CellPose](https://github.com/MouseLand/cellpose). They support a workflow where CellPose segments images automatically, hand-corrected masks from the CellPose GUI take precedence wherever they exist, and every run logs the exact parameters it used.

| Script | What it does |
|---|---|
| `AutomatedCellposeSegmentation_WithManualMasksInput.py` | Segments paired target (GBM) and macrophage images, one row of results per field of view |
| `F480_CellTrace_ADCP_Analysis.py` | Uses manual macrophage masks (F4/80 or CellTracker Green) plus DAPI nuclei to count macrophages, GBM cells and multinucleation (ADCP assays) |
| `GBM_NearestNeighbors_Tiling.py` | Spatial analysis of segmented target cells: nearest-neighbour distances, comparison to random placement, and patchiness |

---

## Installation

Python 3.9+ is recommended.

```bash
pip install cellpose numpy pandas tifffile matplotlib scikit-image scipy
```

A CUDA-capable GPU is used automatically if CellPose detects one; otherwise everything runs on CPU.

**CellPose version note**
- `F480_CellTrace_ADCP_Analysis.py` defaults to the `cpdino` model for DAPI, which needs a recent **Cellpose 4.x** (`pip install -U cellpose`).
- `AutomatedCellposeSegmentation_WithManualMasksInput.py` passes `model_type="cyto3"`, a Cellpose 3.x model name. Under Cellpose 4.x the `model_type` argument is ignored and the default Cellpose-SAM model is used.
- `GBM_NearestNeighbors_Tiling.py` does not use CellPose at all, only the saved masks.

---

## File naming conventions

All scripts group images into fields of view (FOVs) by matching a channel suffix at the **end of the file stem** (case-insensitive; a separating `_`, `-` or space is optional):

```
Well3_Pos2_C1.tif        ->  base "Well3_Pos2", channel C1
Well3_Pos2_C4.tif        ->  base "Well3_Pos2", channel C4
Well3_Pos2_C4_seg.npy    ->  hand-corrected CellPose GUI mask for Well3_Pos2_C4.tif
```

Manual masks are the `<image name>_seg.npy` files the CellPose GUI writes next to the image when you save, so no renaming is needed. Manual masks are always used **exactly as drawn**: no size, brightness or SNR filters are applied to them.

---

## 1. `AutomatedCellposeSegmentation_WithManualMasksInput.py`

Dual-channel segmentation of targets and macrophages.

| Channel | Default suffix |
|---|---|
| target (GBM) | `C4` |
| macrophage | `C1` |

**For each FOV:**
1. Loads the target and macrophage images (percentile-normalised to 0–1).
2. Uses `<image>_seg.npy` if it exists and `MANUAL_MODE = True`; otherwise runs CellPose with the per-channel `SEG_PARAMS`.
3. Filters **auto** masks by area (µm²) and, for targets, by brightness relative to the whole image.
4. Reports cell counts and total/mean area in both px² and µm².

**Key settings** (top of the script):

| Setting | Meaning |
|---|---|
| `INPUT_DIRS`, `OUTPUT_ROOT` | Input folders and output folder (**must be filled in before running**) |
| `PIXEL_SIZE_UM` | µm per pixel (default 1.3811) |
| `MANUAL_MODE` | Prefer `_seg.npy` masks when present |
| `CHANNEL_SUFFIX` | Which suffix is target vs macrophage |
| `SEG_PARAMS` | CellPose `eval` parameters per channel (diameter, thresholds, …) |
| `FILTERS` | `min_area_um2`, `max_area_um2`, `min_brightness` per channel (`None` disables) |

**Usage**
```bash
python AutomatedCellposeSegmentation_WithManualMasksInput.py --input_dirs D:/data/run1 D:/data/run2 --output_root D:/out
```
Options: `--pixel_size_um`, `--model_type`, `--target_suffix`, `--macrophage_suffix`, `--manual` / `--no_manual`, `--no_sam`, `--no_settings`.

**Outputs** (in `OUTPUT_ROOT`)
```
per_image_summary_<timestamp>.csv    one row per FOV, target_* and macrophage_* columns
per_cell_detail_<timestamp>.csv      one row per segmented cell
run_settings_<timestamp>.txt         every parameter used
<folder>/target/{masks,overlays,outlines}/
<folder>/macrophage/{masks,overlays,outlines}/
<folder>/combined/<base>_combined.png   targets (red) + macrophages (cyan) on one image
```
The `target/masks/*_mask.tif` files are the input for the nearest-neighbour script.

---

## 2. `F480_CellTrace_ADCP_Analysis.py`

Classifies cells in macrophage–glioblastoma co-culture (e.g. ADCP) images using nuclei.

| Channel | Suffix | Role |
|---|---|---|
| CellTracker Green | `C1` | Macrophage label; can supply the macrophage mask |
| DAPI | `C2` | Nuclei (manual `_seg.npy` if present, else auto-segmented) |
| F4/80 | `C3` | Macrophage marker; can supply the macrophage mask |
| Far red | `C4` | Ignored |

**Macrophage masks** come from one hand-drawn CellPose GUI mask per FOV, on **either** F4/80 (`<base>_C3_seg.npy`) or CellTracker Green (`<base>_C1_seg.npy`).
- If both exist, `WHEN_BOTH_MASKS` picks the most recently saved (`"newest"`) or follows `MAC_MASK_PRIORITY` (`"priority"`).
- If neither exists, `IF_NO_MANUAL_MAC` either skips the FOV (`"skip"`, default) or auto-segments F4/80 (`"auto_f480"`).

**Classification logic**
1. A nucleus is **inside a macrophage** if ≥ `NUC_OVERLAP_FRAC` (default 0.5) of its area lies inside the macrophage masks.
2. **GBM cells** = nuclei not inside macrophages. Every nucleus of a multinucleated macrophage is excluded from the GBM count.
3. With `REQUIRE_NUCLEUS = True`, macrophage masks containing no nucleus are treated as fragments/debris and not counted, but still count as macrophage area for step 1.
4. Optional `CTG_CHECK` (off by default): a nucleus inside an F4/80 mask but with mean CellTracker signal below `CTG_CHECK_THRESHOLD` is reassigned to GBM, to catch F4/80 outlines that spread over neighbouring tumour cells.
5. Nuclei per macrophage are counted to measure multinucleation.

DAPI auto-segmentation uses a tiled background subtraction (`BG_BLOCK_PX`, `BG_PERCENTILE`) and can feed CellPose either the raw image or the background-subtracted one (`SEG_INPUT`).

**Usage**
```bash
python F480_CellTrace_ADCP_Analysis.py
python F480_CellTrace_ADCP_Analysis.py --max_fields 3              # quick test
python F480_CellTrace_ADCP_Analysis.py --name_filter Scene-22      # one field
python F480_CellTrace_ADCP_Analysis.py --prefer ctg --fallback auto_f480
```
Options: `--input_dirs`, `--output_root`, `--pixel_size_um`, `--nuc_overlap`, `--max_fields`, `--name_filter`, `--prefer {f480,ctg}`, `--fallback {skip,auto_f480}`, `--no_settings`.

**Outputs**
```
per_image_summary_<ts>.csv     one row per FOV: n_macrophages, n_GBM_cells_nuclei_outside_mac,
                               pct_cells_macrophages, ratio_GBM_to_macrophage,
                               mean_nuclei_per_macrophage, n_macrophages_multinucleated,
                               mac_mask_source, ...
per_cell_detail_<ts>.csv       one row per macrophage mask (n_nuclei, counted_as_macrophage, intensities)
per_nucleus_detail_<ts>.csv    one row per DAPI nucleus (class, frac_in_macrophage, mean_ctg, ...)
skipped_fields_<ts>.csv        FOVs left out and why (only if any)
run_settings_<ts>.txt
<folder>/{dapi,macrophage}/{overlays,outlines,masks}/
<folder>/classification/<base>_classification.png
```
The classification figure shows an RGB composite (R = F4/80, G = CellTracker, B = DAPI) next to nuclei coloured by class: yellow = in macrophage, cyan = GBM, magenta = CellTracker-negative under a mask. Macrophage outlines are red (F4/80 mask) or green (CellTracker mask); grey outlines are nucleus-free masks that were not counted.

---

## 3. `GBM_NearestNeighbors_Tiling.py`

Spatial analysis of the target masks written by script 1. It reads `MASK_ROOT/<folder>/target/masks/*_mask.tif`.

**Per FOV it computes:**
- **Nearest-neighbour distance (NND)** from each target cell's centroid to the nearest other target centroid: mean, median, SD, min, max, and the fraction of mutual nearest-neighbour pairs.
- **Comparison to random** (`COMPARE_TO_RANDOM`): the same number of cells is placed at random `NSIM` times in the same frame, keeping a minimum spacing (`MIN_SPACING_UM`, `"auto"` = smallest observed NND capped at the median cell diameter).
  - `nnd_ratio_vs_random < 1`: clustered; `≈ 1`: random; `> 1`: evenly spread.
  - Edge effects cancel in this ratio, so it is density-corrected. The raw mean NND is not, and grows whenever cell number drops.
  - `p_clustered` is a one-sided Monte Carlo p-value.
- **Patchiness** (`PATCHINESS`): the image is cut into square tiles of `TILE_SIZE_UM` (default 177 µm = 128 px), mask coverage is measured per tile, and `patchiness_cv` = SD / mean of tile coverage.
  - ~0.3: even coverage; > 1: islands with empty space between.
  - Not density-corrected; read it alongside `coverage_frac` and `nnd_ratio_vs_random`, and keep `TILE_SIZE_UM` the same across conditions you compare.

**Usage**
```bash
python GBM_NearestNeighbors_Tiling.py --mask_root D:/out
python GBM_NearestNeighbors_Tiling.py --mask_root D:/out --no_random --tile_size_um 200
```
Options: `--mask_root`, `--output_root` (default `<mask_root>/<OUTPUT_SUBFOLDER>`), `--pixel_size_um`, `--no_random`, `--tile_size_um`, `--no_patchiness`.

**Outputs**
```
per_fov_nnd_<ts>.csv           final measurements, one row per FOV
per_folder_summary_<ts>.csv    mean / median / SD of FOV values per folder
per_cell_nnd_<ts>.csv          one row per target cell
per_tile_coverage_<ts>.csv     one row per patchiness tile
nn_settings_<ts>.txt
<folder>/centroids_tif/        label-valued centroid dots, same size as the mask (overlays in Fiji)
<folder>/centroids_png/        outlines + centroids + cell labels
<folder>/nn_png/               each cell linked to its nearest neighbour, coloured by distance
<folder>/nnd_hist_png/         per-cell NND histogram and observed vs random mean NND
<folder>/patchiness_png/       tile grid with % coverage per tile
```
`cell_label` matches `cell_label` in script 1's `per_cell_detail` CSV, so the tables merge on `folder + base_name + cell_label`. Long Windows paths (>260 characters) are handled automatically.

---

## Typical workflow

```
1. (optional) Hand-correct masks in the CellPose GUI  ->  *_seg.npy next to the images
2. AutomatedCellposeSegmentation_WithManualMasksInput.py   ->  counts, areas, target masks
3. GBM_NearestNeighbors_Tiling.py  (MASK_ROOT = output of step 2)  ->  spatial metrics

   F480_CellTrace_ADCP_Analysis.py  runs independently on F4/80 / CellTracker / DAPI image sets
```

## Configuration and reproducibility

- Each script is configured in the `CONFIGURATION` block at the top; command-line options override those values for a single run.
- Set `INPUT_DIRS`, `OUTPUT_ROOT` / `MASK_ROOT` to your own paths before running.
- `PIXEL_SIZE_UM` must be the same in every script for a given dataset.
- Every run writes a timestamped settings file next to its CSVs, so results can always be traced back to the parameters that produced them. Output files are timestamped and never overwrite earlier runs.
