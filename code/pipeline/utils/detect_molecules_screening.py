#!/usr/bin/env python3
"""
Detect molecule bounding boxes in screening/optimization table page images.

For each page image containing a reaction condition screening table:
  1. Locate the reaction scheme region (between table title and table data)
  2. Detect molecular structure regions using OpenCV contour analysis
  3. Draw numbered bounding boxes on an annotated copy
  4. Save cropped molecule images for downstream external OCSR
  5. Output JSON index mapping box_id → bbox coordinates

The annotated images + JSON enable a two-stage pipeline:
  Stage 1: VLM matches molecular labels (1a, 2a, ...) to box IDs via annotated image
  Stage 2: an external OCSR component converts cropped molecules to SMILES
  → Final: label → box_id → SMILES

Output structure:
  mol_boxes_screening/
    {Journal}/{DOI_dir}/
      {page}_annotated.png          # Full page with numbered boxes
      {page}_mol_000.png            # Cropped molecule 0
      {page}_mol_001.png            # ...
    boxes_index.json                # Master index

Usage:
    # Quick test on a single image
    python scripts/detect_molecules_screening.py --single Angew/10.1002__anie.202418901/p6.png

    # Test on N images
    python scripts/detect_molecules_screening.py --test 10

    # Process one journal
    python scripts/detect_molecules_screening.py --journal ACSCatal

    # Resume from checkpoint
    python scripts/detect_molecules_screening.py --resume

    # Full run (all 1051 images)
    python scripts/detect_molecules_screening.py

    # Debug mode: save intermediate images
    python scripts/detect_molecules_screening.py --test 5 --debug
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_DIR = BASE_DIR / "screening_tables_qwen72b_2025"
DEFAULT_OUTPUT_DIR = BASE_DIR / "mol_boxes_screening"

# ── box drawing colors (BGR) ──────────────────────────────────────────────────
COLORS = [
    (0, 0, 255),      # red
    (0, 165, 255),    # orange
    (0, 220, 0),      # green
    (255, 180, 0),    # cyan-blue
    (255, 0, 0),      # blue
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (180, 0, 255),    # purple
    (0, 180, 180),    # olive
    (200, 100, 50),   # steel blue
    (100, 200, 0),    # lime
    (50, 50, 255),    # dark red
]

# ── reference thresholds at 300 DPI ──────────────────────────────────────────
# All pixel-based values are defined at 300 DPI and auto-scaled.
REF_DPI = 300

# Molecule region area thresholds
REF_MIN_AREA = 6000        # filter small text fragments (~80x75 min at 300 DPI)
REF_MAX_AREA = 500000

# Dimension thresholds
REF_MIN_DIM = 55           # min side length (both w and h must exceed this)
REF_MAX_DIM = 800          # max single side length

# Morphological ops — moderate: connect nearby bond segments within a molecule
REF_CLOSE_KERNEL = 5       # morph close kernel
REF_CLOSE_ITERS = 2        # morph close iterations
REF_DILATE_KERNEL = 3      # dilation kernel
REF_DILATE_ITERS = 1       # dilation iterations

# Box processing
REF_PADDING = 15           # pixel padding around each crop
REF_MERGE_DIST = 20        # merge boxes within this pixel distance
REF_ROW_TOLERANCE = 80     # vertical tolerance for spatial row grouping

# Filtering
REF_MAX_FILL = 0.15        # max fill ratio
REF_MAX_ASPECT = 4.5       # absolute max aspect ratio
REF_MAX_WIDTH_FRAC = 0.60  # skip boxes wider than this fraction of image width

# Table line removal
REF_LINE_KERNEL = 80       # morph open kernel for detecting long lines

# Reaction scheme region detection
# The scheme is typically between two "dense text bands" on the page.
# We look for sparse horizontal bands (low ink projection) that contain
# the reaction scheme, bounded by dense text above and table data below.
REF_DENSE_TEXT_THRESHOLD = 0.08   # row-level ink ratio above this → text row
REF_SPARSE_RUN_MIN = 50          # min consecutive sparse rows for a scheme band
REF_SCHEME_VERTICAL_PAD = 30     # extra padding above/below detected scheme region


def _detect_dpi(img_width: int) -> int:
    """Guess rendering DPI from image width."""
    return 600 if img_width > 4000 else 300


def _remove_long_lines(binary: np.ndarray, scale: float) -> np.ndarray:
    """Remove long horizontal and vertical lines (table rules) from binary image."""
    result = binary.copy()
    k_len = max(40, int(REF_LINE_KERNEL * scale))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    result = cv2.subtract(result, h_lines)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    result = cv2.subtract(result, v_lines)

    return result


def _find_scheme_regions(binary: np.ndarray, scale: float) -> list[tuple[int, int]]:
    """Find reaction scheme regions using horizontal projection analysis.

    Strategy: The page typically has dense text bands (paragraphs) and sparse
    bands (figures, reaction schemes, tables). Reaction scheme regions are
    sparse bands that contain structural drawings with a mix of short strokes
    at various angles, but NOT long runs of dense text.

    We look for runs of "sparse" rows (ink ratio below threshold) that are
    tall enough to contain a reaction scheme (at least ~50px at 300 DPI).

    Returns list of (y_start, y_end) tuples.
    """
    h, w = binary.shape
    # Smoothed horizontal projection (average ink ratio per row, smoothed)
    h_proj = np.sum(binary > 0, axis=1).astype(float) / w

    # Smooth the projection to avoid noise from individual characters
    kernel_size = max(3, int(10 * scale)) | 1
    h_proj_smooth = cv2.GaussianBlur(h_proj.reshape(-1, 1), (1, kernel_size), 0).flatten()

    # A row is "sparse" if its smoothed ink ratio is below the threshold
    # Dense text rows have high ratios; scheme areas have low ratios
    sparse_rows = h_proj_smooth < REF_DENSE_TEXT_THRESHOLD

    # Find contiguous runs of sparse rows
    min_run = int(REF_SPARSE_RUN_MIN * scale)
    pad = int(REF_SCHEME_VERTICAL_PAD * scale)

    regions = []
    in_run = False
    start = 0

    for y in range(h):
        if sparse_rows[y] and not in_run:
            start = y
            in_run = True
        elif not sparse_rows[y] and in_run:
            if y - start >= min_run:
                # Check that this sparse band actually has SOME ink
                # (completely blank areas are not schemes)
                band_ink = np.mean(h_proj[start:y])
                if band_ink > 0.001:
                    regions.append((max(0, start - pad), min(h, y + pad)))
            in_run = False

    if in_run and h - start >= min_run:
        band_ink = np.mean(h_proj[start:h])
        if band_ink > 0.001:
            regions.append((max(0, start - pad), min(h, h)))

    return regions


def _edge_angle_entropy(gray_region: np.ndarray) -> float:
    """Compute Shannon entropy of gradient angle distribution.

    Molecules have bonds at diverse angles → high entropy.
    Text has mostly horizontal strokes → lower entropy.
    """
    if gray_region.size == 0 or gray_region.shape[0] < 5 or gray_region.shape[1] < 5:
        return 0.0

    gx = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # Only consider pixels with significant gradient (edges)
    mag_thresh = np.percentile(mag, 80)
    mask = mag > max(mag_thresh, 10.0)

    if np.sum(mask) < 20:
        return 0.0

    angles = np.arctan2(gy[mask], gx[mask]) + np.pi
    counts, _ = np.histogram(angles, bins=12, range=(0, 2 * np.pi))
    counts = counts.astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    return entropy


def detect_molecules(image_path: str, debug_dir: str = None) -> list[dict]:
    """Detect molecule structure regions in a screening table page image.

    Strategy:
      1. Adaptive threshold + remove table lines
      2. Find sparse bands (reaction scheme regions) via horizontal projection
      3. Within each sparse band, run contour detection with morphological ops
      4. Filter by area, dimension, aspect ratio, fill ratio
      5. Merge nearby boxes, spatial sort, assign IDs

    Returns list of dicts with keys: mol_id, bbox, _fill, _entropy.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dpi = _detect_dpi(w)
    s = dpi / REF_DPI

    # Scale thresholds
    min_area = int(REF_MIN_AREA * s * s)
    max_area = int(REF_MAX_AREA * s * s)
    min_dim = int(REF_MIN_DIM * s)
    max_dim = int(REF_MAX_DIM * s)
    padding = int(REF_PADDING * s)
    merge_dist = int(REF_MERGE_DIST * s)
    row_tol = int(REF_ROW_TOLERANCE * s)
    close_k = max(3, int(REF_CLOSE_KERNEL * s)) | 1
    dilate_k = max(3, int(REF_DILATE_KERNEL * s)) | 1
    block_size = max(3, int(51 * s)) | 1

    # ── Step 1: Adaptive threshold ────────────────────────────────────────────
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, 15
    )

    # ── Step 2: Remove long lines (table rules) ──────────────────────────────
    binary_clean = _remove_long_lines(binary, s)

    # ── Step 3: Find reaction scheme regions ──────────────────────────────────
    scheme_regions = _find_scheme_regions(binary_clean, s)

    # If no scheme regions found, search the full image as fallback
    if not scheme_regions:
        scheme_regions = [(0, h)]

    # ── Step 4: Morphological ops ─────────────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    closed = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_close,
                               iterations=REF_CLOSE_ITERS)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=REF_DILATE_ITERS)

    # ── Step 5: Find contours within scheme regions ───────────────────────────
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        x, y_cnt, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        cy = y_cnt + bh / 2

        # Check if contour center falls within a scheme region
        in_scheme = any(sy1 <= cy <= sy2 for sy1, sy2 in scheme_regions)
        if not in_scheme:
            continue

        if area < min_area or area > max_area:
            continue
        if bw > w * REF_MAX_WIDTH_FRAC:
            continue
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if aspect > REF_MAX_ASPECT:
            continue
        if bw < min_dim or bh < min_dim:
            continue

        raw_boxes.append([x, y_cnt, x + bw, y_cnt + bh])

    if not raw_boxes:
        if debug_dir:
            _save_debug(debug_dir, image_path, binary, binary_clean,
                        closed, dilated, [], [], img, scheme_regions)
        return []

    # ── Step 6: Merge nearby boxes ────────────────────────────────────────────
    merged = _merge_boxes(raw_boxes, merge_dist)

    # ── Step 7: Post-merge filtering ──────────────────────────────────────────
    results = []
    for (x1, y1, x2, y2) in merged:
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w, x2 + padding)
        py2 = min(h, y2 + padding)
        bw, bh = px2 - px1, py2 - py1

        if bw < min_dim or bh < min_dim:
            continue
        if bw > max_dim and bh > max_dim:
            continue

        box_area = bw * bh
        if box_area < min_area or box_area > max_area:
            continue

        # Fill ratio filter
        region = binary[py1:py2, px1:px2]
        fill = np.sum(region > 0) / (bw * bh + 1)
        if fill > REF_MAX_FILL:
            continue

        # Edge angle entropy (informational, kept for downstream use)
        gray_crop = gray[py1:py2, px1:px2]
        entropy = _edge_angle_entropy(gray_crop)

        results.append({
            "bbox": [px1, py1, px2, py2],
            "_fill": round(fill, 4),
            "_entropy": round(entropy, 3),
        })

    # ── Step 8: Spatial sort ──────────────────────────────────────────────────
    results = _spatial_sort(results, row_tolerance=row_tol)

    # ── Step 9: Assign mol IDs ────────────────────────────────────────────────
    for idx, det in enumerate(results):
        det["mol_id"] = f"mol_{idx:03d}"

    if debug_dir:
        _save_debug(debug_dir, image_path, binary, binary_clean,
                    closed, dilated, raw_boxes, results, img, scheme_regions)

    return results


# ── box utilities ─────────────────────────────────────────────────────────────

def _merge_boxes(boxes: list, gap: int) -> list:
    """Iteratively merge bounding boxes that overlap or are within `gap` pixels."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            bx = list(boxes[i])
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if _boxes_close(bx, boxes[j], gap):
                    bx = [min(bx[0], boxes[j][0]),
                          min(bx[1], boxes[j][1]),
                          max(bx[2], boxes[j][2]),
                          max(bx[3], boxes[j][3])]
                    used[j] = True
                    changed = True
            merged.append(bx)
        boxes = merged
    return boxes


def _boxes_close(a, b, gap):
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or
                a[3] + gap < b[1] or b[3] + gap < a[1])


def _spatial_sort(detections: list, row_tolerance: int = 100) -> list:
    """Sort detections: group into rows by y-center, sort each row by x."""
    if not detections:
        return detections
    dets = sorted(detections, key=lambda d: (d["bbox"][1] + d["bbox"][3]) / 2)
    rows = []
    current_row = [dets[0]]
    for det in dets[1:]:
        cy = (det["bbox"][1] + det["bbox"][3]) / 2
        prev_cy = (current_row[-1]["bbox"][1] + current_row[-1]["bbox"][3]) / 2
        if abs(cy - prev_cy) < row_tolerance:
            current_row.append(det)
        else:
            rows.append(current_row)
            current_row = [det]
    rows.append(current_row)
    result = []
    for row in rows:
        row.sort(key=lambda d: d["bbox"][0])
        result.extend(row)
    return result


# ── visualization ─────────────────────────────────────────────────────────────

def draw_annotated_image(image_path: str, detections: list) -> np.ndarray | None:
    """Draw numbered bounding boxes on image and return annotated copy."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    s = _detect_dpi(w) / REF_DPI

    line_w = max(2, int(3 * s))
    font_scale = max(0.7, 1.0 * s)
    font_w = max(1, int(2 * s))
    label_pad = int(6 * s)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        mol_id = det["mol_id"]
        idx = int(mol_id.split("_")[-1])
        color = (0, 0, 255)  # always red

        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_w)

        label = mol_id
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_w)
        ly = y1 - label_pad
        if ly - th - label_pad < 0:
            ly = y2 + th + 2 * label_pad

        cv2.rectangle(img,
                      (x1, ly - th - label_pad),
                      (x1 + tw + 2 * label_pad, ly + label_pad),
                      color, -1)
        cv2.putText(img, label, (x1 + label_pad, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_w, cv2.LINE_AA)

    return img


def _save_debug(debug_dir, image_path, binary, binary_clean,
                closed, dilated, raw_boxes, results, img, scheme_regions=None):
    """Save intermediate images for parameter tuning."""
    stem = Path(image_path).stem
    ddir = Path(debug_dir)
    ddir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(ddir / f"{stem}_1_binary.png"), binary)
    cv2.imwrite(str(ddir / f"{stem}_2_lines_removed.png"), binary_clean)
    cv2.imwrite(str(ddir / f"{stem}_3_closed.png"), closed)
    cv2.imwrite(str(ddir / f"{stem}_4_dilated.png"), dilated)

    dbg = img.copy()
    h, w = dbg.shape[:2]
    s = _detect_dpi(w) / REF_DPI

    # Draw scheme regions as green overlay
    if scheme_regions:
        overlay = dbg.copy()
        for (sy1, sy2) in scheme_regions:
            cv2.rectangle(overlay, (0, sy1), (w, sy2), (0, 200, 0), -1)
        cv2.addWeighted(overlay, 0.15, dbg, 0.85, 0, dbg)
        for (sy1, sy2) in scheme_regions:
            cv2.rectangle(dbg, (0, sy1), (w, sy2), (0, 200, 0), 2)
            cv2.putText(dbg, "scheme_region", (10, sy1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 200, 0), 2)

    # Draw raw boxes in blue
    for box in raw_boxes:
        if isinstance(box, dict):
            x1, y1, x2, y2 = box["bbox"]
        else:
            x1, y1, x2, y2 = box
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw final boxes in red with metrics
    font_scale = max(0.4, 0.6 * s)
    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 3)
        entropy = det.get("_entropy", 0)
        fill = det.get("_fill", 0)
        mol_id = det.get("mol_id", "")
        label = f"{mol_id} e={entropy:.2f} f={fill:.3f}"
        cv2.putText(dbg, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

    cv2.imwrite(str(ddir / f"{stem}_5_boxes.png"), dbg)


# ── per-image processing ─────────────────────────────────────────────────────

def process_one_image(args_tuple):
    """Process a single screening table image. Called by process pool."""
    image_path, rel_key, output_dir, debug = args_tuple

    if not os.path.exists(image_path):
        return rel_key, [], f"Not found: {image_path}"

    debug_dir = None
    if debug:
        parts = Path(rel_key).parts
        journal = parts[0] if len(parts) >= 3 else "unknown"
        doi_dir = parts[1] if len(parts) >= 3 else "unknown"
        debug_dir = str(output_dir / "_debug" / journal / doi_dir)

    detections = detect_molecules(image_path, debug_dir=debug_dir)

    if not detections:
        return rel_key, [], None

    parts = Path(rel_key).parts
    journal = parts[0] if len(parts) >= 3 else "unknown"
    doi_dir = parts[1] if len(parts) >= 3 else "unknown"
    page_stem = Path(rel_key).stem

    out_sub = output_dir / journal / doi_dir
    out_sub.mkdir(parents=True, exist_ok=True)

    annotated = draw_annotated_image(image_path, detections)
    if annotated is not None:
        cv2.imwrite(str(out_sub / f"{page_stem}_annotated.png"), annotated)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    results = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        mol_id = det["mol_id"]

        crop = img[y1:y2, x1:x2]
        crop_rel = f"{journal}/{doi_dir}/{page_stem}_{mol_id}.png"
        cv2.imwrite(str(out_sub / f"{page_stem}_{mol_id}.png"), crop)

        results.append({
            "mol_id": mol_id,
            "bbox": [x1, y1, x2, y2],
            "bbox_normalized": [
                round(x1 / w, 4), round(y1 / h, 4),
                round(x2 / w, 4), round(y2 / h, 4),
            ],
            "crop_image": crop_rel,
            "source_image": rel_key,
            "source_size": [w, h],
            "angle_entropy": det.get("_entropy", 0),
            "fill_ratio": det.get("_fill", 0),
        })

    return rel_key, results, None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect molecule bounding boxes in screening table images")
    parser.add_argument("--images-dir", type=str, default=str(DEFAULT_IMAGES_DIR),
                        help="Input directory with screening table page images")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for annotated images, crops, index")
    parser.add_argument("--test", type=int, default=0,
                        help="Process only first N images for testing")
    parser.add_argument("--journal", type=str, default=None,
                        help="Filter by journal (e.g. ACSCatal, JACS, Angew)")
    parser.add_argument("--single", type=str, default=None,
                        help="Process single image (relative path within images-dir)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed images")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate images (binary, contours, etc.)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    if args.single:
        single_path = images_dir / args.single
        if not single_path.exists():
            print(f"ERROR: {single_path} not found")
            return
        all_images = [single_path]
    else:
        all_images = sorted(images_dir.rglob("*.png"))

    if args.journal and not args.single:
        all_images = [p for p in all_images
                      if p.relative_to(images_dir).parts[0] == args.journal]

    tasks = []
    for img_path in all_images:
        rel_key = str(img_path.relative_to(images_dir))
        tasks.append((str(img_path), rel_key, output_dir, args.debug))

    # Load index
    index_path = output_dir / "boxes_index.json"
    if args.resume and index_path.exists():
        with open(index_path) as f:
            boxes_index = json.load(f)
    else:
        boxes_index = {}

    # Resume: skip done
    if args.resume:
        done_keys = set()
        for doi_key, pages in boxes_index.items():
            for page_key, page_data in pages.items():
                src = page_data.get("source_image", "")
                if src:
                    done_keys.add(src)

        before = len(tasks)
        tasks = [t for t in tasks if t[1] not in done_keys]
        skipped = before - len(tasks)
        if skipped:
            print(f"Resume: skipping {skipped} already-processed images")

    if args.test > 0:
        tasks = tasks[:args.test]

    total = len(tasks)
    print(f"Molecule detection for screening tables")
    print(f"  Input:  {images_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Images: {total}")
    print(f"  Debug:  {args.debug}")
    print("=" * 60)

    if not tasks:
        print("Nothing to process!")
        return

    t0 = time.time()
    total_mols = 0
    errors = 0
    empty = 0

    n_workers = 1 if args.debug else args.workers

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_one_image, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            rel_key, results, error = future.result()

            if error:
                print(f"  [{i}/{total}] ERROR {rel_key}: {error}")
                errors += 1
                continue

            if not results:
                empty += 1
                print(f"  [{i}/{total}] {rel_key}: 0 molecules (empty)")
                continue

            parts = Path(rel_key).parts
            journal = parts[0] if len(parts) >= 3 else "unknown"
            doi_dir = parts[1] if len(parts) >= 3 else "unknown"
            page_stem = Path(rel_key).stem
            doi_key = f"{journal}/{doi_dir}"

            if doi_key not in boxes_index:
                boxes_index[doi_key] = {}

            annot_rel = f"{journal}/{doi_dir}/{page_stem}_annotated.png"
            boxes_index[doi_key][page_stem] = {
                "source_image": rel_key,
                "annotated_image": annot_rel,
                "num_molecules": len(results),
                "molecules": results,
            }

            total_mols += len(results)
            print(f"  [{i}/{total}] {rel_key}: {len(results)} molecules")

            if i % 50 == 0:
                with open(index_path, "w") as f:
                    json.dump(boxes_index, f, ensure_ascii=False, indent=2)

    with open(index_path, "w") as f:
        json.dump(boxes_index, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Done! {total_mols} molecules from {total} images")
    print(f"  Empty (no detections): {empty}")
    print(f"  Errors: {errors}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Index: {index_path}")


if __name__ == "__main__":
    main()
