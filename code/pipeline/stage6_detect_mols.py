#!/usr/bin/env python3
"""
Batch annotate screening table images with DECIMER-Segmentation molecule bboxes.

For each page image, runs Mask R-CNN to detect molecules and draws red bounding
boxes with mol_id labels on the original image. No cropping.

Output structure mirrors input:
  {output_dir}/{Journal}/{DOI_dir}/{page}_annotated.png

Usage:
    # Full run
    python scripts/decimer_batch_annotate.py

    # Test on first N images
    python scripts/decimer_batch_annotate.py --test 10

    # Single journal
    python scripts/decimer_batch_annotate.py --journal JACS

    # Resume (skip already-processed)
    python scripts/decimer_batch_annotate.py --resume
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_DIR = BASE_DIR / "screening_tables_si"
DEFAULT_OUTPUT_DIR = BASE_DIR / "mol_boxes_decimer_si"


def load_decimer_model():
    """Load DECIMER model once."""
    from decimer_segmentation.decimer_segmentation import get_model
    model = get_model()
    return model


def detect_and_annotate(model, image_path: str) -> tuple:
    """Run DECIMER detection and draw red boxes on image.

    Returns:
        annotated image (np.ndarray or None), list of detection dicts
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, []

    # DECIMER expects BGR (cv2 default)
    results = model.detect([img], verbose=0)
    scores = results[0]["scores"]
    bboxes = results[0]["rois"]  # (y0, x0, y1, x1)

    if len(bboxes) == 0:
        return img, []

    h, w = img.shape[:2]
    s = 2.0 if w > 4000 else 1.0

    annotated = img.copy()
    detections = []

    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        y0, x0, y1, x1 = bbox
        # Expand bbox by padding to avoid clipping molecule edges
        box_pad = max(5, int(8 * s))
        y0 = max(0, y0 - box_pad)
        x0 = max(0, x0 - box_pad)
        y1 = min(h, y1 + box_pad)
        x1 = min(w, x1 + box_pad)
        color = (0, 0, 255)  # red

        line_w = max(1, int(1.5 * s))
        font_scale = max(0.8, 1.0 * s)
        font_w = max(2, int(2.5 * s))

        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, line_w)

        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_w)
        pad = int(4 * s)
        ly = y0 - pad
        if ly - th - pad < 0:
            ly = y1 + th + 2 * pad

        cv2.rectangle(annotated,
                      (x0, ly - th - pad),
                      (x0 + tw + 2 * pad, ly + pad),
                      color, -1)
        cv2.putText(annotated, label, (x0 + pad, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_w, cv2.LINE_AA)

        detections.append({
            "mol_id": label,
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "bbox_normalized": [
                round(x0 / w, 4), round(y0 / h, 4),
                round(x1 / w, 4), round(y1 / h, 4),
            ],
            "score": round(float(score), 4),
            "source_size": [w, h],
        })

    return annotated, detections


def main():
    parser = argparse.ArgumentParser(
        description="Batch annotate images with DECIMER molecule detection")
    parser.add_argument("--images-dir", type=str, default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test", type=int, default=0,
                        help="Process only first N images")
    parser.add_argument("--journal", type=str, default=None,
                        help="Filter by journal")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip already-processed images")
    parser.add_argument("--split", type=str, default=None,
                        help="分片: 0/8 表示取第1份, 1/8 表示取第2份, etc.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    all_images = sorted(images_dir.rglob("*.png"))
    if args.journal:
        all_images = [p for p in all_images
                      if p.relative_to(images_dir).parts[0] == args.journal]

    tasks = []
    for img_path in all_images:
        rel = img_path.relative_to(images_dir)
        parts = rel.parts
        if len(parts) < 3:
            continue
        journal, doi_dir, page_file = parts[0], parts[1], rel.stem
        out_sub = output_dir / journal / doi_dir
        out_file = out_sub / f"{page_file}_annotated.png"
        if args.resume and out_file.exists():
            continue
        tasks.append((str(img_path), str(rel), str(out_sub), str(out_file)))

    if args.test > 0:
        tasks = tasks[:args.test]

    if args.split:
        idx, parts = args.split.split("/")
        idx, parts = int(idx), int(parts)
        chunk_size = (len(tasks) + parts - 1) // parts
        tasks = tasks[idx * chunk_size : (idx + 1) * chunk_size]

    total = len(tasks)
    print(f"DECIMER batch annotation")
    print(f"  Input:  {images_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Images: {total}")
    print("=" * 60)

    if not tasks:
        print("Nothing to process!")
        return

    # Load model once
    print("Loading DECIMER model...")
    t0 = time.time()
    model = load_decimer_model()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load or init index
    index_path = output_dir / "boxes_index.json"
    if args.resume and index_path.exists():
        with open(index_path) as f:
            boxes_index = json.load(f)
    else:
        boxes_index = {}

    t_start = time.time()
    total_mols = 0
    empty = 0
    errors = 0

    for i, (img_path, rel_key, out_sub_str, out_file_str) in enumerate(tasks, 1):
        t1 = time.time()
        try:
            annotated, detections = detect_and_annotate(model, img_path)
        except Exception as e:
            print(f"  [{i}/{total}] ERROR {rel_key}: {e}")
            errors += 1
            continue

        if annotated is None:
            print(f"  [{i}/{total}] ERROR {rel_key}: could not read image")
            errors += 1
            continue

        if not detections:
            empty += 1
            dt = time.time() - t1
            print(f"  [{i}/{total}] {rel_key}: 0 molecules ({dt:.1f}s)")
            continue

        # Save annotated image
        Path(out_sub_str).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_file_str, annotated)

        # Update index
        parts = Path(rel_key).parts
        journal, doi_dir, page_file = parts[0], parts[1], Path(rel_key).stem
        doi_key = f"{journal}/{doi_dir}"
        if doi_key not in boxes_index:
            boxes_index[doi_key] = {}
        annot_rel = f"{journal}/{doi_dir}/{page_file}_annotated.png"
        boxes_index[doi_key][page_file] = {
            "source_image": rel_key,
            "annotated_image": annot_rel,
            "num_molecules": len(detections),
            "molecules": detections,
        }

        total_mols += len(detections)
        dt = time.time() - t1
        elapsed = time.time() - t_start
        avg = elapsed / i
        eta = avg * (total - i)
        print(f"  [{i}/{total}] {rel_key}: {len(detections)} molecules "
              f"({dt:.1f}s, ETA {eta/60:.0f}min)")

        # Checkpoint every 20 images
        if i % 20 == 0:
            with open(index_path, "w") as f:
                json.dump(boxes_index, f, ensure_ascii=False, indent=2)

    # Final save
    with open(index_path, "w") as f:
        json.dump(boxes_index, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"Done! {total_mols} molecules from {total} images")
    print(f"  Empty: {empty}")
    print(f"  Errors: {errors}")
    print(f"  Elapsed: {elapsed/60:.1f}min")
    print(f"  Index: {index_path}")


if __name__ == "__main__":
    main()
