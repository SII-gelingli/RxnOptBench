#!/usr/bin/env python3
"""
Render all pages from a journal-organized PDF directory at 600 DPI.

Output: all_pages_hires_2025/{Journal}/{doi_safe}/p{page_num}.png

Usage:
    python scripts/render_all_2025.py                    # Full run
    python scripts/render_all_2025.py --test 5           # 5 PDFs per journal
    python scripts/render_all_2025.py --journal JACS     # Single journal
    python scripts/render_all_2025.py --workers 16       # More parallel workers
    python scripts/render_all_2025.py --dpi 600          # DPI (default 600)
    python scripts/render_all_2025.py --resume           # Skip existing images
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF


BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers_2025"
OUTPUT_DIR = BASE_DIR / "all_pages_hires_2025"
DEFAULT_DPI = 600


def render_page(args_tuple):
    """Render a single page from PDF at target DPI."""
    pdf_path, journal, doi_safe, page_idx, output_dir, dpi = args_tuple

    img_dir = output_dir / journal / doi_safe
    img_path = img_dir / f"p{page_idx + 1}.png"

    if img_path.exists():
        return {"status": "skipped"}

    try:
        doc = fitz.open(str(pdf_path))
        if page_idx >= len(doc):
            doc.close()
            return {"status": "error", "error": f"Page {page_idx} out of range"}
        page = doc[page_idx]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        os.makedirs(img_dir, exist_ok=True)
        pix.save(str(img_path))
        doc.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Render all pages from a journal-organized PDF directory"
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--test", type=int, default=0, help="N PDFs per journal")
    parser.add_argument("--journal", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Skip existing images")
    parser.add_argument("--papers-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir) if args.papers_dir else PAPERS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (pdf, page) tasks
    tasks = []
    journals = sorted(
        d.name for d in papers_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    )

    if args.journal:
        journals = [j for j in journals if j == args.journal]

    for journal in journals:
        j_dir = papers_dir / journal
        pdfs = sorted(f for f in os.listdir(j_dir) if f.endswith(".pdf"))
        if args.test > 0:
            pdfs = pdfs[:args.test]

        for pdf_name in pdfs:
            pdf_path = j_dir / pdf_name
            doi_safe = pdf_name.replace(".pdf", "")
            try:
                doc = fitz.open(str(pdf_path))
                n_pages = len(doc)
                doc.close()
            except Exception:
                continue

            for page_idx in range(n_pages):
                if args.resume:
                    img_path = output_dir / journal / doi_safe / f"p{page_idx + 1}.png"
                    if img_path.exists():
                        continue
                tasks.append((str(pdf_path), journal, doi_safe, page_idx, output_dir, args.dpi))

    total = len(tasks)
    total_pdfs = len(set(t[0] for t in tasks))
    print(f"Rendering {total} pages from {total_pdfs} PDFs at {args.dpi} DPI")
    print(f"  Journals: {', '.join(journals)}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {output_dir}")
    print()

    if total == 0:
        print("Nothing to render.")
        return

    stats = {"ok": 0, "skipped": 0, "error": 0}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(render_page, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            stats[result["status"]] = stats.get(result["status"], 0) + 1
            if result["status"] == "error":
                print(f"  [ERROR] {result.get('error', '')}")
            if i % 500 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{total}] {elapsed:.0f}s | {rate:.1f} pg/s | "
                      f"ok:{stats['ok']} skip:{stats['skipped']} err:{stats['error']}")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Rendered: {stats['ok']} new, {stats['skipped']} skipped, {stats['error']} errors")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
