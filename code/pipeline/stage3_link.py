#!/usr/bin/env python3
"""Create symlinks for pages identified as condition screening tables.

Usage:
    python pipeline/stage3_link.py --source si \
      --results-dir scope_screen_results_2025_si \
      --images-dir all_si_pages_hires_2025 \
      --output-dir screening_tables_si --no-exclude
"""

import argparse
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Journals with IF < 10
LOW_IF_JOURNALS = {"OrgLett", "ChemSci", "JACSAu", "GreenChem"}


def main():
    parser = argparse.ArgumentParser(description="Create symlinks for screening table pages")
    parser.add_argument("--source", required=True,
                        help="Suffix of results dir, e.g. 'qwen72b' -> scope_screen_results_2025_qwen72b/")
    parser.add_argument("--images-dir", default="all_pages_hires_2025",
                        help="Name of images directory (default: all_pages_hires_2025)")
    parser.add_argument("--results-dir", default=None,
                        help="Explicit Stage 2 results directory")
    parser.add_argument("--output-dir", default=None,
                        help="Explicit linked-image output directory")
    parser.add_argument("--no-exclude", action="store_true",
                        help="Do not exclude low-IF journals")
    args = parser.parse_args()

    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else BASE / f"scope_screen_results_2025_{args.source}"
    )
    if not results_dir.is_absolute():
        results_dir = BASE / results_dir
    images_dir = BASE / args.images_dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else BASE / f"screening_tables_{args.source}"
    )
    if not output_dir.is_absolute():
        output_dir = BASE / output_dir
    exclude = set() if args.no_exclude else LOW_IF_JOURNALS

    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found")
        return

    count = 0
    missing = 0
    skipped_low_if = 0

    for jf in sorted(results_dir.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)

        if not data.get("is_screening_table"):
            continue

        src = data.get("_source", {})
        journal = src.get("journal", "")
        doi_safe = src.get("doi_safe", "")
        page = src.get("page")

        if not (journal and doi_safe and page is not None):
            name = jf.stem
            parts = name.rsplit("_p", 1)
            if len(parts) == 2:
                page = int(parts[1])
                rest = parts[0]
                idx = rest.find("_10.")
                if idx > 0:
                    journal = rest[:idx]
                    doi_safe = rest[idx + 1:]
                else:
                    print(f"  SKIP (cannot parse): {jf.name}")
                    continue
            else:
                print(f"  SKIP (cannot parse): {jf.name}")
                continue

        if journal in exclude:
            skipped_low_if += 1
            continue

        img_path = images_dir / journal / doi_safe / f"p{page}.png"
        if not img_path.exists():
            print(f"  MISSING image: {img_path}")
            missing += 1
            continue

        link_dir = output_dir / journal / doi_safe
        link_dir.mkdir(parents=True, exist_ok=True)
        link_path = link_dir / f"p{page}.png"

        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        os.symlink(str(img_path), str(link_path))
        count += 1

    print(f"\nDone: {count} symlinks created in {output_dir}")
    if skipped_low_if:
        print(f"  ({skipped_low_if} skipped from low-IF journals: {exclude})")
    if missing:
        print(f"  ({missing} missing source images)")


if __name__ == "__main__":
    main()
