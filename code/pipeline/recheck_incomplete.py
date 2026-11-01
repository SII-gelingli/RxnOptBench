#!/usr/bin/env python3
"""
Re-scan screening_tables_recognized/ with updated incompleteness rules,
update is_incomplete flags in-place, and refresh _incomplete_tables/ index.

Usage:
    python3 pipeline/recheck_incomplete.py
    python3 pipeline/recheck_incomplete.py --dry-run
    python3 pipeline/recheck_incomplete.py --journal JACS
"""

import argparse
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RECOGNIZED_DIR = BASE_DIR / "screening_tables_recognized"


def check_incomplete(table: dict) -> tuple[bool, str]:
    """Apply post-validation rules to detect incomplete tables.

    Returns (is_incomplete, reason).
    """
    footnotes = table.get("footnotes", {})
    caption = table.get("caption", "")
    rows = table.get("rows", [])
    columns = table.get("columns", [])

    # Already marked by model
    if table.get("is_incomplete"):
        return True, table.get("incomplete_reason", "marked by model")

    # Check 1: footnote keys with empty values
    if footnotes and all(not str(v).strip() for v in footnotes.values()):
        return True, (
            f"footnote markers {','.join(footnotes.keys())} present "
            f"but text not found on this page (auto-detected)"
        )

    # Check 2: no caption and no footnotes
    if not caption.strip() and not footnotes:
        if rows and columns:
            return True, "no caption/footnotes below table (auto-detected, likely on next page)"

    # Check 3: first entry doesn't start from 1
    if rows:
        entry_col = None
        for col in columns:
            if re.match(r'^(entry|no\.?|run|#|\s*$)', col.lower().strip()):
                entry_col = col
                break
        if entry_col:
            first_val = str(rows[0].get(entry_col, "")).strip().rstrip("abcdefghij*†‡§")
            if first_val.isdigit() and int(first_val) > 1:
                return True, f"first entry is {first_val}, earlier rows likely on previous page (auto-detected)"

    return False, ""


def main():
    parser = argparse.ArgumentParser(description="Re-check incomplete tables with updated rules")
    parser.add_argument("--recognized-dir", type=str, default=str(DEFAULT_RECOGNIZED_DIR))
    parser.add_argument("--journal", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't modify files")
    args = parser.parse_args()

    recognized_dir = Path(args.recognized_dir)
    incomplete_dir = recognized_dir / "_incomplete_tables"

    # Collect all JSON files
    all_files = sorted(recognized_dir.rglob("*.json"))
    all_files = [f for f in all_files
                 if not f.name.startswith("_")
                 and "_debug" not in str(f)
                 and "_incomplete" not in str(f)
                 and "_progress" not in f.name]

    if args.journal:
        journals = args.journal.split(",")
        all_files = [f for f in all_files if f.parent.name in journals]

    print(f"Re-checking incomplete tables {'(DRY-RUN)' if args.dry_run else '(APPLY)'}")
    print(f"  Input: {recognized_dir}")
    print(f"  Files: {len(all_files)}")
    print("=" * 60)

    total_tables = 0
    already_incomplete = 0
    newly_incomplete = 0
    complete = 0
    modified_files = 0

    newly_found = []

    for jf in all_files:
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            data = [data]

        file_modified = False
        for table in data:
            if not isinstance(table, dict):
                continue
            total_tables += 1

            was_incomplete = table.get("is_incomplete", False)
            is_incomplete, reason = check_incomplete(table)

            if is_incomplete and not was_incomplete:
                newly_incomplete += 1
                newly_found.append({
                    "file": str(jf.relative_to(recognized_dir)),
                    "page": table.get("page", ""),
                    "title": table.get("table_title", "")[:80],
                    "reason": reason,
                    "rows": len(table.get("rows", [])),
                })
                table["is_incomplete"] = True
                table["incomplete_reason"] = reason
                file_modified = True
            elif was_incomplete:
                already_incomplete += 1
            else:
                complete += 1

        if file_modified and not args.dry_run:
            jf.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            modified_files += 1

            # Also write to _incomplete_tables/ — one file per page
            journal = jf.parent.name
            doi_stem = jf.stem  # e.g. 10.1021__jacs.4c13723_si
            inc_tables = [t for t in data if t.get("is_incomplete")]
            # Group by page
            from collections import defaultdict
            by_page = defaultdict(list)
            for t in inc_tables:
                page = t.get("page", "unknown")
                by_page[page].append(t)
            for page, tables in by_page.items():
                page_clean = page.replace(".png", "")
                inc_out_dir = incomplete_dir / journal / doi_stem
                inc_out_dir.mkdir(parents=True, exist_ok=True)
                out_data = {
                    "image": f"{journal}/{doi_stem}/{page}",
                    "page": page,
                    "tables": tables,
                }
                (inc_out_dir / f"{page_clean}.png.json").write_text(
                    json.dumps(out_data, ensure_ascii=False, indent=2))

    print(f"\nResults:")
    print(f"  Total tables:        {total_tables}")
    print(f"  Already incomplete:  {already_incomplete}")
    print(f"  Newly detected:      {newly_incomplete}")
    print(f"  Complete:            {complete}")
    if not args.dry_run:
        print(f"  Files modified:      {modified_files}")

    if newly_found:
        print(f"\nNewly detected incomplete tables:")
        for item in newly_found:
            print(f"  {item['file']} p{item['page']}: {item['title']}")
            print(f"    reason: {item['reason']} ({item['rows']} rows)")


if __name__ == "__main__":
    main()
