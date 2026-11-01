#!/usr/bin/env python3
"""
Stage 1: Text-based pre-screening for optimization tables in 2025 papers.

Auto-detects all journal subdirectories under papers_2025/.
Scans PDFs using keyword/regex matching to identify candidate pages.
Outputs optimization_candidates_2025.json for Stage 2 (VLM confirmation).

Scoring tiers:
  1.0  - Table caption with optimization/screening/evaluation keywords
  0.9  - Section header like "Reaction Optimization"
  0.8  - Main-paper Table 1 (commonly optimization table)
  0.7  - Page has >=3 tabular rows + optimization keywords
  0.5  - Table reference + multiple optimization keywords
  0.4  - Adjacent page to a high-confidence hit (tables may span pages)

Usage:
    python scripts/screen_stage1_2025.py                   # Full run
    python scripts/screen_stage1_2025.py --test 5          # 5 PDFs per journal
    python scripts/screen_stage1_2025.py --journal JACS    # Single journal
    python scripts/screen_stage1_2025.py --threshold 0.4   # Lower threshold
    python scripts/screen_stage1_2025.py --resume          # Merge with existing results
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers_2025"
SI_DIR = BASE_DIR / "supporting_info_2025"
METADATA_DIR = BASE_DIR / "metadata_2025"
OUTPUT_FILE = BASE_DIR / "optimization_candidates_2025.json"

# ── Patterns ───────────────────────────────────────────────────────────────────

# Table caption: "Table 1. Optimization of ..."
RE_TABLE_CAPTION = re.compile(
    r"^Table\s+(\d+)[.:\s]+(.*)", re.IGNORECASE
)

# Section headers that indicate optimization
RE_OPT_SECTION = re.compile(
    r"^(?:reaction\s+optimization|optimization\s+of|"
    r"condition\s+optimization|screening\s+of\s+(?:reaction\s+)?condition|"
    r"optimization\s+stud|optimiz\w+\s+(?:the\s+)?reaction\s+condition|"
    r"survey\s+of\s+(?:reaction\s+)?condition|"
    r"evaluation\s+of\s+(?:reaction\s+)?condition)",
    re.IGNORECASE,
)

# Keywords in table captions / nearby text
OPT_CAPTION_KEYWORDS = [
    "optimization", "optimisation", "screening", "evaluation of",
    "effect of", "survey of", "investigation of",
    "control experiment", "condition screening",
]

# Broader optimization keywords for body text
OPT_BODY_KEYWORDS = [
    "optimization", "optimisation", "screening", "optimal",
    "best condition", "optimal condition", "reaction condition",
    "catalyst loading", "equiv", "equivalents",
    "solvent effect", "temperature effect", "ligand effect",
    "base effect", "additive effect",
]

# Tabular row pattern: entry-number then values separated by whitespace
RE_TABLE_ROW = re.compile(
    r"^\s*(\d{1,2})\s+\S+.*\s+\d+\s*%?\s*$"
)

# Table reference in text
RE_TABLE_REF = re.compile(r"Table\s+(\d+)", re.IGNORECASE)

# Yield pattern
RE_YIELD = re.compile(r"\d{1,3}\s*%")

# Entry/row patterns
RE_ENTRY_LINE = re.compile(r"^\s*(?:entry\s+)?\d{1,2}\s+", re.IGNORECASE)


def score_page(page_text: str, page_idx: int, is_si: bool) -> tuple[float, str]:
    """Score a page for likelihood of containing an optimization table."""
    if not page_text or len(page_text.strip()) < 50:
        return 0.0, ""

    lines = page_text.split("\n")
    text_lower = page_text.lower()
    best_score = 0.0
    best_reason = ""

    def _update(score, reason):
        nonlocal best_score, best_reason
        if score > best_score:
            best_score = score
            best_reason = reason

    # ── Tier 1.0: Table caption with optimization keywords ──
    for raw_line in lines:
        line = raw_line.strip()
        if len(line) < 5 or len(line) > 300:
            continue

        m = RE_TABLE_CAPTION.match(line)
        if m:
            table_num = int(m.group(1))
            subtitle = m.group(2).strip()
            idx = lines.index(raw_line)
            if len(subtitle) < 10:
                for j in range(idx + 1, min(idx + 4, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        break
                    subtitle += " " + nxt
                    if len(subtitle) > 300:
                        break

            sub_lower = subtitle.lower()
            if any(kw in sub_lower for kw in OPT_CAPTION_KEYWORDS):
                _update(1.0, f"Table {table_num} caption: {subtitle[:120]}")
                break

            # Tier 0.8: Main-paper Table 1
            if table_num == 1 and not is_si:
                _update(0.8, f"Main paper Table 1: {subtitle[:120]}")

    if best_score >= 1.0:
        return best_score, best_reason

    # ── Tier 0.9: Section header ──
    for raw_line in lines:
        line = raw_line.strip()
        if 5 < len(line) < 120 and RE_OPT_SECTION.match(line):
            _update(0.9, f"Section header: {line[:120]}")
            break

    # ── Tier 0.7: >=3 tabular rows + optimization keywords ──
    row_count = sum(1 for ln in lines if RE_TABLE_ROW.match(ln.strip()))
    entry_count = sum(1 for ln in lines if RE_ENTRY_LINE.match(ln.strip()))
    effective_rows = max(row_count, entry_count)

    has_opt_keyword = any(kw in text_lower for kw in OPT_BODY_KEYWORDS[:4])
    has_yield = len(RE_YIELD.findall(page_text)) >= 2

    if effective_rows >= 3 and has_opt_keyword:
        _update(0.7, f"Tabular rows ({effective_rows}) + optimization keywords")

    # ── Tier 0.5: Table reference + multiple optimization keywords ──
    table_refs = RE_TABLE_REF.findall(page_text)
    opt_keyword_count = sum(1 for kw in OPT_BODY_KEYWORDS if kw in text_lower)
    if table_refs and opt_keyword_count >= 3:
        _update(0.5, f"Table refs {table_refs[:3]} + {opt_keyword_count} opt keywords")

    # Boost: page has yield values + optimization keywords
    if has_yield and has_opt_keyword and effective_rows >= 2:
        _update(max(best_score, 0.5), best_reason or "Yields + optimization keywords")

    return best_score, best_reason


def process_pdf(args: tuple) -> dict | None:
    """Process a single PDF file and score all pages."""
    pdf_path, journal, doi, title, source, threshold = args
    is_si = (source == "supporting_info")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [ERROR] Cannot open {pdf_path}: {e}", file=sys.stderr)
        return None

    page_scores = []
    num_pages = len(doc)

    for page_idx in range(num_pages):
        page = doc[page_idx]
        text = page.get_text()
        score, reason = score_page(text, page_idx, is_si)
        if score > 0:
            page_scores.append((page_idx, score, reason))

    doc.close()

    # ── Tier 0.4: Adjacent pages to high-confidence hits ──
    high_conf_pages = {ps[0] for ps in page_scores if ps[1] >= 0.7}
    already_scored = {ps[0] for ps in page_scores}

    for hc_page in high_conf_pages:
        for adj in [hc_page - 1, hc_page + 1]:
            if 0 <= adj < num_pages and adj not in already_scored:
                page_scores.append(
                    (adj, 0.4, f"Adjacent to high-confidence p{hc_page + 1}")
                )
                already_scored.add(adj)

    candidates = [
        {
            "page": ps[0] + 1,  # 1-indexed
            "score": ps[1],
            "reason": ps[2],
            "source": source,
        }
        for ps in page_scores
        if ps[1] >= threshold
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["page"])

    return {
        "doi": doi,
        "journal": journal,
        "title": title or "",
        "source": source,
        "pdf_path": str(pdf_path),
        "total_pages": num_pages,
        "candidates": candidates,
    }


def doi_from_filename(filename: str) -> str:
    """Convert filename like '10.1021__jacs.0c02331.pdf' to DOI."""
    return filename.replace(".pdf", "").replace("__", "/")


def auto_detect_journals(papers_root: Path) -> list[str]:
    """Auto-detect journal subdirectories (skip hidden dirs and files)."""
    journals = []
    if not papers_root.exists():
        return journals
    for item in sorted(papers_root.iterdir()):
        if item.is_dir() and not item.name.startswith((".", "_")):
            journals.append(item.name)
    return journals


def load_metadata(journal: str, meta_root: Path) -> dict:
    """Load metadata JSON for a journal."""
    meta_path = meta_root / f"{journal}.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Text pre-screening for optimization tables (2025 papers)"
    )
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: process N PDFs per journal")
    parser.add_argument("--journal", type=str, default=None,
                        help="Process only this journal")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum score threshold (default: 0.5)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--no-si", action="store_true",
                        help="Skip supporting information PDFs")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output file path")
    parser.add_argument("--papers-dir", type=str, default=None,
                        help="Override papers directory")
    parser.add_argument("--si-dir", type=str, default=None,
                        help="Override SI directory")
    parser.add_argument("--metadata-dir", type=str, default=None,
                        help="Override metadata directory")
    parser.add_argument("--resume", action="store_true",
                        help="Merge with existing output (keep previous results)")
    args = parser.parse_args()

    papers_root = Path(args.papers_dir) if args.papers_dir else PAPERS_DIR
    si_root = Path(args.si_dir) if args.si_dir else SI_DIR
    meta_root = Path(args.metadata_dir) if args.metadata_dir else METADATA_DIR
    output_file = Path(args.output) if args.output else OUTPUT_FILE
    threshold = args.threshold

    # Auto-detect journals from directory listing
    if args.journal:
        journals = [args.journal]
    else:
        journals = auto_detect_journals(papers_root)
        if not journals:
            print(f"No journal directories found in {papers_root}")
            return

    print(f"Auto-detected journals: {', '.join(journals)}")

    # Load existing results for resume mode
    existing_results = {}
    if args.resume and output_file.exists():
        with open(output_file) as f:
            existing_data = json.load(f)
        existing_results = existing_data.get("papers", {})
        print(f"Resume: loaded {len(existing_results)} existing DOIs")

    # ── Collect all tasks ──
    tasks = []
    for journal in journals:
        metadata = load_metadata(journal, meta_root)

        # Main papers
        papers_dir = papers_root / journal
        if papers_dir.exists():
            pdfs = sorted(f for f in os.listdir(papers_dir) if f.endswith(".pdf"))
            if args.test > 0:
                pdfs = pdfs[:args.test]
            for pdf_name in pdfs:
                doi = doi_from_filename(pdf_name)
                # Skip already-processed DOIs in resume mode
                if args.resume and doi in existing_results:
                    continue
                title = metadata.get(doi, {}).get("title", "")
                tasks.append((
                    str(papers_dir / pdf_name), journal, doi, title,
                    "paper", threshold,
                ))

        # Supporting info
        if not args.no_si:
            si_dir = si_root / journal
            if si_dir.exists():
                pdfs = sorted(f for f in os.listdir(si_dir) if f.endswith(".pdf"))
                if args.test > 0:
                    pdfs = pdfs[:args.test]
                for pdf_name in pdfs:
                    doi = doi_from_filename(pdf_name)
                    if args.resume and doi in existing_results:
                        continue
                    title = metadata.get(doi, {}).get("title", "")
                    tasks.append((
                        str(si_dir / pdf_name), journal, doi, title,
                        "supporting_info", threshold,
                    ))

    total_tasks = len(tasks)
    n_papers = sum(1 for t in tasks if t[4] == "paper")
    n_si = sum(1 for t in tasks if t[4] == "supporting_info")

    print(f"\nStage 1: Text pre-screening for optimization tables (2025)")
    print(f"  PDFs to scan: {total_tasks} ({n_papers} papers + {n_si} SI)")
    print(f"  Journals: {', '.join(journals)}")
    print(f"  Threshold: {threshold}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {output_file}")
    if args.resume:
        print(f"  Resume: keeping {len(existing_results)} existing results")
    print()

    if total_tasks == 0:
        if args.resume and existing_results:
            print("All PDFs already processed. Nothing new to scan.")
        else:
            print("No PDFs found.")
        return

    # ── Process all PDFs ──
    all_results = dict(existing_results)  # Start from existing in resume mode
    stats = {
        "total_pdfs": total_tasks,
        "pdfs_with_candidates": 0,
        "total_candidate_pages": 0,
        "errors": 0,
        "by_journal": {},
        "by_source": {"paper": 0, "supporting_info": 0},
        "by_score": {str(s): 0 for s in [1.0, 0.9, 0.8, 0.7, 0.5, 0.4]},
    }

    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_pdf, task): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            task = futures[future]
            journal = task[1]
            doi = task[2]
            source = task[4]

            if journal not in stats["by_journal"]:
                stats["by_journal"][journal] = {
                    "total": 0, "matched": 0, "candidate_pages": 0,
                }
            stats["by_journal"][journal]["total"] += 1

            try:
                result = future.result()
            except Exception as e:
                print(f"  [ERROR] {doi}: {e}", file=sys.stderr)
                stats["errors"] += 1
                continue

            if result is None:
                continue

            n_cands = len(result["candidates"])
            stats["pdfs_with_candidates"] += 1
            stats["total_candidate_pages"] += n_cands
            stats["by_journal"][journal]["matched"] += 1
            stats["by_journal"][journal]["candidate_pages"] += n_cands
            stats["by_source"][source] += n_cands

            for c in result["candidates"]:
                score_key = str(c["score"])
                if score_key in stats["by_score"]:
                    stats["by_score"][score_key] += 1

            # Merge into all_results by DOI
            if doi not in all_results:
                all_results[doi] = {
                    "doi": doi,
                    "journal": journal,
                    "title": result["title"],
                    "pages": [],
                }
            all_results[doi]["pages"].extend(result["candidates"])

            if completed % 100 == 0 or completed == total_tasks:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed}/{total_tasks}] {rate:.1f} PDFs/s "
                    f"| matched: {stats['pdfs_with_candidates']} "
                    f"| candidates: {stats['total_candidate_pages']}"
                )

    elapsed = time.time() - t0

    # De-duplicate pages within a DOI
    for doi_entry in all_results.values():
        seen = set()
        unique_pages = []
        for p in sorted(doi_entry["pages"], key=lambda x: (-x["score"], x["page"])):
            key = (p["page"], p["source"])
            if key not in seen:
                seen.add(key)
                unique_pages.append(p)
        unique_pages.sort(key=lambda x: (x["source"], x["page"]))
        doi_entry["pages"] = unique_pages

    # Recount totals (including existing in resume mode)
    total_candidate_pages = sum(len(e["pages"]) for e in all_results.values())

    # ── Write output ──
    output = {
        "metadata": {
            "stage": 1,
            "description": "Text-based pre-screening for optimization tables (2025 papers)",
            "threshold": threshold,
            "total_pdfs_scanned": total_tasks + (len(existing_results) if args.resume else 0),
            "total_candidate_pages": total_candidate_pages,
            "total_dois_with_candidates": len(all_results),
            "by_source": stats["by_source"],
            "by_score": stats["by_score"],
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "papers": all_results,
    }

    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print()
    print("=" * 60)
    print("STAGE 1 PRE-SCREENING COMPLETE (2025)")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  New PDFs scanned: {total_tasks}")
    print(f"  New PDFs with candidates: {stats['pdfs_with_candidates']}")
    print(f"  New candidate pages: {stats['total_candidate_pages']}")
    print(f"  Errors: {stats['errors']}")
    print()
    print(f"  Total DOIs with candidates: {len(all_results)}")
    print(f"  Total candidate pages: {total_candidate_pages}")
    print()
    print("  Score distribution (new):")
    for score_val in ["1.0", "0.9", "0.8", "0.7", "0.5", "0.4"]:
        cnt = stats["by_score"].get(score_val, 0)
        if cnt > 0:
            print(f"    {score_val}: {cnt:5d} pages")
    print()
    print("  Per-journal breakdown (new):")
    for j in sorted(stats["by_journal"]):
        s = stats["by_journal"][j]
        print(f"    {j:12s}: {s['matched']:3d}/{s['total']:3d} PDFs "
              f"| {s['candidate_pages']:4d} pages")
    print()
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
