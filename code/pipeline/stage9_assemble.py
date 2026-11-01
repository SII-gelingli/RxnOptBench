#!/usr/bin/env python3
"""
Assemble final structured screening table data by merging three data streams:
  1. Table structure (screening_tables_processed/ or screening_tables_recognized/)
  2. Label-SMILES alignment (mol_labels_screening/)
  3. SMILES data (mol_smiles_decimer_si/)

Produces one JSON file per page with the complete structured representation.

Usage:
    python scripts/assemble_screening_table.py --test 5
    python scripts/assemble_screening_table.py --journal JACS --resume
    python scripts/assemble_screening_table.py   # full run
"""

import argparse
import json
import re
import time
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TABLE_DIRS = [
    BASE_DIR / "screening_tables_processed",
    BASE_DIR / "screening_tables_recognized",
    BASE_DIR / "screening_tables_recognized_2025",
]
DEFAULT_LABELS_DIRS = [
    BASE_DIR / "mol_labels_screening",
    BASE_DIR / "mol_labels_screening_si",
]
DEFAULT_ALIGNMENTS_DIRS = [
    BASE_DIR / "mol_table_alignments_si",
]
DEFAULT_SMILES_DIR = BASE_DIR / "mol_smiles_decimer_si"
DEFAULT_OUTPUT_DIR = BASE_DIR / "screening_tables_final"

PIPELINE_VERSION = "1.0.0"

# ── column type keywords ────────────────────────────────────────────────────
COLUMN_TYPE_RULES = [
    ("entry",      [r"entry", r"no\.?$", r"run", r"#"]),
    ("catalyst",   [r"catalyst", r"cat\.?$", r"pd", r"ni", r"cu", r"metal"]),
    ("ligand",     [r"ligand", r"^L$", r"phosphine"]),
    ("solvent",    [r"solvent", r"medium"]),
    ("base",       [r"base"]),
    ("additive",   [r"additive", r"add\.?"]),
    ("temperature", [r"temp", r"T\s*\(", r"°C", r"°"]),
    ("time",       [r"time", r"hour", r"\bh\b", r"min\b"]),
    ("yield",      [r"yield", r"conv\.?", r"%\s*$"]),
    ("ee",         [r"ee\b", r"enantio"]),
    ("dr",         [r"dr\b", r"diastereo"]),
    ("selectivity", [r"selectiv", r"ratio", r"e/z", r"Z/E", r"cis/trans"]),
    ("substrate",  [r"substrate", r"sub\b", r"aryl", r"nucleophile"]),
    ("product",    [r"product", r"prod\b"]),
    ("reagent",    [r"reagent"]),
    ("equiv",      [r"equiv", r"eq\.?"]),
    ("concentration", [r"conc", r"mol\s*[%/]", r"\[M\]", r"mmol"]),
    ("loading",    [r"loading", r"mol\s*%"]),
    ("pressure",   [r"pressure", r"atm", r"psi", r"bar"]),
]

# ── compound label pattern (reused from align_mol_labels) ────────────────────
LABEL_PATTERN = re.compile(
    r"^(?:"
    r"[1-9]\d{0,2}[a-z]{0,3}'?"
    r"|L\d{1,2}"
    r"|[Cc]at\.?\s*\d*"
    r"|[A-Z][a-z]?\d*"
    r"|R\d*"
    r"|S\d{1,2}"
    r"|P\d{1,2}"
    r")$"
)


def classify_column(col_name: str) -> str:
    """Infer column semantic type from its name."""
    col_lower = col_name.lower().strip()
    for ctype, patterns in COLUMN_TYPE_RULES:
        for pat in patterns:
            if re.search(pat, col_lower, re.IGNORECASE):
                return ctype
    return "other"


def build_label_smiles_map(label_data: dict) -> dict:
    """Build a label → SMILES lookup from mol_labels_screening output."""
    mapping = {}
    for mol in label_data.get("molecules", []):
        label = mol.get("compound_label")
        if not label:
            continue
        smiles = mol.get("smiles_expanded") or mol.get("smiles", "")
        entry = {
            "smiles": smiles,
            "role": mol.get("role", "other"),
            "confidence": mol.get("confidence", 0.0),
            "rdkit_valid": mol.get("rdkit_valid", False),
            "mol_id": mol.get("mol_id", ""),
        }
        # Keep highest confidence if duplicate labels
        if label not in mapping or entry["confidence"] > mapping[label]["confidence"]:
            mapping[label] = entry
    return mapping


def _normalize_label(label: str) -> str:
    """Strip stereochemistry prefixes like (±)-, (+)-, (-)-, (R)-, (S)- etc."""
    return re.sub(r'^[\(\[]?[±+\-RSrs][,/]?[±+\-RSrs]?[\)\]]?-?', '', label).strip()


def _lookup_label(label: str, label_map: dict) -> dict | None:
    """Look up a label in label_map with fuzzy matching.

    Tries in order:
    1. Exact match
    2. Strip stereochemistry prefix: (S)-1a → 1a
    3. Strip numeric variant suffix: 1d-1 → 1d, 3aa-2 → 3aa
    4. Both prefix and suffix stripped
    """
    if label in label_map:
        return label_map[label]
    # Try normalized (strip stereo prefix)
    norm = _normalize_label(label)
    if norm and norm in label_map:
        return label_map[norm]
    # Try stripping variant suffix like -1, -2 (common in chemistry: 1d-1 is variant of 1d)
    base = re.sub(r'-\d+$', '', label).strip()
    if base and base != label and base in label_map:
        return label_map[base]
    # Try both: strip prefix then suffix
    base_norm = re.sub(r'-\d+$', '', norm).strip() if norm else ""
    if base_norm and base_norm != norm and base_norm in label_map:
        return label_map[base_norm]
    return None


def _normalize_title(title: str) -> str:
    """Normalize a table_title for fuzzy matching between stage4 and stage8b."""
    t = (title or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def load_table_alignments_for_page(alignments_dirs: list,
                                   journal: str, doi_dir: str,
                                   page_key: str) -> list[dict]:
    """Load stage8b per-table alignment records for a page.

    Returns a list of alignment records (may be empty). Matches across
    multiple alignment dirs (first match wins per dir).
    """
    for ad in alignments_dirs:
        path = Path(ad) / journal / doi_dir / f"{page_key}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            return data
    return []


def match_alignment_to_table(alignments: list, table_title: str,
                             table_page: str) -> dict | None:
    """Find an alignment record whose table matches the given table."""
    if not alignments:
        return None
    norm_title = _normalize_title(table_title)
    # Prefer exact normalized title match
    for rec in alignments:
        if _normalize_title(rec.get("table_title", "")) == norm_title:
            return rec
    # If only one record on this page, fall back to it
    if len(alignments) == 1:
        return alignments[0]
    return None


def apply_table_alignment(reaction_out: dict, alignment_record: dict) -> dict:
    """Overlay stage8b alignment SMILES onto a reaction dict.

    For each role and slot_idx, if the alignment has a box-resolved SMILES,
    it replaces whatever assign_reaction_smiles produced (including any
    mismatched `ocsr_label` hit). Label/description are preserved from the
    stage4 item; only the SMILES-related fields are overwritten.
    """
    aligns = alignment_record.get("alignments", {}) or {}
    for role in ("substrates", "products", "catalysts", "ligands"):
        slots = reaction_out.get(role, []) or []
        slot_aligns = {a["slot_idx"]: a for a in aligns.get(role, [])
                       if isinstance(a, dict) and "slot_idx" in a}
        for idx, item in enumerate(slots):
            if not isinstance(item, dict):
                continue
            a = slot_aligns.get(idx)
            if not a or not a.get("smiles"):
                continue
            item["smiles"] = a["smiles"]
            item["rdkit_valid"] = bool(a.get("rdkit_valid"))
            item["smiles_source"] = a.get("smiles_source", "ocsr_table_aligned")
            if a.get("box_id"):
                item["box_id"] = a["box_id"]
    return reaction_out


def assign_reaction_smiles(reaction: dict, label_map: dict,
                           label_data: dict = None) -> dict:
    """Assign SMILES to reaction molecules using mol_labels (OCSR) as the
    sole authoritative source.

    Stage 4 provides only labels and descriptions (no SMILES). SMILES come
    exclusively from mol_labels (stage 7/8 OCSR results).

    Strategy:
      1. Label match: look up each reaction item's label in label_map
      2. Role-based fallback: for unmatched items, find mol_labels entries
         with the same role (substrate/product/etc.) by positional order
    """
    _KEY_TO_ROLES = {
        "substrates": {"substrate"},
        "products": {"product"},
        "catalysts": {"catalyst"},
        "ligands": {"ligand"},
    }

    result = {}
    used_mol_ids = set()

    for key in ("substrates", "products", "catalysts", "ligands"):
        items = reaction.get(key, [])
        new_items = []
        unfilled_indices = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            # Discard any VLM-generated SMILES from stage4 — OCSR is the sole source
            new_item.pop("smiles", None)
            new_item.pop("rdkit_valid", None)

            label = new_item.get("label", "")
            lm = _lookup_label(label, label_map) if label else None
            if lm and lm["smiles"]:
                new_item["smiles"] = lm["smiles"]
                new_item["smiles_confidence"] = lm["confidence"]
                new_item["rdkit_valid"] = lm["rdkit_valid"]
                new_item["smiles_source"] = "ocsr_label"
                used_mol_ids.add(lm.get("mol_id", ""))
            else:
                unfilled_indices.append(len(new_items))
            new_items.append(new_item)

        # NOTE: Role-based zip fallback disabled — it matched substrates/products
        # to OCSR molecules by position order within the same role bucket, which
        # caused wrong SMILES assignment when labels could not be matched
        # precisely (e.g. "SCF source" getting assigned the SMILES of product 4).
        # Only label-precise OCSR matches (smiles_source="ocsr_label") are kept.

        result[key] = new_items

    result["conditions_on_arrow"] = reaction.get("conditions_on_arrow", [])
    return result


def annotate_rows_with_smiles(rows: list, columns: list,
                              column_types: dict,
                              label_map: dict) -> list:
    """Scan table rows for cells referencing compound labels and add _smiles_refs.

    Only matches cells that look like compound labels (e.g. 1a, L2, S1).
    Pure numbers (entry indices) are never matched to avoid false positives
    like Entry "3" matching product label "3".
    """
    annotated = []
    # Columns that may reference compounds — exclude entry (pure row numbers)
    compound_cols = {col for col, ctype in column_types.items()
                     if ctype in ("substrate", "product", "catalyst", "ligand",
                                  "other")}
    for row in rows:
        if not isinstance(row, dict):
            annotated.append(row)
            continue
        new_row = dict(row)
        smiles_refs = {}
        for col in columns:
            if col not in compound_cols:
                continue
            val = str(new_row.get(col, "")).strip()
            if not val:
                continue
            # Skip pure numbers — they are entry indices, not compound labels
            if val.isdigit():
                continue
            if val in label_map:
                smiles_refs[col] = {
                    "label": val,
                    "smiles": label_map[val]["smiles"],
                    "role": label_map[val]["role"],
                }
        if smiles_refs:
            new_row["_smiles_refs"] = smiles_refs
        annotated.append(new_row)
    return annotated


def build_table_index(table_dirs: list) -> dict:
    """Pre-scan all table directories once, return {doi_safe: [(path, priority)]}.

    Files from ALL directories are indexed (not deduplicated at file level).
    Deduplication happens at table/page level in load_table_data_indexed,
    so that processed tables override recognized ones per-page while
    pages only present in recognized are still included.
    """
    index = {}  # doi_safe -> [(Path, priority)]
    for priority, table_dir in enumerate(table_dirs):
        td = Path(table_dir)
        if not td.exists():
            continue
        for jf in td.rglob("*.json"):
            if "_progress" in jf.name or "_debug" in str(jf):
                continue
            doi_safe = jf.stem
            index.setdefault(doi_safe, []).append((jf, priority))
    return index


def _table_page_key(table: dict) -> str:
    """Extract a normalized page key from a table dict for dedup."""
    page = str(table.get("page", "")).replace(".png", "")
    if page:
        return page
    image = str(table.get("image", ""))
    if image:
        return Path(image).stem
    return ""


def _table_dedup_key(table: dict) -> str:
    """Build a dedup key for a table: page_key + normalized title."""
    pk = _table_page_key(table)
    title = re.sub(r'[\s^a-z*]+', '', table.get("table_title", "").lower().strip())
    return f"{pk}::{title}"


def load_table_data_indexed(table_index: dict, doi_safe: str) -> list[dict]:
    """Load tables using pre-built index, merging across directories.

    Per-table dedup: if the same table (same page + title) exists in both
    processed/ and recognized/, the higher-priority (processed) version is
    used. Tables only found in lower-priority dirs are still included.
    This prevents losing tables when processed/ has fewer tables than
    recognized/ for the same page.
    """
    entries = table_index.get(doi_safe, [])
    if not entries:
        return []

    # Load all tables tagged with their source priority
    tagged = []  # (table, priority, page_key, dedup_key)
    for jf, priority in entries:
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        tables = data if isinstance(data, list) else [data]
        for t in tables:
            if not isinstance(t, dict):
                continue
            dk = _table_dedup_key(t)
            tagged.append((t, priority, _table_page_key(t), dk))

    # Per-table: find the best (lowest) priority source
    table_best = {}
    for t, priority, pk, dk in tagged:
        if dk and (dk not in table_best or priority < table_best[dk]):
            table_best[dk] = priority

    # Keep tables from the best-priority source for each dedup key
    result = []
    for t, priority, pk, dk in tagged:
        if not dk or priority == table_best.get(dk, priority):
            result.append(t)
    return result


def load_table_data(table_dirs: list, doi_safe: str) -> list[dict]:
    """Load all recognized/processed tables for a DOI.

    Directories listed earlier in *table_dirs* take priority: if a file with
    the same relative path (journal/doi.json) has already been loaded from a
    higher-priority directory, the duplicate is skipped.  This prevents the
    same table appearing twice when both screening_tables_processed/ and
    screening_tables_recognized/ contain the same DOI file.
    """
    tables = []
    seen_rel = set()          # relative paths already loaded (dedup key)
    for table_dir in table_dirs:
        td = Path(table_dir)
        if not td.exists():
            continue
        for jf in td.rglob("*.json"):
            if "_progress" in jf.name or "_debug" in str(jf):
                continue
            if doi_safe not in str(jf):
                continue
            rel = jf.relative_to(td)
            if str(rel) in seen_rel:
                continue
            seen_rel.add(str(rel))
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            if isinstance(data, list):
                tables.extend(data)
            elif isinstance(data, dict):
                tables.append(data)
    return tables


def match_table_to_page(tables: list, page_key: str) -> list[dict]:
    """Filter tables that belong to a specific page."""
    matched = []
    # page_key is like "p16", page field might be "p16.png", 16, "p16", etc.
    page_num = page_key.lstrip("p")
    for table in tables:
        if not isinstance(table, dict):
            continue
        tpage = str(table.get("page", ""))
        timage = str(table.get("image", ""))
        # Match by page number or image path containing page_key
        if (tpage == page_key or tpage == page_num
                or tpage.endswith(f"/{page_key}.png")
                or page_key in timage
                or f"p{page_num}" in timage):
            matched.append(table)
    return matched


def extract_doi_from_key(doi_key: str) -> str:
    """Convert a doi_key, e.g. 'Example/10.0000__invented' to '10.0000/invented'."""
    parts = doi_key.split("/", 1)
    if len(parts) == 2:
        doi_safe = parts[1]
    else:
        doi_safe = doi_key
    # Remove trailing _si suffix
    doi_safe = re.sub(r'_si$', '', doi_safe)
    return doi_safe.replace("__", "/")


def assemble_page(doi_key: str, page_key: str,
                  tables: list, label_data: dict,
                  detection_method: str,
                  table_alignments: list = None) -> list[dict]:
    """Assemble final structured output for one page (may have multiple tables).

    If `table_alignments` (stage8b per-table alignment records) is provided,
    it overrides the label-based SMILES match for each slot — this is the
    authoritative source when available.
    """
    label_map = build_label_smiles_map(label_data)
    journal = doi_key.split("/")[0] if "/" in doi_key else ""
    doi = extract_doi_from_key(doi_key)
    page_num = int(page_key.lstrip("p")) if page_key.lstrip("p").isdigit() else page_key

    results = []
    for table in tables:
        columns = table.get("columns", [])
        rows = table.get("rows", [])
        footnotes = table.get("footnotes", {})
        caption = table.get("caption", "")
        reaction = table.get("reaction", {})
        table_title = table.get("table_title", "")

        # Column type classification
        column_types = {col: classify_column(col) for col in columns}

        # Assign reaction SMILES from OCSR (sole source)
        filled_reaction = assign_reaction_smiles(reaction, label_map, label_data)

        # Stage8b per-table alignment overrides label-based match where available
        if table_alignments:
            align_rec = match_alignment_to_table(
                table_alignments, table_title, str(table.get("page", "")))
            if align_rec is not None:
                filled_reaction = apply_table_alignment(filled_reaction, align_rec)

        # Annotate rows
        annotated_rows = annotate_rows_with_smiles(
            rows, columns, column_types, label_map)

        result = {
            "doi": doi,
            "journal": journal,
            "page": page_num,
            "table_title": table_title,
            "reaction": filled_reaction,
            "columns": columns,
            "column_types": column_types,
            "rows": annotated_rows,
            "footnotes": footnotes,
            "caption": caption,
            "label_smiles_map": {
                label: {
                    "smiles": info["smiles"],
                    "role": info["role"],
                    "confidence": info["confidence"],
                }
                for label, info in label_map.items()
            },
            "metadata": {
                "pipeline_version": PIPELINE_VERSION,
                "mol_detection_method": detection_method,
                "ocsr_model": "external_precomputed",
                "label_alignment_method": "vlm+ocr+table_xref",
            },
        }

        # Add column_footnotes if available (from postprocess_vllm)
        if "column_footnotes" in table:
            result["column_footnotes"] = table["column_footnotes"]
        if "notes" in table:
            result["row_notes"] = table["notes"]
        if table.get("completed_from_adjacent"):
            result["completed_from_adjacent"] = True

        results.append(result)

    # If no tables found but we have label data, create a minimal output
    if not tables and label_data.get("molecules"):
        results.append({
            "doi": doi,
            "journal": journal,
            "page": page_num,
            "table_title": "",
            "reaction": {"substrates": [], "products": [],
                         "catalysts": [], "ligands": [],
                         "conditions_on_arrow": []},
            "columns": [],
            "column_types": {},
            "rows": [],
            "footnotes": {},
            "caption": "",
            "label_smiles_map": {
                label: {
                    "smiles": info["smiles"],
                    "role": info["role"],
                    "confidence": info["confidence"],
                }
                for label, info in label_map.items()
            },
            "metadata": {
                "pipeline_version": PIPELINE_VERSION,
                "mol_detection_method": detection_method,
                "ocsr_model": "external_precomputed",
                "label_alignment_method": "vlm+ocr+table_xref",
                "note": "no_table_structure_found",
            },
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Assemble final structured screening table data")
    parser.add_argument("--table-dir", type=str, nargs="*", default=None,
                        help="Table structure directories")
    parser.add_argument("--labels-dir", type=str, nargs="*", default=None,
                        help="Label directories (default: mol_labels_screening + mol_labels_screening_si)")
    parser.add_argument("--alignments-dir", type=str, nargs="*", default=None,
                        help="Stage8b per-table alignment directories "
                             "(default: mol_table_alignments_si)")
    parser.add_argument("--smiles-dir", type=str, default=str(DEFAULT_SMILES_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--detection-method", type=str, default="DECIMER",
                        choices=["DECIMER", "OpenCV"],
                        help="Molecule detection method used")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--journal", type=str, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip already-assembled pages (default: True)")
    args = parser.parse_args()

    # Labels directories
    if args.labels_dir:
        labels_dirs = [Path(d) for d in args.labels_dir]
    else:
        labels_dirs = [d for d in DEFAULT_LABELS_DIRS if d.exists()]

    # Stage8b alignment directories (optional)
    if args.alignments_dir:
        alignments_dirs = [Path(d) for d in args.alignments_dir]
    else:
        alignments_dirs = [d for d in DEFAULT_ALIGNMENTS_DIRS if d.exists()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table directories
    if args.table_dir:
        table_dirs = [str(d) for d in args.table_dir]
    else:
        table_dirs = [str(d) for d in DEFAULT_TABLE_DIRS if d.exists()]

    if not labels_dirs:
        print(f"ERROR: No labels directories found")
        print("Run align_mol_labels.py first.")
        return

    # Collect tasks from label files (across all label dirs)
    # Merge label files for the same page across directories instead of
    # creating duplicate tasks (which causes later dirs to overwrite earlier ones).
    from collections import defaultdict
    _page_label_files = defaultdict(list)  # (journal, doi_dir, page_key) -> [file1, file2]
    for labels_dir in labels_dirs:
        for label_file in sorted(labels_dir.rglob("*.json")):
            rel = label_file.relative_to(labels_dir)
            parts = rel.parts
            if len(parts) < 3:
                continue
            journal = parts[0]
            doi_dir = parts[1]
            page_key = label_file.stem

            if args.journal and journal != args.journal:
                continue

            _page_label_files[(journal, doi_dir, page_key)].append(str(label_file))

    tasks = []
    for (journal, doi_dir, page_key), label_files in sorted(_page_label_files.items()):
        tasks.append({
            "journal": journal,
            "doi_dir": doi_dir,
            "doi_key": f"{journal}/{doi_dir}",
            "page_key": page_key,
            "label_file": label_files[0],
            "label_files": label_files,  # all files for merging
        })

    if args.test > 0:
        tasks = tasks[:args.test]

    # Resume
    if args.resume:
        before = len(tasks)
        new_tasks = []
        for t in tasks:
            out_file = output_dir / t["journal"] / t["doi_dir"] / f"{t['page_key']}.json"
            if out_file.exists():
                continue
            new_tasks.append(t)
        skipped = before - len(new_tasks)
        tasks = new_tasks
        if skipped:
            print(f"Resume: skipping {skipped} already-assembled pages")

    print(f"Screening table assembly")
    print(f"  Labels:     {[str(d) for d in labels_dirs]}")
    print(f"  Tables:     {table_dirs}")
    print(f"  Alignments: {[str(d) for d in alignments_dirs] or '(none)'}")
    print(f"  Output:     {output_dir}")
    print(f"  Pages:   {len(tasks)}")
    print(f"  Detection: {args.detection_method}")
    print("=" * 60)

    if not tasks:
        print("Nothing to process!")
        return

    # Pre-build table index (scan dirs ONCE instead of per-page rglob)
    print("Building table index...", flush=True)
    table_index = build_table_index(table_dirs)
    print(f"  Indexed {sum(len(v) for v in table_index.values())} table files "
          f"for {len(table_index)} DOIs", flush=True)

    t0 = time.time()
    total_tables = 0
    total_with_smiles = 0
    errors = 0

    for i, task in enumerate(tasks, 1):
        try:
            # Load and merge label data from all label dirs
            label_data = json.loads(Path(task["label_file"]).read_text())
            for extra_file in task.get("label_files", [])[1:]:
                try:
                    extra = json.loads(Path(extra_file).read_text())
                    # Merge molecules: add any with labels not yet present
                    existing_labels = {m.get("compound_label") for m in label_data.get("molecules", []) if m.get("compound_label")}
                    for mol in extra.get("molecules", []):
                        lbl = mol.get("compound_label")
                        if lbl and lbl not in existing_labels:
                            label_data.setdefault("molecules", []).append(mol)
                            existing_labels.add(lbl)
                except Exception:
                    pass

            # Load table structure (using index — no rglob per page)
            doi_safe = task["doi_dir"]
            all_tables = load_table_data_indexed(table_index, doi_safe)
            page_tables = match_table_to_page(all_tables, task["page_key"])

            # For cross-page tables (stage4b stitched from prev/next page), also
            # pull in adjacent pages' mol_labels so that compound labels drawn on
            # the neighbour pages show up in label_smiles_map. stage8b already
            # looks at [prev, page, next] for the image side; stage9 needs the
            # same window for labels.
            if any(t.get("completed_from_adjacent") for t in page_tables):
                page_num_str = task["page_key"].lstrip("p")
                if page_num_str.isdigit():
                    page_num = int(page_num_str)
                    existing_labels = {
                        m.get("compound_label")
                        for m in label_data.get("molecules", [])
                        if m.get("compound_label")
                    }
                    for offset in (-1, 1):
                        adj_key = f"p{page_num + offset}"
                        for ld in labels_dirs:
                            adj = Path(ld) / task["journal"] / task["doi_dir"] / f"{adj_key}.json"
                            if not adj.exists():
                                continue
                            try:
                                extra = json.loads(adj.read_text())
                            except Exception:
                                continue
                            for mol in extra.get("molecules", []):
                                lbl = mol.get("compound_label")
                                if lbl and lbl not in existing_labels:
                                    # Annotate which page it came from so downstream
                                    # consumers can trace provenance if needed.
                                    mol_copy = dict(mol)
                                    mol_copy.setdefault("source_page", adj_key)
                                    label_data.setdefault("molecules", []).append(mol_copy)
                                    existing_labels.add(lbl)

            # Load stage8b per-table alignments for this page (if any)
            table_alignments = load_table_alignments_for_page(
                alignments_dirs, task["journal"], task["doi_dir"],
                task["page_key"]
            ) if alignments_dirs else []

            # Assemble
            assembled = assemble_page(
                task["doi_key"], task["page_key"],
                page_tables, label_data,
                args.detection_method,
                table_alignments=table_alignments,
            )

            # Save
            out_sub = output_dir / task["journal"] / task["doi_dir"]
            out_sub.mkdir(parents=True, exist_ok=True)
            out_file = out_sub / f"{task['page_key']}.json"
            out_file.write_text(json.dumps(assembled, ensure_ascii=False, indent=2))

            n_tables = len(assembled)
            n_with_map = sum(1 for a in assembled if a.get("label_smiles_map"))
            total_tables += n_tables
            total_with_smiles += n_with_map

            if i % 20 == 0 or i == len(tasks) or i <= 3:
                elapsed = time.time() - t0
                print(f"  [{i}/{len(tasks)}] {task['doi_key']}/{task['page_key']}: "
                      f"{n_tables} table(s), "
                      f"{len(label_data.get('molecules', []))} mol labels | "
                      f"{elapsed:.0f}s elapsed")

        except Exception as e:
            print(f"  [{i}/{len(tasks)}] ERROR {task['doi_key']}/{task['page_key']}: {e}")
            errors += 1

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Done! {total_tables} tables from {len(tasks)} pages in {elapsed:.1f}s")
    print(f"  Tables with SMILES map: {total_with_smiles}/{total_tables}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
