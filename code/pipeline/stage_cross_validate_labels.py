#!/usr/bin/env python3
"""
Cross-validate stage8 (page-level box→label) against stage8b (table slot→box)
and backfill missing compound_labels in stage8 outputs using stage8b data.

Runs AFTER both stage8 and stage8b complete. No VLM calls; pure local join.

Input:
  mol_labels_screening_si/{Journal}/{DOI}/p{N}.json        (stage8 output)
  mol_table_alignments_si/{Journal}/{DOI}/p{N}.json        (stage8b output)

Output (in place):
  mol_labels_screening_si/{Journal}/{DOI}/p{N}.json  updated with:
    - compound_label filled in from 8b slot.label when originally None
    - role inferred from 8b slot kind (substrates→substrate, etc.) when 'other'
    - confidence set to 0.60 (distinguishes from stage8's 0.80)
    - warnings entry "backfilled_from_8b"

Usage:
  python3 pipeline/stage_cross_validate_labels.py
  python3 pipeline/stage_cross_validate_labels.py --dry-run
  python3 pipeline/stage_cross_validate_labels.py --labels-dir mol_labels_screening_si \\
      --alignments-dir mol_table_alignments_si
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SLOT_ROLE = {
    "substrates": "substrate",
    "products": "product",
    "catalysts": "catalyst",
    "ligands": "ligand",
}


def load_alignments(alignments_dir: Path) -> dict:
    """Build reverse index: (journal, doi, page_key, mol_id) -> (label, role).

    If a box is claimed by multiple slots across tables with different labels,
    the first non-None label wins; conflicts are logged to stats.
    """
    reverse = {}  # (journal, doi, page_key, mol_id) -> {"label", "role"}
    conflicts = 0
    n_tables = 0
    n_slots = 0

    for jdir in alignments_dir.iterdir():
        if not jdir.is_dir():
            continue
        for doidir in jdir.iterdir():
            if not doidir.is_dir():
                continue
            for pf in doidir.glob("p*.json"):
                try:
                    records = json.loads(pf.read_text())
                except Exception:
                    continue
                if not isinstance(records, list):
                    records = [records]
                for rec in records:
                    n_tables += 1
                    aligns = rec.get("alignments") or {}
                    for slot_kind, slots in aligns.items():
                        if not isinstance(slots, list):
                            continue
                        for slot in slots:
                            n_slots += 1
                            box_id = slot.get("box_id")
                            label = slot.get("label")
                            if not box_id or not label:
                                continue
                            if ":" not in box_id:
                                continue
                            page_tag, mol_id = box_id.split(":", 1)
                            key = (jdir.name, doidir.name, page_tag,
                                   str(mol_id))
                            role = SLOT_ROLE.get(slot_kind, "other")
                            prev = reverse.get(key)
                            if prev is None:
                                reverse[key] = {"label": label, "role": role}
                            elif prev["label"] != label:
                                conflicts += 1
    print(f"  Loaded alignments: {n_tables} tables, {n_slots} slots "
          f"=> {len(reverse)} unique box→label entries, {conflicts} conflicts")
    return reverse


def backfill_labels(labels_dir: Path, reverse: dict, dry_run: bool):
    n_files = 0
    n_files_touched = 0
    n_mols_scanned = 0
    n_mols_was_none = 0
    n_mols_backfilled = 0
    n_files_fully_recovered = 0  # files that went from 0 labeled → >0

    per_doi_recovery = defaultdict(lambda: [0, 0])  # doi → [before_none, after_none]

    for jdir in labels_dir.iterdir():
        if not jdir.is_dir():
            continue
        for doidir in jdir.iterdir():
            if not doidir.is_dir():
                continue
            for pf in doidir.glob("p*.json"):
                try:
                    data = json.loads(pf.read_text())
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                mols = data.get("molecules", [])
                if not mols:
                    continue
                n_files += 1

                journal = jdir.name
                doi_dir = doidir.name
                page_key = pf.stem  # "p4"

                before_labeled = sum(1 for m in mols if m.get("compound_label"))
                touched_this_file = False

                for m in mols:
                    n_mols_scanned += 1
                    if m.get("compound_label"):
                        continue
                    n_mols_was_none += 1
                    mol_id = str(m.get("mol_id", ""))
                    if not mol_id:
                        continue
                    key = (journal, doi_dir, page_key, mol_id)
                    hit = reverse.get(key)
                    if not hit:
                        continue
                    # Backfill
                    m["compound_label"] = hit["label"]
                    if m.get("role") in (None, "", "other"):
                        m["role"] = hit["role"]
                    m["confidence"] = 0.60
                    touched_this_file = True
                    n_mols_backfilled += 1

                if touched_this_file:
                    ws = data.get("warnings") or []
                    if "backfilled_from_8b" not in ws:
                        ws.append("backfilled_from_8b")
                    data["warnings"] = ws
                    n_files_touched += 1
                    after_labeled = sum(1 for m in mols
                                         if m.get("compound_label"))
                    if before_labeled == 0 and after_labeled > 0:
                        n_files_fully_recovered += 1
                    per_doi_recovery[f"{journal}/{doi_dir}"][0] += (len(mols)
                                                                   - before_labeled)
                    per_doi_recovery[f"{journal}/{doi_dir}"][1] += (len(mols)
                                                                   - after_labeled)
                    if not dry_run:
                        pf.write_text(json.dumps(data, ensure_ascii=False,
                                                  indent=2))

    print()
    print("=" * 60)
    print(f"Files scanned:                    {n_files}")
    print(f"Files touched:                    {n_files_touched}")
    print(f"Files fully recovered (0→>0):     {n_files_fully_recovered}")
    print(f"Molecules scanned:                {n_mols_scanned}")
    print(f"Molecules that were compound_label=None: {n_mols_was_none}")
    print(f"Molecules backfilled from 8b:     {n_mols_backfilled} "
          f"({100 * n_mols_backfilled / max(n_mols_was_none, 1):.1f}% recovery)")
    if dry_run:
        print("(DRY RUN — no files written)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", type=str,
                    default=str(BASE_DIR / "mol_labels_screening_si"))
    ap.add_argument("--alignments-dir", type=str,
                    default=str(BASE_DIR / "mol_table_alignments_si"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    alignments_dir = Path(args.alignments_dir)

    print(f"Labels dir:     {labels_dir}")
    print(f"Alignments dir: {alignments_dir}")
    print(f"Dry run:        {args.dry_run}")
    print()

    if not alignments_dir.exists():
        print(f"ERROR: alignments dir not found: {alignments_dir}")
        return
    if not labels_dir.exists():
        print(f"ERROR: labels dir not found: {labels_dir}")
        return

    print("Building reverse box→label index from stage8b alignments ...")
    reverse = load_alignments(alignments_dir)
    print()
    print("Backfilling stage8 labels ...")
    backfill_labels(labels_dir, reverse, args.dry_run)


if __name__ == "__main__":
    main()
