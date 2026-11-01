#!/usr/bin/env python3
"""
Post-process SMILES in expanded results to fix common invalid patterns.

Fixes:
  1. Thiocyanate: [SCN-] / [SC-]#N → [S-]C#N
  2. R-group placeholders: R, R1, R2, Ar, Pg, X → [*] (wildcard)
  3. Structural errors: attempt auto-repair (parenthesis, ring closure, etc.)

Usage:
    # Dry-run (report only)
    python3 pipeline/fix_smiles.py

    # Apply fixes in-place
    python3 pipeline/fix_smiles.py --apply

    # Specific project directory
    python3 pipeline/fix_smiles.py --apply --input-dir /path/to/screening_tables_expanded
"""

import argparse
import json
import re
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "screening_tables_expanded"


# ── Fix rules ───────────────────────────────────────────────────────────────

def fix_thiocyanate(smi: str) -> str:
    """Fix thiocyanate ion: [SCN-] → [S-]C#N, [SC-]#N → [S-]C#N."""
    smi = smi.replace("[SCN-]", "[S-]C#N")
    smi = smi.replace("[SC-]#N", "[S-]C#N")
    smi = smi.replace("[SC#N]", "[S-]C#N")
    smi = smi.replace("[SC#N-]", "[S-]C#N")
    return smi


# Counter-ion shorthand → expanded SMILES
_COUNTERION_MAP = {
    "[BF4-]": "[B-](F)(F)(F)F",
    "[BF4]":  "[B-](F)(F)(F)F",
    "[PF6-]": "[P-](F)(F)(F)(F)(F)F",
    "[PF6]":  "[P-](F)(F)(F)(F)(F)F",
    "[SbF6-]": "[Sb-](F)(F)(F)(F)(F)F",
    "[SbF6]":  "[Sb-](F)(F)(F)(F)(F)F",
    "[ClO4-]": "[O-]Cl(=O)(=O)=O",
    "[ClO4]":  "[O-]Cl(=O)(=O)=O",
    "[OTf-]": "[O-]S(=O)(=O)C(F)(F)F",
    "[OTf]":  "[O-]S(=O)(=O)C(F)(F)F",
    "[NTf2-]": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "[HF-]":  "[H]F",       # not a real ion, likely OCR artifact
    "[HF2-]": "F.[H]F",     # bifluoride
}


def fix_counterions(smi: str) -> str:
    """Expand shorthand counter-ion notation to explicit SMILES."""
    for short, expanded in _COUNTERION_MAP.items():
        if short in smi:
            smi = smi.replace(short, expanded)
    return smi


# R-group tokens, ordered longest-first to avoid partial matches
_RGROUP_TOKENS = [
    # multi-char first
    ("R1", "[*:1]"), ("R2", "[*:2]"), ("R3", "[*:3]"), ("R4", "[*:4]"),
    ("Pg", "[*:10]"), ("Ar", "[*:11]"),
]

# Single-char R/X: must NOT be preceded by letter/[ and NOT followed by
# lowercase letter (would be element like Ra, Rn, Rh) or digit/]
_SINGLE_RGROUP_RE = re.compile(
    r'(?<![A-Za-z\[\]])([RX])(?![a-z0-9\]])'
)

# Catch trailing R at end of SMILES (e.g. "...S(=O)R", "...c1R")
_TRAILING_R_RE = re.compile(r'R$')
# Catch R after bond/paren context that regex above misses
_CONTEXT_R_RE = re.compile(r'(?<=[)(\]])R(?![a-z0-9\]])')


def fix_rgroups(smi: str) -> str:
    """Replace R-group placeholders with RDKit wildcard atoms [*]."""
    for token, replacement in _RGROUP_TOKENS:
        smi = smi.replace(token, replacement)
    # Handle standalone R and X (not inside brackets, not followed by lowercase)
    smi = _SINGLE_RGROUP_RE.sub("[*]", smi)
    # Handle R after closing brackets/parens
    smi = _CONTEXT_R_RE.sub("[*]", smi)
    # Handle leading R (e.g. "RSH" → "[*]S")
    if re.match(r'^R(?![a-z])', smi):
        smi = "[*]" + smi[1:]
    # Handle trailing R
    smi = _TRAILING_R_RE.sub("[*]", smi)
    # Handle [R] explicitly
    smi = smi.replace("[R]", "[*]")
    # Handle [X-] (halide placeholder)
    smi = smi.replace("[X-]", "[Cl-]")
    return smi


def fix_nitrosyl(smi: str) -> str:
    """Fix N=O[O-] → [O-][N]=O (sodium nitrite)."""
    if smi == "N=O[O-].[Na+]":
        return "[Na+].[N](=O)[O-]"
    return smi


def try_kekulize_fix(smi: str) -> str | None:
    """Try sanitizing with partial sanitization to fix kekulization."""
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def fix_smiles(smi: str) -> tuple[str, str]:
    """Try to fix an invalid SMILES string.

    Returns:
        (fixed_smiles, fix_type) where fix_type describes what was done.
        If unfixable, returns (original, "unfixable").
    """
    original = smi

    # Already valid?
    if Chem.MolFromSmiles(smi) is not None:
        return smi, "already_valid"

    # 1. Thiocyanate fix
    fixed = fix_thiocyanate(smi)
    if fixed != smi and Chem.MolFromSmiles(fixed) is not None:
        return fixed, "thiocyanate"

    # 2. Counter-ion fix (BF4, PF6, etc.)
    fixed = fix_counterions(smi)
    if fixed != smi and Chem.MolFromSmiles(fixed) is not None:
        return fixed, "counterion"

    # 3. Nitrosyl fix
    fixed = fix_nitrosyl(smi)
    if fixed != smi and Chem.MolFromSmiles(fixed) is not None:
        return fixed, "nitrosyl"

    # 4. R-group fix
    fixed = fix_rgroups(smi)
    if fixed != smi and Chem.MolFromSmiles(fixed) is not None:
        return fixed, "rgroup"

    # 5. Combined: thiocyanate + counterion + rgroup
    fixed = fix_rgroups(fix_counterions(fix_thiocyanate(smi)))
    if fixed != smi and Chem.MolFromSmiles(fixed) is not None:
        return fixed, "combined"

    # 5. Try kekulize fix on original
    fixed = try_kekulize_fix(smi)
    if fixed is not None:
        return fixed, "kekulize"

    return original, "unfixable"


# ── Main ─────────────────────────────────────────────────────────────────────

def process_file(jf: Path, apply: bool) -> dict:
    """Process one expanded JSON file. Returns stats dict."""
    stats = {"total": 0, "invalid": 0, "fixed": 0, "unfixable": 0, "fixes": []}

    try:
        data = json.load(open(jf))
    except Exception:
        return stats

    if not isinstance(data, list):
        data = [data]

    modified = False

    for table in data:
        if not isinstance(table, dict):
            continue

        # Handle both expanded_rows (stage10 output) and top-level reaction
        rows_to_check = []
        for row in table.get("expanded_rows", []):
            rxn = row.get("reaction")
            if isinstance(rxn, dict):
                rows_to_check.append(rxn)
        # Fallback: top-level reaction
        top_rxn = table.get("reaction")
        if isinstance(top_rxn, dict):
            rows_to_check.append(top_rxn)

        for rxn in rows_to_check:
            for role in ["substrates", "products", "catalysts", "ligands"]:
                for item in rxn.get(role, []):
                    if not isinstance(item, dict):
                        continue
                    smi = item.get("smiles", "")
                    if not smi:
                        continue

                    stats["total"] += 1

                    if Chem.MolFromSmiles(smi) is not None:
                        continue

                    stats["invalid"] += 1
                    fixed, fix_type = fix_smiles(smi)

                    if fix_type != "unfixable":
                        stats["fixed"] += 1
                        stats["fixes"].append({
                            "original": smi,
                            "fixed": fixed,
                            "type": fix_type,
                            "file": str(jf.name),
                        })
                        if apply:
                            item["smiles"] = fixed
                            item["smiles_original"] = smi
                            item["smiles_fix_type"] = fix_type
                            modified = True
                    else:
                        stats["unfixable"] += 1
                        stats["fixes"].append({
                            "original": smi,
                            "fixed": None,
                            "type": "unfixable",
                            "file": str(jf.name),
                        })

    if apply and modified:
        jf.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fix invalid SMILES in expanded results")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--apply", action="store_true",
                        help="Apply fixes in-place (default: dry-run)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found")
        return

    all_files = sorted(input_dir.rglob("*.json"))
    print(f"SMILES post-processing {'(APPLY)' if args.apply else '(DRY-RUN)'}")
    print(f"  Input: {input_dir}")
    print(f"  Files: {len(all_files)}")
    print("=" * 60)

    total_stats = {"total": 0, "invalid": 0, "fixed": 0, "unfixable": 0}
    all_fixes = []
    unfixable_list = []

    for jf in all_files:
        stats = process_file(jf, args.apply)
        total_stats["total"] += stats["total"]
        total_stats["invalid"] += stats["invalid"]
        total_stats["fixed"] += stats["fixed"]
        total_stats["unfixable"] += stats["unfixable"]
        for fix in stats["fixes"]:
            fix["file"] = str(jf.relative_to(input_dir))
            if fix["type"] == "unfixable":
                unfixable_list.append(fix)
            else:
                all_fixes.append(fix)

    print(f"\nResults:")
    print(f"  Total SMILES: {total_stats['total']}")
    print(f"  Invalid:      {total_stats['invalid']}")
    print(f"  Fixed:        {total_stats['fixed']}")
    print(f"  Unfixable:    {total_stats['unfixable']}")

    if all_fixes:
        print(f"\nFixes applied:" if args.apply else "\nFixes (dry-run):")
        # Group by fix type
        from collections import Counter
        type_counts = Counter(f["type"] for f in all_fixes)
        for ftype, count in type_counts.most_common():
            print(f"  {ftype}: {count}")
        print()
        for fix in all_fixes:
            print(f"  [{fix['type']}] {fix['original']}")
            print(f"       → {fix['fixed']}")
            print(f"         ({fix['file']})")

    if unfixable_list:
        print(f"\nUnfixable ({len(unfixable_list)}):")
        for fix in unfixable_list:
            print(f"  {fix['original']}")
            print(f"    ({fix['file']})")

    # Save report
    report_path = input_dir / "_smiles_fix_report.json"
    report = {
        "stats": total_stats,
        "fixes": all_fixes,
        "unfixable": unfixable_list,
        "applied": args.apply,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
