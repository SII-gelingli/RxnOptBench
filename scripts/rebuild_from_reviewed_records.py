#!/usr/bin/env python3
"""Rebuild RxnOptBench questions from the retained reviewed-table snapshot.

This adapter performs no API or network calls. It does not re-extract publisher
documents, repeat OCSR, or replace human review. It converts the released
``reviewed_data/tables`` row-index mapping to the input list expected by the
archived V6 builders and invokes those builders unchanged. Their archived
major-fraction scoring semantics are retained; this is not a target-aware
selectivity relabeling tool.

The reviewed snapshot omitted the original extraction ``meta_info``. A verified
explicit table-ID mapping can restore legacy IDs and their deterministic option
shuffle seeds. Without it, stable IDs derived from the reviewed filename are
used and must not be presented as original submitted IDs. No molecule, outcome,
condition, row order, or field is repaired by this adapter.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_reviewed_tables(artifact_root: Path, output_dir: Path, table_id_map: dict | None = None) -> dict:
    reviewed_root = artifact_root / "reviewed_data"
    tables_root = reviewed_root / "tables"
    manifest_path = reviewed_root / "manifest.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    actual = {p.relative_to(reviewed_root).as_posix(): p for p in tables_root.rglob("*.json")}
    if set(expected) != set(actual):
        raise ValueError("Reviewed manifest/file set mismatch; refusing an incomplete or expanded snapshot")
    table_id_map = table_id_map or {}
    files = []
    claimed_ids: dict[str, str] = {}
    for relative in sorted(actual):
        source = actual[relative]
        if sha256(source) != expected[relative]:
            raise ValueError(f"Reviewed manifest SHA-256 mismatch: {relative}")
        mapping = table_id_map.get(relative)
        if isinstance(mapping, dict) and mapping.get("reviewed_sha256") not in (None, expected[relative]):
            raise ValueError(f"Table-ID bridge was derived from different reviewed bytes: {relative}")
        table_id = mapping["parent_table_id"] if isinstance(mapping, dict) else mapping
        table_id = table_id or source.stem
        if not isinstance(table_id, str) or Path(table_id).name != table_id or table_id in (".", ".."):
            raise ValueError(f"Invalid table ID: {table_id!r}")
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected row-index mapping in {relative}")
        # JSON insertion order is the retained source row order. Do not sort
        # string row labels: lexicographic sorting would put row 10 before row 2.
        rows = []
        row_labels = []
        for row_label, reactions in data.items():
            if not isinstance(reactions, list) or len(reactions) != 1:
                raise ValueError(f"{relative}, row {row_label}: expected exactly one reaction; refusing ambiguous flattening")
            row = copy.deepcopy(reactions[0])
            if not isinstance(row, dict):
                raise ValueError(f"{relative}, row {row_label}: reaction must be an object")
            if "meta_info" in row:
                raise ValueError(f"{relative}, row {row_label}: unexpected existing meta_info; inspect rather than overwrite")
            row["meta_info"] = {"root_img_name": table_id, "entry_idx": str(row_label)}
            rows.append(row)
            row_labels.append(str(row_label))
        if table_id in claimed_ids:
            raise ValueError(f"Repeated output ID {table_id}: {relative} and {claimed_ids[table_id]}")
        claimed_ids[table_id] = relative
        # Legacy readers require '_table_' in the *filename*. This input adapter
        # filename is synthetic; root_img_name above carries the documented ID.
        # A restored legacy ID is already a valid archived input filename.
        # Unmapped reviewed IDs receive a synthetic loader suffix only.
        adapter_name = f"{table_id}.json" if re.search(r"_(?:table|scheme)_\d+$", table_id) else f"{table_id}_table_0.json"
        destination = output_dir / "converted_tables" / adapter_name
        write_json(destination, rows)
        files.append({"reviewed_path": relative, "reviewed_sha256": expected[relative],
                      "parent_table_id": table_id, "restored_legacy_id": relative in table_id_map,
                      "row_count": len(rows), "row_labels": row_labels,
                      "converted_path": destination.relative_to(output_dir).as_posix()})
    return {"network_calls": False, "api_stages_rerun": False,
            "scoring_semantics": "archived_V6_major_fraction_not_target_aware",
            "reviewed_table_count": len(files), "conversion": "one reaction per row; preserved insertion order and every existing field; added meta_info only",
            "files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="Downloaded HF artifact containing reviewed_data/ and optionally data/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; existing directories are refused")
    parser.add_argument("--code-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "code",
                        help="Directory with archived build_multiple_choice.py/build_fill_in_blank.py; sibling evaluation/ must exist")
    parser.add_argument("--table-id-map", type=Path,
                        help="Optional verified reviewed-path → parent-table-ID map (JSON; root may be 'tables')")
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args()
    root = args.artifact_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new directory.")
    mapping = json.loads(args.table_id_map.read_text(encoding="utf-8")) if args.table_id_map else {}
    if "tables" in mapping:
        mapping = mapping["tables"]
    output.mkdir(parents=True)
    report = convert_reviewed_tables(root, output, mapping)
    if not args.convert_only:
        code_root = args.code_root.expanduser().resolve()
        report["builder_commands"] = []
        process_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for name in ("build_multiple_choice.py", "build_fill_in_blank.py"):
            script = code_root / name
            if not script.is_file():
                raise FileNotFoundError(script)
            command = [sys.executable, str(script), "--article-dir", str(output / "converted_tables"),
                       "--output-dir", str(output / "build_outputs")]
            log = output / f"{script.stem}.log"
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=process_env)
            report["builder_commands"].append({"script": name, "script_sha256": sha256(script), "log": log.name})
        report["shared_evaluator_sha256"] = sha256(code_root.parent / "evaluation" / "common_v6.py")
    write_json(output / "rebuild_manifest.json", report)
    print(json.dumps({"reviewed_tables": report["reviewed_table_count"], "output_dir": str(output),
                      "network_calls": False, "scoring_semantics": report["scoring_semantics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
