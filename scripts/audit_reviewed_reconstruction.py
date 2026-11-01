#!/usr/bin/env python3
"""Compare offline reviewed-record rebuilds with frozen submitted questions.

Comparison alone normalizes option/candidate ordering and removes provenance
fields. It never alters a dataset or infers/copies chemistry or gold labels.
Unique unchanged question content from the same DOI can supply an explicit
one-to-one legacy parent-ID bridge, whose derivation is saved with evidence.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def load_datasets(root):
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in root.rglob("*.json")
            if p.name.startswith(("multiple_choice_", "fill_in_blank_")) and isinstance(json.loads(p.read_text(encoding="utf-8")), list)}


def doi_from_reviewed_id(stem):
    return re.sub(r"_si(?:_\d+)*$", "", stem.split("__p")[0])


def remove_source_entry(value):
    if isinstance(value, dict):
        return {k: remove_source_entry(v) for k, v in value.items() if k != "source_entry"}
    if isinstance(value, list):
        return [remove_source_entry(v) for v in value]
    return value


def identity_content(q):
    return [q["question_type"], q.get("target_key"), q.get("varying_keys"),
            q["input"], sorted(q["options"], key=dump)]


def canonical_question(q):
    """Retain all question/answer/utility content; ignore source labels only."""
    result = {k: v for k, v in q.items() if k not in ("id", "input", "answer", "options", "meta")}
    inp = dict(q["input"])
    if "reference_conditions" in inp:
        inp["reference_conditions"] = sorted(remove_source_entry(inp["reference_conditions"]), key=dump)
    meta = q["meta"]
    ignored = {"source_file", "doi", "parent_table_id", "best_entry_indices", "global_best_entry_indices",
               "option_source_entries", "legal_combinations", "yields", "scores", "effective_scores", "option_relative_scores"}
    result["meta"] = {k: v for k, v in meta.items() if k not in ignored}
    if "options" in q:
        bundles = []
        for i, option in enumerate(q["options"]):
            bundle = {"option": option, "is_correct": i in q["answer"]}
            for k in ("yields", "scores", "effective_scores", "option_relative_scores"):
                if k in meta:
                    bundle[k] = meta[k][i]
            bundles.append(bundle)
        result["option_label_pairs"] = sorted(bundles, key=dump)
    else:
        candidates = inp.pop("candidate_values")
        by_index = {key: {item["index"]: item["value"] for item in values}
                    for key, values in candidates.items()}
        def expand(selection):
            return {key: by_index[key][i] for key, i in selection.items()}
        result["candidate_values"] = {k: sorted(v.values(), key=dump) for k, v in by_index.items()}
        result["answer_values"] = sorted((expand(a) for a in q["answer"]), key=dump)
        legal = []
        for record in meta["legal_combinations"]:
            item = {k: v for k, v in record.items() if k not in ("source_entry", "selection")}
            item["selection_values"] = expand(record["selection"])
            legal.append(item)
        result["legal_combinations"] = sorted(legal, key=dump)
    result["input"] = inp
    return result


def make_id_map(original, rebuilt, manifest):
    edges = collections.defaultdict(set)
    evidence = collections.defaultdict(list)
    for filename in ("multiple_choice_all_varying.json", "multiple_choice_single_varying.json"):
        index = collections.defaultdict(list)
        for q in original[filename]:
            index[(q["meta"]["doi"], dump(identity_content(q)))].append(q)
        for q in rebuilt[filename]:
            parent = q["meta"]["parent_table_id"]
            hits = index[(doi_from_reviewed_id(parent), dump(identity_content(q)))]
            if len(hits) == 1:
                old = hits[0]
                edges[parent].add(old["meta"]["parent_table_id"])
                evidence[parent].append({"subset": filename.removesuffix(".json"),
                                         "archived_question_id": old["id"],
                                         "content_sha256": sha(identity_content(q))})
    inverse = collections.Counter(next(iter(v)) for v in edges.values() if len(v) == 1)
    files_by_id = {r["parent_table_id"]: r for r in manifest["files"]}
    accepted = {}
    for parent in sorted(edges):
        hits = edges[parent]
        if len(hits) != 1:
            continue
        old_id = next(iter(hits))
        if inverse[old_id] != 1:
            continue
        file_info = files_by_id[parent]
        accepted[file_info["reviewed_path"]] = {"parent_table_id": old_id,
            "reviewed_sha256": file_info["reviewed_sha256"],
            "evidence": evidence[parent],
            "basis": "Unique unchanged MC input/condition/option content within the same DOI; all matches agree on one legacy parent and mapping is one-to-one. Labels were not used."}
    return {"mapping_kind": "content_verified_legacy_parent_id_bridge_not_original_annotation_history",
            "tables": accepted}


def question_key(q, parent_map):
    parent = q["meta"]["parent_table_id"]
    return (parent_map.get(parent, parent), q["question_type"], q.get("target_key"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-data-dir", type=Path, help="Explicit frozen native JSON directory; use historical/submitted_v0.2.4/data/original_json for historical construction comparison.")
    args = parser.parse_args()
    original = load_datasets(args.reference_data_dir or args.artifact_root / "data" / "original_json")
    rebuilt = load_datasets(args.build_dir / "build_outputs")
    manifest = json.loads((args.build_dir / "rebuild_manifest.json").read_text())
    id_map = make_id_map(original, rebuilt, manifest)
    source_to_stable = {row["reviewed_path"]: row["parent_table_id"] for row in manifest["files"]}
    parent_map = {source_to_stable[path]: mapping["parent_table_id"] for path, mapping in id_map["tables"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reviewed_table_id_map.json").write_text(json.dumps(id_map, indent=2, ensure_ascii=False)+"\n")
    report = {"network_calls": False, "frozen_inputs_mutated": False,
              "reviewed_tables": manifest["reviewed_table_count"],
              "content_verified_legacy_parent_id_count": len(parent_map),
              "comparison_ignores": ["question ID", "provenance metadata", "option/candidate ordering with correctly remapped answers and labels"],
              "warning": "Final reviewed snapshot and archived released builder do not independently reconstruct all frozen submitted questions. Canonical data remain frozen; this report does not authorize replacing them.",
              "subsets": {}}
    for filename in sorted(original):
        old = original[filename]
        new = rebuilt[filename]
        old_ix = {question_key(q, {}): q for q in old}
        new_ix = {question_key(q, parent_map): q for q in new}
        common = set(old_ix) & set(new_ix)
        equal, changed = [], []
        fields = collections.Counter()
        for key in sorted(common, key=dump):
            old_c = canonical_question(old_ix[key]); new_c = canonical_question(new_ix[key])
            if old_c == new_c:
                equal.append(old_ix[key]["id"])
            else:
                diff_keys = [k for k in sorted(set(old_c)|set(new_c)) if old_c.get(k)!=new_c.get(k)]
                fields.update(diff_keys)
                changed.append({"archived_id": old_ix[key]["id"], "reviewed_build_id": new_ix[key]["id"], "changed_field_groups": diff_keys})
        report["subsets"][filename.removesuffix(".json")] = {
            "archived_count": len(old), "rebuilt_count": len(new), "paired_questions": len(common),
            "semantically_identical_paired": len(equal), "changed_paired": len(changed),
            "archived_without_verified_pair": len(set(old_ix)-set(new_ix)),
            "rebuilt_without_verified_pair": len(set(new_ix)-set(old_ix)),
            "changed_field_group_counts": dict(fields), "changed_pairs": changed,
            "archived_unpaired_ids": sorted(old_ix[k]["id"] for k in set(old_ix)-set(new_ix)),
            "rebuilt_unpaired_ids": sorted(new_ix[k]["id"] for k in set(new_ix)-set(old_ix))}
    (args.output_dir / "reconstruction_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    summary = {k: v for k, v in report.items() if k != "subsets"}
    summary["subsets"] = {k: {f: v[f] for f in ("archived_count", "rebuilt_count", "paired_questions", "semantically_identical_paired", "changed_paired", "archived_without_verified_pair", "rebuilt_without_verified_pair", "changed_field_group_counts")} for k, v in report["subsets"].items()}
    (args.output_dir / "construction_reconstruction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"mapped_parents": len(parent_map), "subsets": {k: {f:v[f] for f in ("archived_count","rebuilt_count","paired_questions","semantically_identical_paired","changed_paired")} for k,v in report["subsets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
