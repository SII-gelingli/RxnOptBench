#!/usr/bin/env python3
"""Build an auditable fixed-target-policy candidate without model/API calls.

Input candidate numbering, IDs, inputs and measured condition support are frozen.
Unverified dr/rr axes are null in scoring fields and retained in raw metadata.
No inferred or generic table-acceptance record qualifies as target verification.

The optional --target-ledger must be a JSON object with an ``axes`` list. Each
included axis requires: parent_table_id, axis (dr/rr), human_verified=true,
target_component, other_component, reviewer, verified_at, source_citation and
rows. Each rows item has source_entry, target_amount, other_amount,
source_evidence. Every released row carrying that axis must be mapped. These
fields record an author's supplied human judgement; this script cannot itself
prove the human judgement. With no such supplied ledger all dr/rr are excluded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


VERSION = "0.2.6-target-policy"
TOLERANCE = 1e-9
AXES = ("dr", "rr")
GLOBAL_SCOPE = "all_released_observations_for_parent_table"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entry_key(entry):
    return (entry["row_index"], str(entry["entry_idx"]), str(entry["rxn_id"]))


def observations(question, fib):
    meta = question["meta"]
    if fib:
        rows = [(r["source_entry"], r["score"]) for r in meta["legal_combinations"]]
    else:
        rows = list(zip(meta["option_source_entries"], meta["scores"]))
    rows.extend((r["source_entry"], r["outcome"]) for r in question["input"].get("reference_conditions", []))
    return rows


def collect_rows(datasets):
    tables = defaultdict(dict)
    totals = {}
    for name, dataset in datasets.items():
        for q in dataset:
            table = q["meta"]["parent_table_id"]
            totals[table] = q["meta"]["total_entries_in_source_table"]
            for entry, score in observations(q, name.startswith("fill")):
                key = entry_key(entry)
                if key in tables[table] and tables[table][key]["score"] != score:
                    raise ValueError(f"Conflicting observed scores: {table} {key}")
                tables[table][key] = {"source_entry": copy.deepcopy(entry), "score": copy.deepcopy(score)}
    return tables, totals


def prepare_axes(tables, target_ledger):
    supplied = {}
    for item in target_ledger.get("axes", []):
        key = (item["parent_table_id"], item["axis"])
        if key in supplied or key[1] not in AXES or key[0] not in tables:
            raise ValueError(f"Invalid/duplicate target-ledger axis: {key}")
        if item.get("human_verified") is not True:
            raise ValueError(f"Target ledger not explicitly human verified: {key}")
        for field in ("target_component", "other_component", "reviewer", "verified_at", "source_citation"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Target ledger lacks {field}: {key}")
        if item["target_component"] == item["other_component"]:
            raise ValueError(f"Target and other component must differ: {key}")
        row_map = {}
        for row in item["rows"]:
            row_key = entry_key(row["source_entry"])
            if row_key in row_map or row_key not in tables[key[0]]:
                raise ValueError(f"Invalid/duplicate target-ledger row: {key} {row_key}")
            amounts = [float(row[k]) for k in ("target_amount", "other_amount")]
            if not all(math.isfinite(v) and v >= 0 for v in amounts) or sum(amounts) <= 0:
                raise ValueError(f"Invalid component amounts: {key} {row_key}")
            if not isinstance(row.get("source_evidence"), str) or not row["source_evidence"].strip():
                raise ValueError(f"Missing source-level row evidence: {key} {row_key}")
            row_map[row_key] = {**copy.deepcopy(row), "target_fraction": amounts[0] / sum(amounts)}
        needed = {r for r, value in tables[key[0]].items() if value["score"].get(key[1]) is not None}
        if not needed.issubset(row_map):
            raise ValueError(f"Incomplete axis mapping for {key}: {len(needed - row_map.keys())} missing rows")
        supplied[key] = (copy.deepcopy(item), row_map)
    axes = {}
    ledger = []
    for table in sorted(tables):
        for axis in AXES:
            needed = [key for key, row in tables[table].items() if row["score"].get(axis) is not None]
            if (table, axis) in supplied:
                item, row_map = supplied[(table, axis)]
                status = "include_human_verified_fixed_target"
                reason = "Author-supplied component identity and fixed target, with human verification and row evidence."
            else:
                item, row_map = {}, {}
                status = "exclude_unverified" if needed else "not_observed"
                reason = ("No completed fixed-target component-identity and human-verification ledger supplied; generic table acceptance is insufficient."
                          if needed else "No parsed value for this axis in released observations.")
            record = {"parent_table_id": table, "axis": axis, "status": status,
                      "reason": reason, "observed_rows_with_axis": len(needed),
                      "human_verified": bool(item), "mapping_evidence": item or None}
            axes[(table, axis)] = {**record, "row_map": row_map}
            ledger.append(record)
    return axes, ledger


def effective(raw_score, table, entry, axes):
    """Historic yield and ee/er policy; dr/rr require verified target amounts."""
    clean = copy.deepcopy(raw_score)
    y = float(raw_score["yield"])
    ee, er = raw_score.get("ee"), raw_score.get("er")
    if ee is not None and float(ee) >= 0:
        factor = (100.0 + min(float(ee), 100.0)) / 200.0
    elif er is not None and float(er) >= 0 and math.isfinite(float(er)):
        factor = float(er) / (float(er) + 1.0)
    else:
        factor = 1.0
    utility = max(y, 0.0) * factor
    fractions = {}
    for axis in AXES:
        policy = axes[(table, axis)]
        if raw_score.get(axis) is not None and policy["human_verified"]:
            row = policy["row_map"][entry_key(entry)]
            fraction = row["target_fraction"]
            fractions[axis] = fraction
            # This field is now an explicitly oriented target/other ratio.
            # For a pure target, null avoids non-standard JSON Infinity; use
            # target_fractions as the authoritative scoring input.
            clean[axis] = row["target_amount"] / row["other_amount"] if row["other_amount"] else None
            utility *= fraction
        else:
            clean[axis] = None
    return utility, clean, fractions


def top(values):
    maximum = max(values, default=0.0)
    return [i for i, value in enumerate(values) if math.isclose(value, maximum, rel_tol=0.0, abs_tol=TOLERANCE)]


def relative(values):
    maximum = max(values, default=0.0)
    tied = set(top(values))
    out = []
    for i, value in enumerate(values):
        result = 1.0 if i in tied else round(max(0.0, min(value / maximum, 1.0)), 6) if maximum > 0 else 0.0
        # Prevent non-tied rows becoming exact-optimum merely by rounding.
        if i not in tied and result == 1.0:
            result = 0.999999
        out.append(result)
    return out


def best_record(entries, raw_scores, utilities):
    indices = top(utilities)
    return {"best_yield": max((float(s["yield"]) for s in raw_scores), default=0.0),
            "best_effective_score": round(max(utilities, default=0.0), 6),
            "best_entry_indices": [str(entries[i]["entry_idx"]) for i in indices]}


def score_snapshot(q):
    meta = q["meta"]
    fields = ["best_yield", "best_effective_score", "best_entry_indices", "global_best_yield", "global_best_effective_score", "global_best_entry_indices", "effective_scores", "option_relative_scores"]
    out = {"id": q["id"], "answer": copy.deepcopy(q["answer"]), "meta": {k: copy.deepcopy(meta[k]) for k in fields if k in meta}}
    if "legal_combinations" in meta:
        out["legal_combination_scores"] = [{k: copy.deepcopy(r[k]) for k in ("selection", "effective_score", "relative_score", "source_entry")} for r in meta["legal_combinations"]]
    return out


def relabel_question(q, fib, axes, global_summary):
    q = copy.deepcopy(q)
    meta = q["meta"]
    table = meta["parent_table_id"]
    if fib:
        records = meta["legal_combinations"]
        raw_scores = [copy.deepcopy(r["score"]) for r in records]
        entries = [r["source_entry"] for r in records]
    else:
        raw_scores = copy.deepcopy(meta["scores"])
        entries = meta["option_source_entries"]
    evaluated = [effective(s, table, e, axes) for e, s in zip(entries, raw_scores)]
    utilities = [r[0] for r in evaluated]
    relatives = relative(utilities)
    if fib:
        seen = {}
        for r, raw, result, rel in zip(records, raw_scores, evaluated, relatives):
            key = tuple(r["selection"][k] for k in q["query_keys"])
            if key in seen:
                raise ValueError(f"Duplicate FIB tuple is ambiguous; explicitly aggregate before relabeling: {q['id']}")
            seen[key] = result[0]
            r["raw_observed_score"] = raw
            r["score"] = result[1]
            r["target_fractions"] = result[2]
            r["effective_score"] = round(result[0], 6)
            r["relative_score"] = rel
        q["answer"] = [copy.deepcopy(records[i]["selection"]) for i in top(utilities)]
    else:
        meta["raw_observed_scores"] = raw_scores
        meta["scores"] = [r[1] for r in evaluated]
        meta["target_fractions"] = [r[2] for r in evaluated]
        meta["effective_scores"] = [round(v, 6) for v in utilities]
        meta["option_relative_scores"] = relatives
        q["answer"] = top(utilities)
    meta.update(best_record(entries, raw_scores, utilities))
    if not fib:
        meta.update({"global_" + k: v for k, v in global_summary[table].items()})
        meta["global_best_scope"] = GLOBAL_SCOPE
    meta["best_scope"] = "current_question_observed_candidates"
    meta["selectivity_policy"] = {
        "version": VERSION,
        "ee_er": "unchanged_archived_major_fraction_ee_preferred",
        "dr": axes[(table, "dr")]["status"],
        "rr": axes[(table, "rr")]["status"],
        "raw_observed_scores": "Archived parsed measurements retained without target-identity claims.",
        "reference_outcomes": "Input reference outcomes unchanged; raw observed ratios are not proof of verified target orientation.",
    }
    return q


def write_lines(path, objects):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for obj in objects:
            handle.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def build(source, output, target_ledger_path=None):
    source, output = Path(source), Path(output)
    names = sorted(p.name for p in (source / "data/original_json").glob("*.json"))
    if len(names) != 8:
        raise ValueError(f"Expected eight datasets; found {len(names)}")
    datasets = {name: read(source / "data/original_json" / name) for name in names}
    for name, dataset in datasets.items():
        if any("selectivity_policy" in q.get("meta", {}) for q in dataset):
            raise ValueError(f"Source must be the frozen historical snapshot, not an already relabeled candidate: {name}")
        if len({q["id"] for q in dataset}) != len(dataset):
            raise ValueError(f"Duplicate IDs within {name}")
        lines = [json.loads(s) for s in (source / "data/jsonl" / name.replace(".json", ".jsonl")).read_text(encoding="utf-8").splitlines()]
        if lines != dataset:
            raise ValueError(f"Input native/JSONL mismatch: {name}")
    tables, totals = collect_rows(datasets)
    target_ledger = read(target_ledger_path) if target_ledger_path else {"axes": []}
    axes, ledger = prepare_axes(tables, target_ledger)
    global_summary = {}
    for table, rows in tables.items():
        entries, scores, utilities = [], [], []
        for row in rows.values():
            entries.append(row["source_entry"])
            scores.append(row["score"])
            utilities.append(effective(row["score"], table, row["source_entry"], axes)[0])
        global_summary[table] = best_record(entries, scores, utilities)
    summary = {"version": VERSION, "policy": "Only human-verified fixed-target dr/rr axes participate; all other dr/rr retained in raw metadata.",
               "target_ledger_supplied": bool(target_ledger_path), "included_axis_count": sum(x["human_verified"] for x in ledger),
               "axis_status_counts": dict(Counter(x["status"] for x in ledger)),
               "parent_tables": len(tables), "released_unique_observations": sum(len(r) for r in tables.values()),
               "stated_source_table_entries": sum(totals.values()),
               "parents_with_all_source_rows_represented": sum(len(rows) == totals[t] for t, rows in tables.items()),
               "global_best_scope": GLOBAL_SCOPE, "subsets": {}}
    manifest, snapshots, changes = [], [], []
    for name, old_dataset in datasets.items():
        new_dataset = []
        counts = Counter()
        for old in old_dataset:
            new = relabel_question(old, name.startswith("fill"), axes, global_summary)
            # No prompt, chemistry, candidate, ID, or measured support changes.
            for key in ("id", "question_type", "input", "options", "query_keys", "target_key", "varying_keys"):
                if new.get(key) != old.get(key):
                    raise ValueError(f"Frozen candidate/input changed: {name} {old['id']} {key}")
            old_s, new_s = score_snapshot(old), score_snapshot(new)
            counts["questions"] += 1
            counts["answer_set_changed"] += old["answer"] != new["answer"]
            if name.startswith("fill"):
                oe = [r["effective_score"] for r in old["meta"]["legal_combinations"]]
                ne = [r["effective_score"] for r in new["meta"]["legal_combinations"]]
                or_ = [r["relative_score"] for r in old["meta"]["legal_combinations"]]
                nr = [r["relative_score"] for r in new["meta"]["legal_combinations"]]
                counts["duplicate_selection_tuples"] += len(ne) - len({tuple(r["selection"][k] for k in old["query_keys"]) for r in old["meta"]["legal_combinations"]})
            else:
                oe, ne = old["meta"]["effective_scores"], new["meta"]["effective_scores"]
                or_, nr = old["meta"]["option_relative_scores"], new["meta"]["option_relative_scores"]
            counts["effective_changed"] += oe != ne
            counts["relative_changed"] += or_ != nr
            if old_s != new_s:
                counts["scoring_snapshot_changed"] += 1
                changes.append({"subset": name.removesuffix(".json"), "id": old["id"], "parent_table_id": old["meta"]["parent_table_id"], "old": old_s, "new": new_s})
            snapshots.append({"subset": name.removesuffix(".json"), **old_s})
            new_dataset.append(new)
        native = output / "data/original_json" / name
        jsonl = output / "data/jsonl" / name.replace(".json", ".jsonl")
        write(native, new_dataset)
        write_lines(jsonl, new_dataset)
        if read(native) != [json.loads(s) for s in jsonl.read_text(encoding="utf-8").splitlines()]:
            raise AssertionError(f"Output native/JSONL mismatch: {name}")
        counts["native_jsonl_equivalent"] = True
        summary["subsets"][name.removesuffix(".json")] = dict(counts)
        manifest.append({"subset": name.removesuffix(".json"), "records": len(new_dataset),
                         "historical_native_path": "data/original_json/" + name,
                         "historical_native_sha256": digest(source / "data/original_json" / name),
                         "native_path": str(native.relative_to(output)), "native_sha256": digest(native),
                         "jsonl_path": str(jsonl.relative_to(output)), "jsonl_sha256": digest(jsonl)})
    write(output / "metadata/target_policy_ledger.json", {"version": VERSION, "axes": ledger})
    write(output / "metadata/policy_summary.json", summary)
    write(output / "metadata/relabel_manifest.json", {"version": VERSION, "script_sha256": digest(__file__), "inputs_outputs": manifest,
          "target_ledger_input_sha256": digest(target_ledger_path) if target_ledger_path else None})
    write_lines(output / "metadata/historical_scoring_snapshot.jsonl", snapshots)
    write_lines(output / "metadata/per_question_changes.jsonl", changes)
    (output / "VERSION").write_text(VERSION + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Frozen historical root containing data/original_json and data/jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Separate output directory; never overwrite the historical source")
    parser.add_argument("--target-ledger", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("--output must differ from --source to preserve the historical snapshot")
    summary = build(args.source, args.output, args.target_ledger)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
