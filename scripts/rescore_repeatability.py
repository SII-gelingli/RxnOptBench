#!/usr/bin/env python3
"""Rescore the three archived full repeat runs after exact semantic index mapping.

This script never makes model/API calls and never modifies its input files.
Mapping uses complete condition objects (including text and SMILES), not answer
labels, utilities, source-row indices, or the predictions being evaluated.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] if len(HERE.parents) > 3 else HERE
RUN_ROOT = ROOT / "rebuttal/04_analysis_and_audits/analysis/deepseek_repeatability/outputs"
SOURCE_RELEASE = ROOT / "rebuttal/05_anonymous_code_release/RxnOptBench_anonymous_review_upload_0.2.5_eval_only/data/original_json"
DEFAULT_TARGET = ROOT / "camera_ready_submission/alignment_work/dataset_candidate/data/original_json"
HEADLINE = [
    ("mc_single", "accuracy", "MC-single Acc.", True),
    ("mc_single", "score", "MC-single Score", False),
    ("mc_all", "accuracy", "MC-all Acc.", True),
    ("mc_all", "score", "MC-all Score", False),
    ("fib_all", "hit", "FIB Hit", True),
    ("fib_all", "strict_accuracy", "FIB Strict Acc.", True),
    ("fib_all", "score", r"FIB Score$\,\mid\,$Hit", False),
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def dataset_index(dataset):
    index = {sample["id"]: sample for sample in dataset}
    assert len(index) == len(dataset), "Duplicate IDs within a subset"
    return index


def permutation(source_values, target_values):
    """Exact bijection; refuse missing, added, or ambiguous condition objects."""
    source = [canonical(value) for value in source_values]
    target = [canonical(value) for value in target_values]
    assert len(set(source)) == len(source), "Ambiguous source conditions"
    assert len(set(target)) == len(target), "Ambiguous target conditions"
    assert Counter(source) == Counter(target), "Candidate condition sets differ"
    target_lookup = {value: i for i, value in enumerate(target)}
    return {i: target_lookup[value] for i, value in enumerate(source)}


def make_mapping(source, target, kind):
    old, new = dataset_index(source), dataset_index(target)
    assert old.keys() == new.keys(), "Question ID sets differ"
    result = {}
    moved = 0
    context_changed = []
    for sample_id in old:
        a, b = old[sample_id], new[sample_id]
        assert a["question_type"] == b["question_type"]
        if kind == "mc":
            mapping = permutation(a["options"], b["options"])
            moved += sum(i != j for i, j in mapping.items())
            proof = {
                "old_to_new": mapping,
                "source_candidate_sha256": fingerprint(a["options"]),
                "target_candidate_sha256": fingerprint(b["options"]),
            }
            context_a, context_b = a["input"], b["input"]
        else:
            assert a["query_keys"] == b["query_keys"], "Query key/order changed"
            mapping = {}
            for key in a["query_keys"]:
                av, bv = a["input"]["candidate_values"][key], b["input"]["candidate_values"][key]
                assert all(v["index"] == i for i, v in enumerate(av)), "Source index != position"
                assert all(v["index"] == i for i, v in enumerate(bv)), "Target index != position"
                mapping[key] = permutation([v["value"] for v in av], [v["value"] for v in bv])
                moved += sum(i != j for i, j in mapping[key].items())
            proof = {
                "old_to_new": mapping,
                "source_candidate_sha256": fingerprint(a["input"]["candidate_values"]),
                "target_candidate_sha256": fingerprint(b["input"]["candidate_values"]),
            }
            context_a = {k: v for k, v in a["input"].items() if k != "candidate_values"}
            context_b = {k: v for k, v in b["input"].items() if k != "candidate_values"}
        proof["noncandidate_input_context_identical"] = context_a == context_b
        if context_a != context_b:
            context_changed.append(sample_id)
        result[sample_id] = proof
    return result, {
        "questions": len(source),
        "changed_candidate_positions": moved,
        "all_candidate_condition_objects_match_bijectively": True,
        "noncandidate_input_context_changed_question_ids": context_changed,
    }


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def map_predictions(predictions, mappings, kind):
    mapped = {}
    changed = 0
    for sample_id, prediction in predictions.items():
        if sample_id not in mappings:
            continue
        mapping = mappings[sample_id]["old_to_new"]
        if kind == "mc":
            new = mapping.get(prediction, -1) if integer(prediction) else -1
        elif isinstance(prediction, dict):
            new = copy.deepcopy(prediction)
            for key, keymap in mapping.items():
                if key in prediction:
                    new[key] = keymap.get(prediction[key], -1) if integer(prediction[key]) else -1
        else:
            new = prediction
        changed += new != prediction
        mapped[sample_id] = new
    return mapped, changed


def score(dataset, predictions, kind):
    """Use the paper's full-N MC/FIB Hit/Strict and conditional FIB Score."""
    n = len(dataset)
    hit = exact = valid_count = 0
    score_sum = 0.0
    details = []
    for sample in dataset:
        sample_id = sample["id"]
        pred = predictions.get(sample_id)
        if kind == "mc":
            valid = integer(pred) and 0 <= pred < len(sample["options"])
            relative = sample["meta"]["option_relative_scores"][pred] if valid else 0.0
            best = valid and pred in sample["answer"]
            hit_here = valid
        else:
            keys = sample["query_keys"]
            valid = isinstance(pred, dict) and all(
                key in pred and integer(pred[key])
                and 0 <= pred[key] < len(sample["input"]["candidate_values"][key])
                for key in keys
            )
            lookup = {
                tuple(record["selection"][key] for key in keys): record
                for record in sample["meta"]["legal_combinations"]
            }
            record = lookup.get(tuple(pred[key] for key in keys)) if valid else None
            hit_here = record is not None
            relative = record["relative_score"] if hit_here else None
            best = hit_here and relative == 1.0
        valid_count += valid
        hit += hit_here
        exact += best
        score_sum += relative if relative is not None else 0.0
        details.append({"id": sample_id, "valid": valid, "hit": hit_here, "best": best, "relative_score": relative})
    out = {
        "questions": n,
        "num_missing": len({sample["id"] for sample in dataset} - predictions.keys()),
        "num_extra": len(predictions.keys() - {sample["id"] for sample in dataset}),
        "valid_predictions": valid_count,
        "exact_count": exact,
        "score_sum": score_sum,
    }
    if kind == "mc":
        out.update(accuracy=exact / n, score=score_sum / n)
    else:
        out.update(hit_count=hit, hit=hit / n, strict_accuracy=exact / n, score=score_sum / hit if hit else 0.0)
    return out, details


def summaries(rows, label):
    out = []
    for task in dict.fromkeys(row["task"] for row in rows):
        group = [row[label] for row in rows if row["task"] == task]
        for metric in ("accuracy", "hit", "strict_accuracy", "score"):
            if metric not in group[0]:
                continue
            values = [row[metric] for row in group]
            out.append({
                "task": task, "metric": metric, "runs": len(values), "values": values,
                "mean": statistics.mean(values), "sd": statistics.stdev(values),
                "min": min(values), "max": max(values), "range": max(values) - min(values),
            })
    return out


def latex(summary):
    lookup = {(row["task"], row["metric"]): row for row in summary}
    lines = [
        "% Generated by rescore_repeatability.py; target-policy labels and exact semantic index mapping.",
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{DeepSeek-V4-Pro repeatability audit.} Means and min--max ranges across three independent reruns, rescored under the released target-verification policy. The model predictions are archived and were not regenerated.}",
        r"\label{tab:deepseek_repeatability}", r"\begin{tabular}{lrr}", r"\toprule",
        r"Task and metric & Rerun mean & Rerun range \\", r"\midrule",
    ]
    for task, metric, title, percent in HEADLINE:
        item = lookup[task, metric]
        if percent:
            mean = f"{100 * item['mean']:.2f}\\%"
            spread = f"{100 * item['min']:.2f}--{100 * item['max']:.2f}\\%"
        else:
            mean = f"{item['mean']:.3f}"
            spread = f"{item['min']:.3f}--{item['max']:.3f}"
        lines.append(f"{title} & {mean} & {spread} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, help="Public benchmark artifact root; uses only files inside the supplied public layout by default")
    parser.add_argument("--runs-root", type=Path, help="Directory containing run_01..03 and a run manifest")
    parser.add_argument("--source-dir", type=Path, help="Archived input datasets; only the basename of each manifest dataset path is used")
    parser.add_argument("--dataset-root", "--target-dir", dest="dataset_root", type=Path, help="Datasets to score; defaults to ARTIFACT_ROOT/data/original_json in public mode")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, help="Output directory (default: beside this script)")
    args = parser.parse_args()
    public_mode = args.artifact_root is not None
    artifact_root = args.artifact_root.resolve() if public_mode else ROOT
    if args.runs_root:
        run_root = args.runs_root.resolve()
    elif public_mode:
        run_root = artifact_root / "results/deepseek_repeatability/outputs"
        if not run_root.is_dir():
            run_root = artifact_root / "repeatability/archived_outputs"
    else:
        run_root = RUN_ROOT
    source_dir = args.source_dir.resolve() if args.source_dir else (
        artifact_root / "historical/submitted_v0.2.4/data/original_json" if public_mode else SOURCE_RELEASE
    )
    target_dir = args.dataset_root.resolve() if args.dataset_root else (
        artifact_root / "data/original_json" if public_mode else DEFAULT_TARGET
    )
    output_dir = args.output_dir.resolve() if args.output_dir else HERE
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        manifest_path = run_root / "original_run_manifest.json"
    manifest = read(manifest_path)
    hashes = []

    def relative_path(path):
        try:
            return str(path.resolve().relative_to(artifact_root))
        except ValueError:
            # Explicit overrides and a script installed outside the artifact
            # remain usable without disclosing machine-local absolute paths.
            return "external_inputs/" + path.name

    def record_file(path, role):
        hashes.append({"role": role, "path": relative_path(path), "sha256": sha(path), "bytes": path.stat().st_size})

    record_file(Path(__file__).resolve(), "audit_script")
    record_file(manifest_path, "repeat_run_manifest")
    rows, mappings, source_match = [], {}, []
    for task, metadata in manifest["datasets"].items():
        filename = Path(metadata["path"]).name
        # Public runs must never resolve legacy manifest paths against a private
        # project checkout. The published archive supplies the frozen inputs.
        source_path = source_dir / filename if public_mode or args.source_dir else ROOT / metadata["path"]
        release_path = source_dir / filename
        target_path = target_dir / filename
        source, target = read(source_path), read(target_path)
        assert len(source) == len(target) == metadata["questions"]
        kind = "fib" if task.startswith("fib") else "mc"
        mapping, mapping_summary = make_mapping(source, target, kind)
        mappings[task] = {"summary": mapping_summary, "by_id": mapping}
        equal_release = source_path.read_bytes() == release_path.read_bytes()
        assert equal_release, "Run-manifest dataset differs from archived release; requires provenance adjudication"
        source_check = {"task": task, "archived_source_path": relative_path(release_path), "dataset_selected_by_manifest_basename": public_mode or bool(args.source_dir)}
        if not public_mode and not args.source_dir:
            source_check["source_identical_to_0.2.5_eval_only"] = equal_release
        source_match.append(source_check)
        record_file(source_path, "inference_manifest_dataset")
        record_file(release_path, "archived_source_crosscheck")
        record_file(target_path, "target_policy_dataset")
        for run in range(1, manifest["runs"] + 1):
            run_tag = f"run_{run:02d}"
            prediction_path = run_root / run_tag / f"{task}.json"
            predictions = read(prediction_path)
            mapped, changed = map_predictions(predictions, mapping, kind)
            original, old_details = score(source, predictions, kind)
            current, new_details = score(target, mapped, kind)
            assert original["num_missing"] == current["num_missing"] == 0
            assert original["num_extra"] == current["num_extra"] == 0
            if kind == "fib":
                assert original["hit_count"] == current["hit_count"], "Semantic mapping changed observed-combination membership"
                assert all(a["hit"] == b["hit"] for a, b in zip(old_details, new_details)), "Semantic mapping changed an individual observed-combination hit"
            rows.append({"run": run_tag, "task": task, "prediction_indices_changed": changed, "source_policy": original, "target_policy": current})
            write(output_dir / "mapped_predictions" / run_tag / f"{task}.json", mapped)
            write(output_dir / "details" / run_tag / f"{task}.json", {"source_policy": old_details, "target_policy": new_details})
            record_file(prediction_path, "archived_parsed_prediction")
    source_summary, target_summary = summaries(rows, "source_policy"), summaries(rows, "target_policy")
    report = {
        "model": manifest["model"], "run_count": manifest["runs"],
        "questions_per_run": manifest["questions_per_run"],
        "archived_calls": manifest["questions_per_run"] * manifest["runs"],
        "new_api_calls": 0,
        "target_dataset_directory": relative_path(target_dir),
        "archived_source_directory": relative_path(source_dir),
        "provenance_limit": "The original repeat-run manifest records dataset paths and question counts, but not contemporaneous input-file hashes. This audit reads the published archived source files selected by manifest basename in public mode, records their hashes and exact semantic mappings, and does not reconstruct unrecorded historical request bytes.",
        "metric_contract": {"mc_accuracy": "best choices / all questions", "mc_score": "sum relative utility / all questions", "fib_hit": "observed legal tuples / all questions", "fib_strict_accuracy": "best observed tuples / all questions", "fib_score_given_hit": "sum relative utility / observed legal tuple hits"},
        "mapping_rule": "Exact bijection of complete condition objects, including text and SMILES; independent of answer labels, utility, predictions, and source-row labels.",
        "source_release_checks": source_match,
        "mapping_summaries": {task: mapping["summary"] for task, mapping in mappings.items()},
        "rows": rows, "source_policy_summary": source_summary, "target_policy_summary": target_summary,
        "input_hashes": hashes,
    }
    write(output_dir / "mapping_proof.json", mappings)
    write(output_dir / "repeatability_rescore.json", report)
    with (output_dir / "metrics_full8.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["run", "task", "policy", "questions", "num_missing", "valid_predictions", "hit_count", "exact_count", "accuracy", "hit", "strict_accuracy", "score"]
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for policy in ("source_policy", "target_policy"):
                writer.writerow({"run": row["run"], "task": row["task"], "policy": policy, **row[policy]})
    with (output_dir / "summary_full8.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["policy", "task", "metric", "runs", "mean", "sd", "min", "max", "range"]
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for policy, summary in (("source_policy", source_summary), ("target_policy", target_summary)):
            for row in summary:
                writer.writerow({"policy": policy, **row})
    (output_dir / "update_repeatability.tex").write_text(latex(target_summary), encoding="utf-8")
    source_note = "All 8 source dataset files are selected by manifest basename from the supplied archived source directory." if public_mode or args.source_dir else "All 8 manifest dataset files match archived 0.2.5-eval-only bytes exactly."
    text = ["DeepSeek-V4-Pro repeatability rescoring", "", "Archived model calls: 3 x 4,773 = 14,319; new API calls: 0.", "Input manifests and all 24 parsed prediction files are hashed in repeatability_rescore.json.", source_note, "All candidate maps are exact bijections over the complete condition objects.", "Score rules: MC full-N; FIB Hit/Strict full-N; FIB Score conditional on legal hits.", "", "task\tmetric\tmean\tmin\tmax\tsample_sd"]
    text.extend(f"{row['task']}\t{row['metric']}\t{row['mean']:.9f}\t{row['min']:.9f}\t{row['max']:.9f}\t{row['sd']:.9f}" for row in target_summary)
    text.extend(["", "These are rescored archived outputs, not reruns under altered reference prompts.", "Any noncandidate input changes are itemized in mapping_proof.json.", ""])
    (output_dir / "README.txt").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"all_source_files_read_from_archive": public_mode or bool(args.source_dir), "target_policy_summary": target_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
