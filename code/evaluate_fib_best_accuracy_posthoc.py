#!/usr/bin/env python3
"""
对已保存的 FIB（fib_all）预测 JSON 补算「最优答案」类指标，无需重新跑推理。

V6 官方 evaluate_fill_in_blank.py 已给出：
  - hit_rate：预测组合对应来源表实测联合行的比例
  - best_hit_rate_among_scored：仅在「命中并计分」子集上，relative_score==1 的比例

本脚本保留历史导出字段，并将与论文一致的严格准确率写为：
  - strict_accuracy = best_hit_rate_over_all
  - best_hit_rate_over_all = 预测为「最佳实测组合」的样本数 / 已提交预测总数

（等价于：在全体样本上统计 best_hit，而不仅限于命中子集。）

与 `run_open_source.sh` / `export_summary_to_excel.py` 中「fib_all_reference vs fib_all (same-question subset)」一致：
  - 取 reference 评测集中的题目，将 id 映射回 classic id（去 `_ref` / `_ref_control` 后缀）；
  - 用 **classic 题面 + 来源表实测组合表** 评测 **fib_all.json（无 reference 提示）** 的预测，得到「同等题目子集」指标，
    可与 **fib_all_reference.json + reference 题** 在同一批 id 上对比。

用法示例：

  # 扫描发布结果目录下各模型的 fib_all*.json
  python code/evaluate_fib_best_accuracy_posthoc.py \\
    --results-root results/predictions

  # 单次评测
  python code/evaluate_fib_best_accuracy_posthoc.py \\
    --dataset data/original_json/fill_in_blank_all_blank.json \\
    --prediction results/predictions/open_source/SomeModel/fib_all.json

  # 指定经典 / reference 两套数据（扫描时 fib_all 用 classic，fib_all_reference 用 reference）
  python code/evaluate_fib_best_accuracy_posthoc.py \\
    --results-root results/predictions \\
    --dataset-classic .../fill_in_blank_all_blank.json \\
    --dataset-reference .../fill_in_blank_all_blank_reference.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CLASSIC = (
    ROOT
    / "data"
    / "original_json"
    / "fill_in_blank_all_blank.json"
)
DEFAULT_DATASET_REFERENCE = (
    ROOT
    / "data"
    / "original_json"
    / "fill_in_blank_all_blank_reference.json"
)
DEFAULT_RESULTS_ROOT = ROOT / "results" / "predictions"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset_index(dataset: list) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for sample in dataset:
        sid = sample["id"]
        if sid in index:
            raise ValueError(f"duplicate id in dataset: {sid}")
        index[sid] = sample
    return index


def _selection_key(selection: dict, query_keys: List[str]) -> str:
    return json.dumps({k: selection[k] for k in query_keys}, sort_keys=True, ensure_ascii=False)


def _validate_prediction(pred: Any, sample: dict) -> Tuple[bool, str]:
    query_keys = sample.get("query_keys", [])
    if not isinstance(pred, dict):
        return False, "prediction_not_object"
    for key in query_keys:
        if key not in pred:
            return False, f"missing_key:{key}"
        if not isinstance(pred[key], int):
            return False, f"non_int_index:{key}"
        n_candidates = len(sample["input"]["candidate_values"].get(key, []))
        if pred[key] < 0 or pred[key] >= n_candidates:
            return False, f"index_out_of_range:{key}"
    return True, "ok"


def _prediction_matches_canonical_answer(pred: dict, sample: dict) -> bool:
    """与数据集中显式 answer 列表是否一致（通常即文献最优条）。"""
    answers = sample.get("answer")
    if not isinstance(answers, list) or not answers:
        return False
    query_keys = sample.get("query_keys", [])
    for ref in answers:
        if not isinstance(ref, dict):
            continue
        ok = all(pred.get(k) == ref.get(k) for k in query_keys)
        if ok:
            return True
    return False


def variant_id_to_classic_id(variant_id: str) -> str:
    """与 run_open_source.sh 中 write_paired_classic_subset 一致。"""
    if variant_id.endswith("_ref_control"):
        return variant_id[: -len("_ref_control")]
    if variant_id.endswith("_ref"):
        return variant_id[: -len("_ref")]
    return variant_id


def build_paired_classic_subset(
    classic_dataset: list,
    variant_dataset: list,
    classic_predictions: dict,
) -> Tuple[list, dict]:
    """
    按 reference 题顺序，取出对应 classic 样本与 fib_all（无 ref）预测子集。
    评测时使用 classic 的 legal_combinations（与 summary 里 [fib_all on paired subset] 一致）。
    """
    classic_index = {s["id"]: s for s in classic_dataset}
    subset_dataset: List[dict] = []
    subset_predictions: Dict[str, Any] = {}
    for sample in variant_dataset:
        vid = sample["id"]
        cid = variant_id_to_classic_id(vid)
        if cid not in classic_index:
            raise KeyError(
                f"paired subset: classic id not found for variant id={vid!r} -> {cid!r}"
            )
        subset_dataset.append(classic_index[cid])
        if cid in classic_predictions:
            subset_predictions[cid] = classic_predictions[cid]
    return subset_dataset, subset_predictions


def evaluate_fib_metrics(dataset: list, predictions: dict) -> Dict[str, Any]:
    """
    返回与 V6 evaluate_fill_in_blank 一致的核心量，并保留历史字段
    best_hit_rate_over_all 及 canonical_answer_rate_over_all。
    """
    ds_index = build_dataset_index(dataset)
    pred_ids = set(predictions.keys())
    ds_ids = set(ds_index.keys())
    missing = ds_ids - pred_ids
    extra = pred_ids - ds_ids
    eval_ids = sorted(ds_ids & pred_ids)
    n_total = len(eval_ids)

    num_scored = 0
    num_unhit = 0
    num_invalid = 0
    hit_count = 0
    exact_best_count = 0
    canonical_match_count = 0
    relative_score_sum = 0.0

    for sample_id in eval_ids:
        sample = ds_index[sample_id]
        prediction = predictions[sample_id]
        query_keys = sample.get("query_keys", [])
        valid, _reason = _validate_prediction(prediction, sample)

        lookup = {
            _selection_key(rec["selection"], query_keys): rec
            for rec in sample["meta"].get("legal_combinations", [])
        }
        matched_record = None
        if valid:
            matched_record = lookup.get(_selection_key(prediction, query_keys))

        hit = matched_record is not None
        relative_score = matched_record["relative_score"] if matched_record else None
        best_hit = relative_score == 1.0 if matched_record else False

        if not valid:
            num_invalid += 1
        elif not hit:
            num_unhit += 1
        else:
            hit_count += 1
            num_scored += 1
            relative_score_sum += float(relative_score)
            if best_hit:
                exact_best_count += 1

        if valid and _prediction_matches_canonical_answer(prediction, sample):
            canonical_match_count += 1

    return {
        "num_samples": n_total,
        "num_missing_predictions": len(missing),
        "num_extra_predictions": len(extra),
        "num_scored": num_scored,
        "num_unhit": num_unhit,
        "num_invalid": num_invalid,
        "hit_rate": hit_count / n_total if n_total else 0.0,
        "avg_relative_score_on_scored": relative_score_sum / num_scored if num_scored else 0.0,
        "best_hit_count": exact_best_count,
        "best_hit_rate_among_scored": exact_best_count / num_scored if num_scored else 0.0,
        "strict_accuracy": exact_best_count / n_total if n_total else 0.0,
        "best_hit_rate_over_all": exact_best_count / n_total if n_total else 0.0,
        "canonical_answer_match_count": canonical_match_count,
        "canonical_answer_rate_over_all": canonical_match_count / n_total if n_total else 0.0,
    }


def is_primary_fib_prediction_json(name: str) -> bool:
    """主预测文件：fib_all.json / fib_all_reference.json 等，排除 *_tokens 等辅助文件。"""
    if not name.endswith(".json") or not name.startswith("fib_all"):
        return False
    base = name[: -len(".json")]
    noise_suffixes = ("_tokens", "_raw", "_reasoning", "_parse_stats", "_parse_methods")
    if any(base.endswith(s) for s in noise_suffixes):
        return False
    return base == "fib_all" or base.startswith("fib_all_")


def model_directories(results_root: Path) -> List[Path]:
    """Accept either a tier directory or the release's tier/model tree."""
    direct = sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path / "fib_all.json").is_file()
    )
    if direct:
        return direct
    return sorted(
        model
        for tier in results_root.iterdir()
        if tier.is_dir()
        for model in tier.iterdir()
        if model.is_dir() and (model / "fib_all.json").is_file()
    )


@dataclass
class ScanEntry:
    model_dir: str
    prediction_file: str
    dataset_file: str
    metrics: Dict[str, Any]


def scan_results(
    results_root: Path,
    dataset_classic: Path,
    dataset_reference: Path,
) -> List[ScanEntry]:
    ds_classic = load_json(dataset_classic)
    ds_ref = load_json(dataset_reference)
    entries: List[ScanEntry] = []

    if not results_root.is_dir():
        return entries

    for model_dir in model_directories(results_root):
        if model_dir.name.startswith("_"):
            continue

        for pred_path in sorted(model_dir.iterdir()):
            if not pred_path.is_file():
                continue
            name = pred_path.name
            if not is_primary_fib_prediction_json(name):
                continue
            tag = name[: -len(".json")]
            if tag == "fib_all":
                dataset_path = dataset_classic
                dataset_obj = ds_classic
            elif tag == "fib_all_reference" or tag.startswith("fib_all_reference"):
                dataset_path = dataset_reference
                dataset_obj = ds_ref
            else:
                # fib_all_other — 尽量用 classic 数据（可能 id 对不上则跳过）
                dataset_path = dataset_classic
                dataset_obj = ds_classic

            try:
                predictions = load_json(pred_path)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(predictions, dict):
                continue

            metrics = evaluate_fib_metrics(dataset_obj, predictions)
            metrics["prediction_tag"] = tag
            entries.append(
                ScanEntry(
                    model_dir=str(model_dir),
                    prediction_file=str(pred_path),
                    dataset_file=str(dataset_path),
                    metrics=metrics,
                )
            )
    return entries


def compute_fib_same_question_pairs(
    results_root: Path,
    dataset_classic: Path,
    dataset_reference: Path,
) -> List[Dict[str, Any]]:
    """
    每个同时含 fib_all.json 与 fib_all_reference.json 的模型目录，计算
    reference 全量 vs classic 在「与 reference 同题」子集上的指标。
    """
    ds_classic = load_json(dataset_classic)
    ds_ref = load_json(dataset_reference)
    out: List[Dict[str, Any]] = []

    for model_dir in model_directories(results_root):
        if model_dir.name.startswith("_"):
            continue
        p_classic = model_dir / "fib_all.json"
        p_ref = model_dir / "fib_all_reference.json"
        if not p_classic.is_file() or not p_ref.is_file():
            continue
        try:
            pred_classic = load_json(p_classic)
            pred_ref = load_json(p_ref)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(pred_classic, dict) or not isinstance(pred_ref, dict):
            continue

        subset_ds, subset_pred = build_paired_classic_subset(
            ds_classic, ds_ref, pred_classic
        )
        m_ref = evaluate_fib_metrics(ds_ref, pred_ref)
        m_paired = evaluate_fib_metrics(subset_ds, subset_pred)

        out.append(
            {
                "model_dir": str(model_dir),
                "num_questions_reference": m_ref["num_samples"],
                "num_questions_paired_eval": m_paired["num_samples"],
                "fib_all_reference": {
                    "prediction_file": str(p_ref),
                    "dataset_file": str(dataset_reference),
                    "metrics": m_ref,
                },
                "fib_all_on_paired_classic_subset": {
                    "prediction_file": str(p_classic),
                    "dataset_file": str(dataset_classic),
                    "note": "无 reference 提示的 fib_all 预测，在同一批题 id 上映射到 classic 题面后评测",
                    "metrics": m_paired,
                },
            }
        )
    return out


def print_metrics_line(label: str, m: Dict[str, Any]) -> None:
    n = m["num_samples"]
    print(f"\n=== {label} ===")
    print(f"  samples:              {n}")
    print(f"  hit_rate:             {m['hit_rate']:.1%}")
    print(f"  avg_rel_score(scored):{m['avg_relative_score_on_scored']:.3f}")
    print(
        f"  best among Hits:      {m['best_hit_rate_among_scored']:.1%}  "
        f"({m['best_hit_count']}/{m['num_scored']})"
    )
    print(
        f"  strict accuracy:      {m['strict_accuracy']:.1%}  "
        f"({m['best_hit_count']}/{n})"
    )
    print(
        f"  canonical answer rate:{m['canonical_answer_rate_over_all']:.1%}  "
        f"({m['canonical_answer_match_count']}/{n})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc FIB best-answer metrics on saved JSON.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="扫描该目录下各子目录的 fib_all*.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="与 --prediction 联用：单个评测集 JSON",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        default=None,
        help="单个预测文件路径（与 --dataset 一起使用）",
    )
    parser.add_argument(
        "--dataset-classic",
        type=Path,
        default=Path(DEFAULT_DATASET_CLASSIC),
        help="扫描时 fib_all.json 使用的评测集",
    )
    parser.add_argument(
        "--dataset-reference",
        type=Path,
        default=Path(DEFAULT_DATASET_REFERENCE),
        help="扫描时 fib_all_reference*.json 使用的评测集",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="写入汇总 JSON；默认扫描时写到 results-root/fib_best_accuracy_posthoc.json",
    )
    args = parser.parse_args()

    if args.prediction:
        if not args.dataset:
            parser.error("--prediction 需要同时提供 --dataset")
        ds = load_json(args.dataset)
        pred = load_json(args.prediction)
        m = evaluate_fib_metrics(ds, pred)
        m["prediction_file"] = str(args.prediction)
        m["dataset_file"] = str(args.dataset)
        print_metrics_line(Path(args.prediction).name, m)
        out = args.output_json
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)
        return

    if not args.dataset_classic.is_file():
        raise SystemExit(f"classic dataset not found: {args.dataset_classic}")
    if not args.dataset_reference.is_file():
        raise SystemExit(f"reference dataset not found: {args.dataset_reference}")

    entries = scan_results(args.results_root, args.dataset_classic, args.dataset_reference)
    if not entries:
        print(f"No fib_all*.json found under {args.results_root}")
        return

    paired = compute_fib_same_question_pairs(
        args.results_root, args.dataset_classic, args.dataset_reference
    )
    report = {
        "results_root": str(args.results_root),
        "dataset_classic": str(args.dataset_classic),
        "dataset_reference": str(args.dataset_reference),
        "runs": [asdict(e) for e in entries],
        "fib_same_question_pairs": paired,
    }
    out_json = args.output_json or (args.results_root / "fib_best_accuracy_posthoc.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_json}")

    for e in entries:
        rel = os.path.relpath(e.prediction_file, str(args.results_root))
        print_metrics_line(rel, e.metrics)

    if paired:
        print("\n" + "=" * 60)
        print(" FIB: reference 题上「带 ref」vs「不带 ref、同等 classic 题」")
        print("=" * 60)
        for block in paired:
            rel_m = os.path.relpath(block["model_dir"], str(args.results_root))
            print(f"\n>>> {rel_m}")
            print(
                f"    题目数(reference): {block['num_questions_reference']} | "
                f"paired 评测交集: {block['num_questions_paired_eval']}"
            )
            print_metrics_line(
                "fib_all_reference（有 reference 提示）",
                block["fib_all_reference"]["metrics"],
            )
            print_metrics_line(
                "fib_all on paired classic subset（无 ref 提示，同题 classic 表）",
                block["fib_all_on_paired_classic_subset"]["metrics"],
            )


if __name__ == "__main__":
    main()
