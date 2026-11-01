"""
填空题评测脚本 (V6)

V6 填空题只接受结构化预测：
  id -> {condition_key: candidate_index}

只有当预测的候选编号组合命中真实存在的条件组合时，才参与主均分。
"""

import argparse
import json
import os
from typing import Dict, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset_index(dataset: list) -> Dict[str, dict]:
    index = {}
    for sample in dataset:
        sample_id = sample["id"]
        if sample_id in index:
            raise ValueError(f"评测集中存在重复 id: {sample_id}")
        index[sample_id] = sample
    return index


def _selection_key(selection: dict, query_keys: List[str]) -> str:
    return json.dumps({key: selection[key] for key in query_keys}, sort_keys=True, ensure_ascii=False)


def _validate_prediction(pred: dict, sample: dict) -> Tuple[bool, str]:
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


def evaluate(dataset: list, predictions: dict) -> Tuple[dict, list]:
    ds_index = build_dataset_index(dataset)
    pred_ids = set(predictions.keys())
    ds_ids = set(ds_index.keys())

    missing = ds_ids - pred_ids
    extra = pred_ids - ds_ids
    if missing:
        print(f"[WARN] 预测文件缺少 {len(missing)} 个样本: {sorted(missing)[:5]}...")
    if extra:
        print(f"[WARN] 预测文件多出 {len(extra)} 个未知 id: {sorted(extra)[:5]}...")

    eval_ids = sorted(ds_ids & pred_ids)
    n_total = len(eval_ids)
    num_scored = 0
    num_unhit = 0
    num_invalid = 0
    hit_count = 0
    exact_best_count = 0
    relative_score_sum = 0.0
    details = []

    for sample_id in eval_ids:
        sample = ds_index[sample_id]
        prediction = predictions[sample_id]
        query_keys = sample.get("query_keys", [])
        valid, reason = _validate_prediction(prediction, sample)

        lookup = {
            _selection_key(record["selection"], query_keys): record
            for record in sample["meta"].get("legal_combinations", [])
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
            relative_score_sum += relative_score
            if best_hit:
                exact_best_count += 1

        details.append({
            "id": sample_id,
            "prediction": prediction,
            "valid_prediction": valid,
            "status": "scored" if hit else ("invalid" if not valid else "unhit"),
            "reason": reason if not valid else ("hit" if hit else "combination_not_found"),
            "relative_score": relative_score,
            "best_hit": best_hit,
        })

    results = {
        "num_samples": n_total,
        "num_missing": len(missing),
        "num_scored": num_scored,
        "num_unhit": num_unhit,
        "num_invalid": num_invalid,
        "hit_rate": hit_count / n_total if n_total else 0.0,
        "avg_relative_score": relative_score_sum / num_scored if num_scored else 0.0,
        "best_hit_rate_among_scored": exact_best_count / num_scored if num_scored else 0.0,
        "best_hit_count": exact_best_count,
    }
    return results, details


def print_results(results: dict, details: list, show_detail: bool):
    print("\n=======================================================")
    print("  填空题评测结果 (V6)")
    print("=======================================================")
    print(f"  评测样本数:              {results['num_samples']}")
    if results["num_missing"] > 0:
        print(f"  缺失样本数:              {results['num_missing']}")
    print(f"  命中并计分样本数:        {results['num_scored']}")
    print(f"  未命中合法组合数:        {results['num_unhit']}")
    print(f"  非法预测数:              {results['num_invalid']}")
    print(f"  hit_rate:                {results['hit_rate']:.1%}")
    print(f"  命中样本平均得分:        {results['avg_relative_score']:.3f}")
    print(
        f"  命中样本最优率:          {results['best_hit_rate_among_scored']:.1%}"
        f"  ({results['best_hit_count']}/{results['num_scored']})"
        if results["num_scored"]
        else "  命中样本最优率:          0.0%"
    )
    print("=======================================================")

    if show_detail:
        print("\n  逐样本详情:")
        for item in details:
            score_text = "-" if item["relative_score"] is None else f"{item['relative_score']:.3f}"
            print(
                f"  {item['id']:50s}"
                f"  status={item['status']}"
                f"  reason={item['reason']}"
                f"  score={score_text}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description="填空题评测脚本 (V6)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="评测集路径 (默认同目录下 fill_in_blank_all_blank.json)",
    )
    parser.add_argument("--prediction", required=True, help="预测结果 JSON 路径")
    parser.add_argument("--detail", action="store_true", help="输出逐样本详情")
    args = parser.parse_args()

    ds_path = args.dataset or os.path.join(SCRIPT_DIR, "fill_in_blank_all_blank.json")
    dataset = load_json(ds_path)
    predictions = load_json(args.prediction)
    results, details = evaluate(dataset, predictions)
    print_results(results, details, args.detail)


if __name__ == "__main__":
    main()
