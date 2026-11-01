"""
选择题评测脚本 (V6)

本脚本只接受 `id -> option_index` 的单选预测。
主指标是平均相对得分，预测值必须是单个 0-based 选项索引。

注意：
- `avg_relative_score` 基于 `meta.option_relative_scores`
- `avg_yield_ratio` 基于 `predicted_yield / meta.best_yield`
- 对于 `single_varying`，这里的 `best_yield` 是题目局部真实选项集内的最佳产率
"""

import argparse
import json
import os
from typing import Dict, Tuple


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
    exact_correct = 0
    valid_prediction_count = 0
    relative_score_sum = 0.0
    yield_ratio_sum = 0.0
    details = []

    for sample_id in eval_ids:
        sample = ds_index[sample_id]
        prediction = predictions[sample_id]
        yields_list = sample["meta"].get("yields", [])
        relative_scores = sample["meta"].get("option_relative_scores", [])
        best_yield = sample["meta"].get("best_yield", 0.0)
        answer_indices = sample["answer"]

        valid = isinstance(prediction, int) and 0 <= prediction < len(sample.get("options", []))
        pred_yield = yields_list[prediction] if valid and prediction < len(yields_list) else 0.0
        pred_relative_score = relative_scores[prediction] if valid and prediction < len(relative_scores) else 0.0
        exact = valid and prediction in answer_indices

        if valid:
            valid_prediction_count += 1
        if exact:
            exact_correct += 1
        relative_score_sum += pred_relative_score
        if best_yield > 0:
            yield_ratio_sum += pred_yield / best_yield
        else:
            yield_ratio_sum += 1.0 if pred_yield == 0 else 0.0

        details.append({
            "id": sample_id,
            "predicted": prediction,
            "valid_prediction": valid,
            "answer": answer_indices,
            "exact": exact,
            "relative_score": pred_relative_score,
            "pred_yield": pred_yield,
            "best_yield": best_yield,
        })

    results = {
        "num_samples": n_total,
        "num_missing": len(missing),
        "num_valid_predictions": valid_prediction_count,
        "exact_match_accuracy": exact_correct / n_total if n_total else 0.0,
        "avg_relative_score": relative_score_sum / n_total if n_total else 0.0,
        "avg_yield_ratio": yield_ratio_sum / n_total if n_total else 0.0,
        "exact_match_count": exact_correct,
    }
    return results, details


def print_results(results: dict, details: list, show_detail: bool):
    print("\n=======================================================")
    print("  选择题评测结果 (V6)")
    print("=======================================================")
    print(f"  评测样本数:              {results['num_samples']}")
    if results["num_missing"] > 0:
        print(f"  缺失样本数:              {results['num_missing']}")
    print(f"  合法预测数:              {results['num_valid_predictions']}")
    print(
        f"  精确命中率:              {results['exact_match_accuracy']:.1%}"
        f"  ({results['exact_match_count']}/{results['num_samples']})"
        if results["num_samples"]
        else "  精确命中率:              0.0%"
    )
    print(f"  平均相对得分:            {results['avg_relative_score']:.3f}")
    print(f"  平均产率比:              {results['avg_yield_ratio']:.1%}")
    print("=======================================================")

    if show_detail:
        print("\n  逐样本详情:")
        for item in details:
            mark = "O" if item["exact"] else "X"
            print(
                f"  {mark}  {item['id']:50s}"
                f"  pred={item['predicted']}"
                f"  ans={item['answer']}"
                f"  score={item['relative_score']:.3f}"
                f"  yield={item['pred_yield']:.0f}/{item['best_yield']:.0f}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description="选择题评测脚本 (V6)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="评测集路径 (默认同目录下 multiple_choice_single_varying.json)",
    )
    parser.add_argument("--prediction", required=True, help="预测结果 JSON 路径")
    parser.add_argument("--detail", action="store_true", help="输出逐样本详情")
    args = parser.parse_args()

    ds_path = args.dataset or os.path.join(SCRIPT_DIR, "multiple_choice_single_varying.json")
    dataset = load_json(ds_path)
    predictions = load_json(args.prediction)
    results, details = evaluate(dataset, predictions)
    print_results(results, details, args.detail)


if __name__ == "__main__":
    main()
