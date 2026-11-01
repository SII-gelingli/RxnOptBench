#!/usr/bin/env python3
"""Reproduce the validity and robustness analyses used in the XiMt rebuttal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


TIERS = {
    "chemistry": "Chemistry LLMs",
    "open_source": "Open Source LLMs",
    "proprietary": "Proprietary LLMs",
}
SELECTIVITY_KEYS = ("ee", "dr", "er", "rr")
FIB_CANDIDATE_SHUFFLE_SEED = 20260504
AUDIT_ROOT = Path(__file__).resolve().parent
DEFAULT_RELEASE_ROOT = AUDIT_ROOT.parents[1]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.percentile(np.asarray(list(values), dtype=float), q))


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "q1": percentile(array, 25),
        "median": percentile(array, 50),
        "mean": float(array.mean()),
        "q3": percentile(array, 75),
        "max": float(array.max()),
    }


def valid_mc_index(value: Any, option_count: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value < option_count:
        return value
    return None


def major_fraction(key: str, value: float) -> float:
    if key == "ee":
        return (100.0 + min(abs(value), 100.0)) / 200.0
    ratio = max(value, 0.0)
    if math.isinf(ratio):
        return 1.0
    return ratio / (ratio + 1.0) if ratio > 0 else 0.0


def objective_diagnostics(
    question: dict[str, Any],
) -> tuple[list[bool], list[str], list[float] | None, list[float] | None]:
    scores = question["meta"]["scores"]
    complete_keys = [
        key for key in SELECTIVITY_KEYS
        if all(score.get(key) is not None for score in scores)
    ]
    vectors: list[list[float]] = []
    for score in scores:
        vector = [float(score["yield"]) / 100.0]
        vector.extend(
            major_fraction(key, float(score[key])) for key in complete_keys
        )
        vectors.append(vector)

    mask = []
    for index, vector in enumerate(vectors):
        dominated = False
        for other_index, other in enumerate(vectors):
            if index == other_index:
                continue
            weakly_better = all(a >= b for a, b in zip(other, vector))
            strictly_better = any(a > b for a, b in zip(other, vector))
            if weakly_better and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    if not complete_keys:
        return mask, complete_keys, None, None

    additive = [float(np.mean(vector)) for vector in vectors]
    maximin = [min(vector) for vector in vectors]
    additive_best = max(additive)
    maximin_best = max(maximin)
    additive_relative = [
        value / additive_best if additive_best > 0 else 0.0 for value in additive
    ]
    maximin_relative = [
        value / maximin_best if maximin_best > 0 else 0.0 for value in maximin
    ]
    return mask, complete_keys, additive_relative, maximin_relative


def rank_quartiles(rows: list[dict[str, Any]], key: str) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: (rows[index][key], rows[index]["id"]))
    for quartile, indices in enumerate(np.array_split(ordered, 4), start=1):
        for index in indices:
            rows[index]["spread_quartile"] = f"Q{quartile}"


def analyze_outcome_spread(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        yields = [float(value) for value in question["meta"]["yields"]]
        effective = [float(value) for value in question["meta"]["effective_scores"]]
        selectivity_keys = [
            key for key in SELECTIVITY_KEYS
            if any(score.get(key) is not None for score in question["meta"]["scores"])
        ]
        mask, complete_keys, _, _ = objective_diagnostics(question)
        rows.append(
            {
                "id": question["id"],
                "doi": question["meta"]["doi"],
                "option_count": len(question["options"]),
                "yield_min": min(yields),
                "yield_max": max(yields),
                "yield_spread_pp": max(yields) - min(yields),
                "yield_relative_spread": (
                    (max(yields) - min(yields)) / max(yields) if max(yields) > 0 else 0.0
                ),
                "effective_min": min(effective),
                "effective_max": max(effective),
                "effective_spread": max(effective) - min(effective),
                "has_selectivity": bool(selectivity_keys),
                "selectivity_keys": ",".join(selectivity_keys),
                "complete_selectivity_keys": ",".join(complete_keys),
                "pareto_option_count": sum(mask),
            }
        )
    rank_quartiles(rows, "yield_spread_pp")
    return rows


def analyze_model_performance(
    questions: list[dict[str, Any]],
    spread_rows: list[dict[str, Any]],
    results_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    spread_by_id = {row["id"]: row for row in spread_rows}
    prediction_rows: list[dict[str, Any]] = []

    for tier_dir, tier_name in TIERS.items():
        for model_dir in sorted((results_root / tier_dir).iterdir()):
            predictions = load_json(model_dir / "mc_all.json")
            for question in questions:
                option_count = len(question["options"])
                predicted = valid_mc_index(predictions.get(question["id"]), option_count)
                complete_selectivity_keys = spread_by_id[question["id"]][
                    "complete_selectivity_keys"
                ]
                complete_selectivity = bool(complete_selectivity_keys)
                if predicted is None:
                    product_relative = 0.0
                    yield_relative = 0.0
                    exact = 0
                    pareto = 0
                    additive_relative = None
                    maximin_relative = None
                else:
                    yields = [float(value) for value in question["meta"]["yields"]]
                    product_relative = float(
                        question["meta"]["option_relative_scores"][predicted]
                    )
                    yield_relative = yields[predicted] / max(yields) if max(yields) > 0 else 0.0
                    exact = int(predicted in question["answer"])
                    (
                        mask,
                        _,
                        additive_relative_scores,
                        maximin_relative_scores,
                    ) = objective_diagnostics(question)
                    pareto = int(mask[predicted])
                    additive_relative = (
                        additive_relative_scores[predicted]
                        if additive_relative_scores is not None
                        else None
                    )
                    maximin_relative = (
                        maximin_relative_scores[predicted]
                        if maximin_relative_scores is not None
                        else None
                    )
                prediction_rows.append(
                    {
                        "tier": tier_name,
                        "model": model_dir.name,
                        "id": question["id"],
                        "predicted_index": predicted,
                        "valid_prediction": predicted is not None,
                        "spread_quartile": spread_by_id[question["id"]]["spread_quartile"],
                        "has_selectivity": spread_by_id[question["id"]]["has_selectivity"],
                        "complete_selectivity": complete_selectivity,
                        "complete_selectivity_keys": complete_selectivity_keys,
                        "product_relative_score": product_relative,
                        "yield_relative_score": yield_relative,
                        "exact": exact,
                        "pareto_optimal": pareto if complete_selectivity else None,
                        "additive_relative_score": additive_relative,
                        "maximin_relative_score": maximin_relative,
                    }
                )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(row["tier"], row["model"], row["spread_quartile"])].append(row)
    performance_rows = []
    for (tier, model, quartile), rows in sorted(grouped.items()):
        performance_rows.append(
            {
                "tier": tier,
                "model": model,
                "spread_quartile": quartile,
                "n": len(rows),
                "accuracy_pct": 100.0 * np.mean([row["exact"] for row in rows]),
                "product_score_pct": 100.0
                * np.mean([row["product_relative_score"] for row in rows]),
                "yield_only_score_pct": 100.0
                * np.mean([row["yield_relative_score"] for row in rows]),
            }
        )

    model_rows = []
    grouped_models: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped_models[(row["tier"], row["model"])].append(row)
    for (tier, model), rows in sorted(grouped_models.items()):
        selectivity_rows = [row for row in rows if row["has_selectivity"]]
        pareto_rows = [row for row in rows if row["complete_selectivity"]]
        model_rows.append(
            {
                "tier": tier,
                "model": model,
                "n": len(rows),
                "accuracy_pct": 100.0 * np.mean([row["exact"] for row in rows]),
                "product_score_pct": 100.0
                * np.mean([row["product_relative_score"] for row in rows]),
                "yield_only_score_pct": 100.0
                * np.mean([row["yield_relative_score"] for row in rows]),
                "selectivity_n": len(selectivity_rows),
                "selectivity_product_score_pct": 100.0
                * np.mean([row["product_relative_score"] for row in selectivity_rows]),
                "selectivity_yield_only_score_pct": 100.0
                * np.mean([row["yield_relative_score"] for row in selectivity_rows]),
                "pareto_analysis_n": len(pareto_rows),
                "pareto_optimal_pct": 100.0
                * np.mean([row["pareto_optimal"] for row in pareto_rows]),
                "complete_selectivity_product_score_pct": 100.0
                * np.mean([row["product_relative_score"] for row in pareto_rows]),
                "additive_score_pct": 100.0
                * np.mean([row["additive_relative_score"] for row in pareto_rows]),
                "maximin_score_pct": 100.0
                * np.mean([row["maximin_relative_score"] for row in pareto_rows]),
            }
        )
    return prediction_rows, performance_rows, model_rows


def position_rows(questions: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    counts: dict[tuple[int, int], int] = Counter()
    for question in questions:
        option_count = len(question["options"])
        for answer in question["answer"]:
            counts[(option_count, int(answer))] += 1
    rows = []
    for (option_count, position), count in sorted(counts.items()):
        total = sum(
            value
            for (arity, _), value in counts.items()
            if arity == option_count
        )
        rows.append(
            {
                "task": task,
                "option_count": option_count,
                "gold_position": position,
                "count": count,
                "conditional_pct": 100.0 * count / total,
            }
        )
    return rows


def virtual_fib_gold_positions(
    question: dict[str, Any],
) -> Iterable[tuple[str, int, int]]:
    """Yield FIB gold positions after the deterministic submission-time shuffle.

    The distributed ``data/original_json`` FIB file preserves the original
    candidate order.  The rebuttal audit used the common ``candidate_shuffle_v3``
    index space, so reproduce that permutation in memory without creating or
    depending on a second dataset copy.
    """

    sample_id = question["id"]
    gold_selection = question["answer"][0]
    for key, old_position in gold_selection.items():
        option_count = len(question["input"]["candidate_values"][key])
        payload = (
            f"{FIB_CANDIDATE_SHUFFLE_SEED}::{sample_id}::{key}"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        order = list(range(option_count))
        random.Random(int(digest[:16], 16)).shuffle(order)
        old_to_new = {
            old_index: new_index
            for new_index, old_index in enumerate(order)
        }
        yield key, option_count, old_to_new[int(old_position)]


def fib_position_rows(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, int], int] = Counter()
    for question in questions:
        for _, option_count, position in virtual_fib_gold_positions(question):
            counts[(option_count, position)] += 1
    rows = []
    for (option_count, position), count in sorted(counts.items()):
        total = sum(
            value
            for (arity, _), value in counts.items()
            if arity == option_count
        )
        rows.append(
            {
                "task": "fib_all",
                "option_count": option_count,
                "gold_position": position,
                "count": count,
                "conditional_pct": 100.0 * count / total,
            }
        )
    return rows


def first_position_audit(
    questions: list[dict[str, Any]], seed: int = 20260725, simulations: int = 100_000
) -> dict[str, float | int]:
    observed = int(sum(0 in question["answer"] for question in questions))
    answer_arity_counts = Counter(
        (len(question["answer"]), len(question["options"])) for question in questions
    )
    expected = float(
        sum(count * answer_count / option_count
            for (answer_count, option_count), count in answer_arity_counts.items())
    )
    rng = np.random.default_rng(seed)
    simulated = np.zeros(simulations, dtype=int)
    for (answer_count, option_count), count in answer_arity_counts.items():
        simulated += rng.binomial(
            count, answer_count / option_count, size=simulations
        )
    distance = abs(observed - expected)
    pvalue = float(
        (1 + np.sum(np.abs(simulated - expected) >= distance)) / (simulations + 1)
    )
    return {
        "n": len(questions),
        "observed_first_count": observed,
        "observed_first_pct": 100.0 * observed / len(questions),
        "conditional_uniform_expected_count": expected,
        "conditional_uniform_expected_pct": 100.0 * expected / len(questions),
        "monte_carlo_two_sided_pvalue": pvalue,
        "simulations": simulations,
    }


def fib_first_position_audit(
    questions: list[dict[str, Any]], seed: int = 20260727, simulations: int = 100_000
) -> dict[str, float | int]:
    arities = []
    observed = 0
    for question in questions:
        for _, option_count, position in virtual_fib_gold_positions(question):
            arities.append(option_count)
            observed += int(position == 0)
    arity_counts = Counter(arities)
    expected = float(sum(count / arity for arity, count in arity_counts.items()))
    rng = np.random.default_rng(seed)
    simulated = np.zeros(simulations, dtype=int)
    for arity, count in arity_counts.items():
        simulated += rng.binomial(count, 1.0 / arity, size=simulations)
    distance = abs(observed - expected)
    pvalue = float(
        (1 + np.sum(np.abs(simulated - expected) >= distance)) / (simulations + 1)
    )
    return {
        "n_key_occurrences": len(arities),
        "observed_first_count": observed,
        "observed_first_pct": 100.0 * observed / len(arities),
        "conditional_uniform_expected_count": expected,
        "conditional_uniform_expected_pct": 100.0 * expected / len(arities),
        "monte_carlo_two_sided_pvalue": pvalue,
        "simulations": simulations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDIT_ROOT / "outputs",
    )
    parser.add_argument("--data-root", type=Path, help="Override native JSON directory, e.g. a named historical snapshot.")
    parser.add_argument("--results-root", type=Path, help="Override parsed-output directory in the downloaded gated artifact.")
    args = parser.parse_args()

    release_root = args.release_root
    data_root = args.data_root or release_root / "data/original_json"
    results_root = args.results_root or release_root / "results/predictions"
    mc_all = load_json(data_root / "multiple_choice_all_varying.json")
    mc_single = load_json(data_root / "multiple_choice_single_varying.json")
    fib_all = load_json(data_root / "fill_in_blank_all_blank.json")

    spread_rows = analyze_outcome_spread(mc_all)
    prediction_rows, performance_rows, model_rows = analyze_model_performance(
        mc_all, spread_rows, results_root
    )

    product_scores = [row["product_score_pct"] for row in model_rows]
    yield_scores = [row["yield_only_score_pct"] for row in model_rows]
    rank_correlation = spearmanr(product_scores, yield_scores)
    selectivity_product = [
        row["selectivity_product_score_pct"] for row in model_rows
    ]
    selectivity_yield = [
        row["selectivity_yield_only_score_pct"] for row in model_rows
    ]
    selectivity_rank_correlation = spearmanr(selectivity_product, selectivity_yield)
    complete_product = [
        row["complete_selectivity_product_score_pct"] for row in model_rows
    ]
    additive_scores = [row["additive_score_pct"] for row in model_rows]
    maximin_scores = [row["maximin_score_pct"] for row in model_rows]
    additive_rank_correlation = spearmanr(complete_product, additive_scores)
    maximin_rank_correlation = spearmanr(complete_product, maximin_scores)
    overall_by_model = {
        row["model"]: row["product_score_pct"] for row in model_rows
    }
    spread_rank_correlations = {}
    tier_spread_scores = {}
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        quartile_rows = [
            row for row in performance_rows if row["spread_quartile"] == quartile
        ]
        by_model = {row["model"]: row["product_score_pct"] for row in quartile_rows}
        models = sorted(by_model)
        spread_rank_correlations[quartile] = float(
            spearmanr(
                [overall_by_model[model] for model in models],
                [by_model[model] for model in models],
            ).statistic
        )
        tier_spread_scores[quartile] = {
            tier: float(
                np.mean(
                    [
                        row["product_score_pct"]
                        for row in quartile_rows
                        if row["tier"] == tier
                    ]
                )
            )
            for tier in TIERS.values()
        }

    positions = (
        position_rows(mc_single, "mc_single")
        + position_rows(mc_all, "mc_all")
        + fib_position_rows(fib_all)
    )

    yield_spreads = [row["yield_spread_pp"] for row in spread_rows]
    effective_spreads = [row["effective_spread"] for row in spread_rows]
    summary = {
        "outcome_spread": {
            "yield_spread_pp": describe(yield_spreads),
            "effective_spread": describe(effective_spreads),
            "tables_at_least_10pp": sum(value >= 10 for value in yield_spreads),
            "tables_at_least_20pp": sum(value >= 20 for value in yield_spreads),
            "tables_at_least_50pp": sum(value >= 50 for value in yield_spreads),
            "zero_spread_tables": sum(value == 0 for value in yield_spreads),
        },
        "selectivity": {
            "tables_with_any_selectivity": sum(
                row["has_selectivity"] for row in spread_rows
            ),
            "product_and_yield_best_overlap": sum(
                int(bool(
                    set(question["answer"])
                    & {
                        index
                        for index, value in enumerate(question["meta"]["yields"])
                        if value == max(question["meta"]["yields"])
                    }
                ))
                for question in mc_all
            ),
            "total_tables": len(mc_all),
            "tables_with_complete_selectivity_axis": sum(
                bool(row["complete_selectivity_keys"]) for row in spread_rows
            ),
        },
        "score_robustness": {
            "spearman_product_vs_yield_all_models": float(rank_correlation.statistic),
            "spearman_pvalue": float(rank_correlation.pvalue),
            "spearman_product_vs_yield_selectivity_tables": float(
                selectivity_rank_correlation.statistic
            ),
            "selectivity_spearman_pvalue": float(
                selectivity_rank_correlation.pvalue
            ),
            "spearman_product_vs_additive_complete_tables": float(
                additive_rank_correlation.statistic
            ),
            "additive_spearman_pvalue": float(additive_rank_correlation.pvalue),
            "spearman_product_vs_maximin_complete_tables": float(
                maximin_rank_correlation.statistic
            ),
            "maximin_spearman_pvalue": float(maximin_rank_correlation.pvalue),
        },
        "outcome_spread_robustness": {
            "model_rank_spearman_vs_overall": spread_rank_correlations,
            "tier_product_score_pct": tier_spread_scores,
        },
        "answer_positions": {
            "mc_single": first_position_audit(mc_single),
            "mc_all": first_position_audit(mc_all, seed=20260726),
            "fib_all": fib_first_position_audit(fib_all),
            "fib_virtual_shuffle": {
                "scheme": "candidate_shuffle_v3",
                "seed": FIB_CANDIDATE_SHUFFLE_SEED,
                "key_derivation": "sha256(seed::sample_id::key)",
            },
        },
    }

    write_tsv(args.output_dir / "outcome_spread_tables.tsv", spread_rows)
    write_tsv(args.output_dir / "prediction_diagnostics.tsv", prediction_rows)
    write_tsv(args.output_dir / "performance_by_spread.tsv", performance_rows)
    write_tsv(args.output_dir / "model_score_robustness.tsv", model_rows)
    write_tsv(args.output_dir / "answer_positions.tsv", positions)
    write_json(args.output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
