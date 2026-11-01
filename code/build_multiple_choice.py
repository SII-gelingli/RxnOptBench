"""
选择题评测集构建脚本 (V6)

本脚本负责：
1. 从原始 table 数据构建选择题数据集
2. 生成 prompt 模板、示例预测和 `report.md`
3. 为输出目录复制构建/评测脚本与共享 helper 快照

其中 `single_varying` 的核心规则是：
1. 在原表中枚举其它变化条件的真实固定组合
2. 选择对目标条件覆盖值最多的一组作为 `fixed_conditions`
3. 只把这组下真实出现过的目标条件取值作为候选选项
4. 单条件题的最优答案与相对得分只在该局部真实选项集内计算
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")
if EVALUATION_DIR not in sys.path:
    sys.path.insert(0, EVALUATION_DIR)

import common_v6 as core
from common_v6 import build_reference_records, compute_relative_scores, configure_runtime


DEFAULT_ARTICLE_DIR = os.path.join(ROOT_DIR, "inputs", "pipeline_output")
OUTPUT_DIR = os.environ.get(
    "RXNOPTBENCH_BUILD_OUTPUT_DIR",
    os.path.join(ROOT_DIR, "build_outputs"),
)
DATA_SOURCES: List[Tuple[str, str]] = [
    (
        os.environ.get("RXNOPTBENCH_ARTICLE_DIR", DEFAULT_ARTICLE_DIR),
        os.environ.get("RXNOPTBENCH_METADATA_DIR", ""),
    ),
]
SIOC_MAPPING_CSV = os.environ.get(
    "RXNOPTBENCH_SIOC_MAPPING_CSV",
    os.path.join(ROOT_DIR, "metadata", "metadata_new.csv"),
)

YEAR_FILTER: Union[None, int, List[int]] = None
CONDITION_KEYS = None
CONDITION_REQUIRE = "content"
SMILES_EXEMPT_KEYS = ["solvents"]
REACTANT_PRODUCT_REQUIRE = "smiles"
INPUT_PRODUCTS_MODE = "main_only"
YIELD_FIELD_PRIORITY = ["isolated_yield", "yield_ratio"]

EXPAND_CURRENT_CONDITIONS = True
CONDITION_KEYS_BLACKLIST = ["indeterminate_mol_idt", "description", "indeterminate_descriptions", "other_info", "selected_conditions", "scale"]
NORMALIZE_CONDITIONS = True

BALANCE_ALL_VARYING_OPTIONS = False
BALANCE_MAX_IMBALANCE_RATIO = 1.5
BALANCE_MIN_OPTIONS = 4

MIN_REFERENCE_OPTIONS = 2
MAX_REFERENCE_OPTIONS = 5
MIN_REFERENCE_REMAINING_EFFECTIVE_OPTIONS = 5
REFERENCE_LOW_MID_FRACTION = 2.0 / 3.0

QUESTION_TYPE_TO_FILE = {
    "single_varying": "multiple_choice_single_varying.json",
    "all_varying": "multiple_choice_all_varying.json",
    "single_varying_reference": "multiple_choice_single_varying_reference.json",
    "all_varying_reference": "multiple_choice_all_varying_reference.json",
    "single_varying_reference_control": "multiple_choice_single_varying_reference_control.json",
    "all_varying_reference_control": "multiple_choice_all_varying_reference_control.json",
}

QUESTION_TYPE_TO_PROMPT = {
    "single_varying": "prompt_single_varying.txt",
    "all_varying": "prompt_all_varying.txt",
    "single_varying_reference": "prompt_single_varying_reference.txt",
    "all_varying_reference": "prompt_all_varying_reference.txt",
    "single_varying_reference_control": "prompt_single_varying.txt",
    "all_varying_reference_control": "prompt_all_varying.txt",
}

SETTINGS_DESCRIPTIONS = {
    "YEAR_FILTER": "论文发表年份筛选。None=不限；整数 N=仅 ≥N 年；列表=仅指定年份",
    "CONDITION_KEYS": "提取的反应条件类别。None=自动发现所有条件键",
    "CONDITION_REQUIRE": "条件项最低数据要求",
    "SMILES_EXEMPT_KEYS": "CONDITION_REQUIRE=smiles 时豁免 SMILES 要求的类别",
    "REACTANT_PRODUCT_REQUIRE": "反应物/产物完整性（表级）",
    "INPUT_PRODUCTS_MODE": "输入产物范围",
    "YIELD_FIELD_PRIORITY": "产率字段解析优先级",
    "EXPAND_CURRENT_CONDITIONS": "是否展开 current_conditions 子字段",
    "CONDITION_KEYS_BLACKLIST": "条件键黑名单",
    "NORMALIZE_CONDITIONS": "条件文本比较时是否归一化",
    "BALANCE_ALL_VARYING_OPTIONS": "是否对 all_varying 选项做均衡化",
    "BALANCE_MAX_IMBALANCE_RATIO": "均衡化目标上界",
    "BALANCE_MIN_OPTIONS": "均衡化保留的最少选项数",
    "MIN_REFERENCE_OPTIONS": "reference 模式下至少提供几个 reference 条件",
    "MAX_REFERENCE_OPTIONS": "reference 模式下最多提供几个 reference 条件",
    "MIN_REFERENCE_REMAINING_EFFECTIVE_OPTIONS": "仅 all_varying reference 使用，去掉 reference 后至少保留多少个有效选项",
    "REFERENCE_LOW_MID_FRACTION": "reference 仅从低到中等表现区间抽取的比例",
}

SCORING_RULE_SUMMARY = (
    "主分数为 predicted_effective / option_set_best_effective。其中 effective = yield * "
    "enantio_major_fraction * dr_major_fraction * rr_major_fraction；对映选择性优先用 ee，"
    "ee 缺失时用 er；缺失 ee/er/dr/rr 时按 1.0 处理，不额外惩罚。"
)
SINGLE_VARYING_RULE_SUMMARY = (
    "single_varying 先在原表中枚举其它变化条件的真实固定组，选择对目标条件覆盖值最多的一组作为 fixed_conditions；"
    "只有该组里真实出现过的组合才会成为选项，最优答案和相对得分都只在这一局部真实选项集内计算。"
)
SINGLE_REFERENCE_RULE_SUMMARY = (
    "single_varying_reference 的 reference 可来自组选项外的真实组合，但不能与任一选项+fixed_conditions 组合重合，"
    "且 effective 不能高于该题选项集内的最佳 effective。"
)


def _sync_core_globals():
    configure_runtime(
        YEAR_FILTER=YEAR_FILTER,
        CONDITION_KEYS=CONDITION_KEYS,
        CONDITION_REQUIRE=CONDITION_REQUIRE,
        SMILES_EXEMPT_KEYS=SMILES_EXEMPT_KEYS,
        REACTANT_PRODUCT_REQUIRE=REACTANT_PRODUCT_REQUIRE,
        INPUT_PRODUCTS_MODE=INPUT_PRODUCTS_MODE,
        YIELD_FIELD_PRIORITY=YIELD_FIELD_PRIORITY,
        EXPAND_CURRENT_CONDITIONS=EXPAND_CURRENT_CONDITIONS,
        CONDITION_KEYS_BLACKLIST=CONDITION_KEYS_BLACKLIST,
        NORMALIZE_CONDITIONS=NORMALIZE_CONDITIONS,
        BALANCE_ALL_VARYING_OPTIONS=BALANCE_ALL_VARYING_OPTIONS,
        BALANCE_MAX_IMBALANCE_RATIO=BALANCE_MAX_IMBALANCE_RATIO,
        BALANCE_MIN_OPTIONS=BALANCE_MIN_OPTIONS,
    )


def get_enabled_filters():
    filters = [
        core.filter_is_table,
        core.filter_reactant_product_require,
        core.filter_same_reactants_products,
    ]
    if YEAR_FILTER is not None:
        filters.append(core.filter_year_range)
    return filters


def _shuffle_bundles(bundles: List[dict], seed_str: str) -> List[dict]:
    shuffled = list(bundles)
    random.Random(seed_str).shuffle(shuffled)
    return shuffled


def _summarize_option_space(option_bundles: List[dict]):
    option_scores = [bundle["score"] for bundle in option_bundles]
    option_yields = [bundle["yield"] for bundle in option_bundles]
    effective_scores = [core.compute_effective_product_score(score) for score in option_scores]
    best_effective_score = max(effective_scores) if effective_scores else 0.0
    best_entry_indices = [
        bundle["entry_info"].get("entry_idx", str(idx))
        for idx, (bundle, effective_score) in enumerate(zip(option_bundles, effective_scores))
        if math.isclose(effective_score, best_effective_score, rel_tol=0.0, abs_tol=1e-9)
    ]
    return {
        "best_yield": max(option_yields) if option_yields else 0.0,
        "best_effective_score": best_effective_score,
        "best_entry_indices": best_entry_indices,
        "relative_scores": compute_relative_scores(option_scores),
        "effective_scores": [round(value, 6) for value in effective_scores],
    }


def _build_meta(
    table_data,
    context,
    common,
    table_id,
    option_bundles,
):
    option_summary = _summarize_option_space(option_bundles)
    option_scores = [bundle["score"] for bundle in option_bundles]
    return {
        "source_file": context["filename"],
        "doi": context.get("doi", ""),
        "best_yield": option_summary["best_yield"],
        "best_effective_score": option_summary["best_effective_score"],
        "best_entry_indices": option_summary["best_entry_indices"],
        "global_best_yield": common["max_yield"],
        "global_best_effective_score": common["best_effective_score"],
        "global_best_entry_indices": common["best_entry_indices"],
        "total_entries_in_source_table": len(table_data),
        "yields": [bundle["yield"] for bundle in option_bundles],
        "effective_scores": option_summary["effective_scores"],
        "scores": [core.score_to_dict(score) for score in option_scores],
        "option_relative_scores": option_summary["relative_scores"],
        "option_source_entries": [bundle["entry_info"] for bundle in option_bundles],
        "parent_table_id": table_id,
        "main_product_smiles": common["main_product_smiles"],
    }


def _make_choice_question(
    table_data,
    context,
    common,
    table_id,
    qid: str,
    question_type: str,
    target_key: Optional[str],
    varying_keys: List[str],
    input_payload: dict,
    option_bundles: List[dict],
):
    if len(option_bundles) < 2:
        return None
    if not core.has_score_variation([b["score"] for b in option_bundles]):
        return None

    shuffled_bundles = _shuffle_bundles(option_bundles, qid)
    option_scores = [bundle["score"] for bundle in shuffled_bundles]
    answer = core.get_top_answer_indices(option_scores)
    return {
        "id": qid,
        "question_type": question_type,
        "target_key": target_key,
        "varying_keys": varying_keys,
        "input": input_payload,
        "options": [bundle["option"] for bundle in shuffled_bundles],
        "answer": answer,
        "meta": _build_meta(
            table_data,
            context,
            common,
            table_id,
            shuffled_bundles,
        ),
    }


def _record_reference_outcome(stats: dict, reference_result: dict):
    stats["reference_attempts"] += 1
    reason = reference_result["info"].get("reason", "")
    if reason == "ok":
        stats["reference_generated"] += 1
    else:
        stats["reference_skipped"][reason] = stats["reference_skipped"].get(reason, 0) + 1


def _build_single_mode_questions(
    common,
    table_data,
    context,
    varying_keys,
    constant_conditions,
    table_id,
    target_key,
    qid,
    stats,
):
    selection = core.build_single_option_bundles(common, table_data, varying_keys, target_key)
    bundles = selection["option_bundles"]
    if len(bundles) < 2:
        return []

    prepared = core.prepare_condition_view(
        bundles,
        common["condition_keys"],
        varying_keys,
        constant_conditions,
        fixed_conditions=selection["fixed_conditions"],
    )
    visible_varying_keys = prepared["visible_varying_keys"]
    if target_key not in visible_varying_keys:
        return []

    base_input = {
        "reactants": common["reactants_info"],
        "products": common["products_info"],
        "constant_conditions": prepared["constant_conditions"],
        "fixed_conditions": prepared["fixed_conditions"],
    }
    bundles = prepared["option_bundles"]

    questions = [
        _make_choice_question(
            table_data,
            context,
            common,
            table_id,
            qid,
            "single_varying",
            target_key,
            visible_varying_keys,
            dict(base_input),
            bundles,
        )
    ]

    reference_result = core.select_single_reference_bundles(
        selection["reference_candidates"],
        selection["local_best_score"],
        min_references=MIN_REFERENCE_OPTIONS,
        max_references=MAX_REFERENCE_OPTIONS,
        low_mid_fraction=REFERENCE_LOW_MID_FRACTION,
    )
    _record_reference_outcome(stats, reference_result)
    if reference_result["info"]["reason"] == "ok":
        ref_input = dict(base_input)
        visible_reference_bundles = core.prune_option_bundles(reference_result["references"], prepared["visible_keys"])
        ref_input["reference_conditions"] = build_reference_records(visible_reference_bundles)
        questions.append(
            _make_choice_question(
                table_data,
                context,
                common,
                table_id,
                f"{qid}_ref",
                "single_varying_reference",
                target_key,
                visible_varying_keys,
                ref_input,
                bundles,
            )
        )
        questions.append(
            _make_choice_question(
                table_data,
                context,
                common,
                table_id,
                f"{qid}_ref_control",
                "single_varying_reference_control",
                target_key,
                visible_varying_keys,
                dict(base_input),
                bundles,
            )
        )
    return [question for question in questions if question]


def _build_all_mode_questions(common, table_data, context, varying_keys, constant_conditions, table_id, qid, stats):
    bundles, balance_info = core.build_all_option_bundles(common, table_data, varying_keys)
    if len(bundles) < 2:
        return []

    prepared = core.prepare_condition_view(
        bundles,
        common["condition_keys"],
        varying_keys,
        constant_conditions,
    )
    visible_varying_keys = prepared["visible_varying_keys"]
    if not visible_varying_keys:
        return []

    base_input = {
        "reactants": common["reactants_info"],
        "products": common["products_info"],
        "constant_conditions": prepared["constant_conditions"],
    }
    bundles = prepared["option_bundles"]

    if core.count_effectively_varying_keys(bundles, visible_varying_keys) <= 1:
        stats.setdefault("all_varying_single_key_skipped", 0)
        stats["all_varying_single_key_skipped"] += 1
        return []

    questions = [
        _make_choice_question(
            table_data,
            context,
            common,
            table_id,
            qid,
            "all_varying",
            None,
            visible_varying_keys,
            dict(base_input),
            bundles,
        )
    ]

    reference_result = core.select_reference_bundles(
        bundles,
        min_references=MIN_REFERENCE_OPTIONS,
        max_references=MAX_REFERENCE_OPTIONS,
        min_remaining_effective_options=MIN_REFERENCE_REMAINING_EFFECTIVE_OPTIONS,
        low_mid_fraction=REFERENCE_LOW_MID_FRACTION,
    )
    _record_reference_outcome(stats, reference_result)
    if reference_result["info"]["reason"] == "ok":
        ref_input = dict(base_input)
        ref_input["reference_conditions"] = build_reference_records(reference_result["references"])
        questions.append(
            _make_choice_question(
                table_data,
                context,
                common,
                table_id,
                f"{qid}_ref",
                "all_varying_reference",
                None,
                visible_varying_keys,
                ref_input,
                reference_result["remaining"],
            )
        )
        questions.append(
            _make_choice_question(
                table_data,
                context,
                common,
                table_id,
                f"{qid}_ref_control",
                "all_varying_reference_control",
                None,
                visible_varying_keys,
                dict(base_input),
                reference_result["remaining"],
            )
        )
    return [question for question in questions if question]


def build_questions(common, table_data, context, stats):
    all_conditions = common["all_conditions"]
    all_keys = common["condition_keys"]
    varying_keys = core.detect_varying_conditions(all_conditions, all_keys)
    if not varying_keys:
        return []

    best_idx = common["best_indices"][0]
    best_condition = all_conditions[best_idx]
    constant_keys = [key for key in all_keys if key not in varying_keys]
    constant_conditions = {key: best_condition[key] for key in constant_keys}
    table_id = table_data[0].get("meta_info", {}).get("root_img_name", "") or os.path.splitext(context["filename"])[0]

    questions = []
    q_num = 1
    if len(varying_keys) >= 2:
        for target_key in varying_keys:
            qid = f"{table_id}_q{q_num}"
            questions.extend(
                _build_single_mode_questions(
                    common,
                    table_data,
                    context,
                    varying_keys,
                    constant_conditions,
                    table_id,
                    target_key,
                    qid,
                    stats,
                )
            )
            q_num += 1

    qid = f"{table_id}_q{q_num}"
    questions.extend(
        _build_all_mode_questions(
            common,
            table_data,
            context,
            varying_keys,
            constant_conditions,
            table_id,
            qid,
            stats,
        )
    )
    return questions


def _format_reference_block(reference_items: List[dict]) -> List[str]:
    lines: List[str] = []
    for idx, ref in enumerate(reference_items, start=1):
        outcome = ref["outcome"]
        score_parts = [f"yield={outcome['yield']}%"]
        if outcome.get("ee") is not None:
            score_parts.append(f"ee={outcome['ee']}")
        if outcome.get("dr") is not None:
            score_parts.append(f"dr={outcome['dr']}")
        if outcome.get("er") is not None:
            score_parts.append(f"er={outcome['er']}")
        if outcome.get("rr") is not None:
            score_parts.append(f"rr={outcome['rr']}")
        lines.append(f"*Reference {idx} ({', '.join(score_parts)}):*")
        lines.append(core._fmt_condition(ref["conditions"]))
        lines.append("")
    return lines


def generate_report(output_dir, dataset_by_type, filter_stats, stats):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_questions = sum(len(items) for items in dataset_by_type.values())
    example_prediction_files = {
        question_type: f"example_prediction_{os.path.splitext(filename)[0]}.json"
        for question_type, filename in QUESTION_TYPE_TO_FILE.items()
    }

    lines: List[str] = []
    lines.append("# 选择题评测集构建报告 (V6)\n")
    lines.append(f"> 生成时间: {ts}")
    lines.append(f"> 输出目录: `{os.path.basename(output_dir)}`")
    lines.append(f"> 题目总数: `{total_questions}`\n")

    lines.append("## 1. 先看什么\n")
    lines.append("如果你是第一次接手这个输出目录，推荐按下面顺序阅读：")
    lines.append("1. 先读 `report.md` 当前节与后面的“文件导航 / 最小接入路径 / JSON 关键字段”。")
    lines.append("2. 再打开一个目标数据集 JSON，例如 `multiple_choice_single_varying.json` 看真实样本。")
    lines.append("3. 根据样本的 `question_type` 选择对应 prompt 文件。")
    lines.append("4. 参考 `example_prediction_*.json` 确认模型输出格式。")
    lines.append("5. 最后用 `evaluate_multiple_choice.py` 跑评测；如果要改构题或评分，再看脚本快照与 `common_v6.py`。")
    lines.append("")

    lines.append("## 2. 文件导航\n")
    lines.append("| 你现在要做什么 | 优先读取哪些文件 | 为什么 |")
    lines.append("|------|------|------|")
    lines.append("| 快速理解这个目录 | `report.md` | 总览题型、字段、评测口径与命令 |")
    lines.append("| 看真实输入样本 | `multiple_choice_*.json` | 真实 schema、`meta` 字段和题型差异都以这里为准 |")
    lines.append("| 渲染题目给模型 | `prompt_*.txt` + 对应数据集 JSON | prompt 模板与样本字段需要一起看 |")
    lines.append("| 对接模型输出格式 | `example_prediction_*.json` | 最快确认 `id -> option_index` 契约 |")
    lines.append("| 运行评测 | `evaluate_multiple_choice.py` | 指标口径、合法预测规则与默认参数都在这里 |")
    lines.append("| 修改选择题文案或 schema | `build_multiple_choice.py` | report、prompt、样本结构都由这里导出 |")
    lines.append("| 修改条件解析、评分或 reference 抽样 | `common_v6.py` | 选择题与填空题共用 helper 在这里 |")
    lines.append("")

    lines.append("## 3. 最小接入路径\n")
    lines.append(
        "把本目录当成自包含评测包使用即可：选择一个数据集 JSON，读取对应 prompt 模板渲染题目，"
        "生成 `id -> option_index` 的预测文件，再用 `evaluate_multiple_choice.py` 评分。"
    )
    lines.append("")
    lines.append("### 3.1 最短工作流")
    lines.append("1. 选一个数据集文件。")
    lines.append("2. 根据 `question_type` 读取对应 prompt 模板。")
    lines.append("3. 用样本里的 `input` 字段渲染 prompt。")
    lines.append("4. 让模型只输出一个 0-based 选项编号。")
    lines.append("5. 把所有结果保存成 `prediction.json`。")
    lines.append("6. 运行 `evaluate_multiple_choice.py --dataset ... --prediction prediction.json`。")
    lines.append("")

    lines.append("## 4. 题型地图\n")
    lines.append("| question_type | 你会看到什么 | 评测时要注意什么 |")
    lines.append("|------|------|------|")
    lines.append("| `single_varying` | 只让 1 个条件键变化；题面会给一个真实 `fixed_conditions` 组 | 所有选项都来自这组固定条件下真实出现过的组合；得分只在该局部选项集内计算 |")
    lines.append("| `all_varying` | 一个选项代表一整组变化条件 | 评分直接在当前候选选项集内进行 |")
    lines.append("| `single_varying_reference` | 单条件题，并额外给 reference 条件组 | reference 来自题内选项之外的真实组合，不能高于该题局部最佳值 |")
    lines.append("| `all_varying_reference` | 全条件题，并额外给 reference 条件组 | 去掉 reference 后的剩余选项才是可选答案空间 |")
    lines.append("| `*_reference_control` | 与 reference 题配对的无 reference 对照题 | 用于对比 reference 信息是否有帮助，评分口径与对应非 control 题一致 |")
    lines.append("")
    lines.append("### 4.1 单条件题的真实组合规则")
    lines.append(f"- {SINGLE_VARYING_RULE_SUMMARY}")
    lines.append(f"- {SINGLE_REFERENCE_RULE_SUMMARY}")
    lines.append("")

    lines.append("## 5. 评分与评测\n")
    lines.append(f"- {SCORING_RULE_SUMMARY}")
    lines.append("- 主指标是 `avg_relative_score`，会对所有评测样本取均值。")
    lines.append("- `exact_match_accuracy` 用来看模型是否直接命中当前题目选项空间内的最优组。")
    lines.append("- `avg_yield_ratio` 用来看预测项产率相对当前题目最佳产率的接近程度。")
    lines.append("")

    lines.append("## 6. 输出文件\n")
    lines.append("### 6.1 数据集")
    lines.append("| 文件 | question_type | 样本数 |")
    lines.append("|------|------|------|")
    for question_type, filename in QUESTION_TYPE_TO_FILE.items():
        lines.append(f"| `{filename}` | `{question_type}` | {len(dataset_by_type[question_type])} |")
    lines.append("")
    lines.append("### 6.2 Prompt")
    lines.append("| 文件 | 对应题型 | 用途 |")
    lines.append("|------|------|------|")
    for question_type, filename in QUESTION_TYPE_TO_PROMPT.items():
        lines.append(f"| `{filename}` | `{question_type}` | 渲染给模型的指令模板 |")
    lines.append("")
    lines.append("### 6.3 示例预测")
    lines.append("| 文件 | 对应数据集 | 用途 |")
    lines.append("|------|------|------|")
    for question_type, filename in QUESTION_TYPE_TO_FILE.items():
        lines.append(
            f"| `{example_prediction_files[question_type]}` | `{filename}` | 用于快速验证预测 JSON 结构与评测链路 |"
        )
    lines.append("")
    lines.append("### 6.4 辅助文件")
    lines.append("| 文件 | 作用 | 什么时候读 |")
    lines.append("|------|------|------|")
    lines.append("| `report.md` | 当前目录的总览与接入指南 | 第一次接手输出目录时先读 |")
    lines.append("| `README.md` | benchmark 数据根目录说明快照 | 想快速了解整个数据目录而不仅是本批输出时读 |")
    lines.append("| `build_multiple_choice.py` | 选择题构建脚本快照 | 需要理解题目是怎么生成时读 |")
    lines.append("| `evaluate_multiple_choice.py` | 选择题评测脚本 | 需要跑分、看指标口径时读 |")
    lines.append("| `common_v6.py` | 共享解析与评分 helper | 需要修改公共逻辑或核对评分时读 |")
    lines.append("")

    lines.append("## 7. JSON 关键字段\n")
    lines.append("### 7.1 顶层字段")
    lines.append("| 字段 | 说明 |")
    lines.append("|------|------|")
    lines.append("| `id` | 题目唯一标识，格式为 `{table_id}_q{N}[_ref][_ref_control]` |")
    lines.append("| `question_type` | 题型标识，决定 prompt 模板和评分逻辑 |")
    lines.append("| `target_key` | 被提问的单个条件键（仅 `single_varying` 系列），`all_varying` 系列为 `null` |")
    lines.append("| `varying_keys` | 题目中所有变化条件键列表 |")
    lines.append("| `input` | 渲染 prompt 所需的全部信息（反应物、产物、固定条件、reference 等） |")
    lines.append("| `options` | 选项列表，每个元素是一组条件字典 |")
    lines.append("| `answer` | 当前题目选项空间内的最优索引列表；并列最优时长度可大于 1，仅用于 exact-hit 诊断 |")
    lines.append("| `meta` | 题目构建元信息，详见下方 |")
    lines.append("")
    lines.append("### 7.2 `input` 子字段")
    lines.append("| 字段 | 说明 |")
    lines.append("|------|------|")
    lines.append("| `input.reactants` | 反应物列表，每项含 `content`（名称）和 `SMILES` |")
    lines.append("| `input.products` | 产物列表，同上 |")
    lines.append("| `input.constant_conditions` | 所有选项共享的不变条件 |")
    lines.append("| `input.fixed_conditions` | 仅单条件题出现；表示题面固定住的真实条件组 |")
    lines.append("| `input.reference_conditions` | 仅 `*_reference` 题出现；提供额外线索的参考条件组与结果 |")
    lines.append("")
    lines.append("### 7.3 `meta` 子字段")
    lines.append("| 字段 | 说明 |")
    lines.append("|------|------|")
    lines.append("| `meta.source_file` | 该题来源的原始 JSON 文件名 |")
    lines.append("| `meta.doi` | 来源论文的 DOI |")
    lines.append("| `meta.best_yield` | 当前题目选项空间内的最佳产率 |")
    lines.append("| `meta.best_effective_score` | 当前题目选项空间内的最佳 effective score（综合 yield、ee、dr 等） |")
    lines.append("| `meta.best_entry_indices` | 当前选项空间内最优条目在原表中的行索引 |")
    lines.append("| `meta.global_best_yield` | 整张源表的全局最佳产率（可能高于当前选项集，仅供诊断） |")
    lines.append("| `meta.global_best_effective_score` | 整张源表的全局最佳 effective score |")
    lines.append("| `meta.global_best_entry_indices` | 整张源表全局最优条目的行索引 |")
    lines.append("| `meta.total_entries_in_source_table` | 源表（原始 JSON 文件）中的总条目数，不是当前选项数 |")
    lines.append("| `meta.yields` | 与 `options` 对齐的各选项产率列表 |")
    lines.append("| `meta.effective_scores` | 与 `options` 对齐的各选项 effective score 列表 |")
    lines.append("| `meta.scores` | 与 `options` 对齐的各选项详细评分（yield/ee/dr/er/rr） |")
    lines.append("| `meta.option_relative_scores` | 与 `options` 对齐的相对分数列表（最优=1.0），是主评测依据 |")
    lines.append("| `meta.option_source_entries` | 与 `options` 对齐的条目来源信息（row_index、entry_idx、rxn_id） |")
    lines.append("| `meta.parent_table_id` | 源表标识（通常为 root_img_name 或文件名去扩展名） |")
    lines.append("| `meta.main_product_smiles` | 用于评分的主产物 SMILES |")
    lines.append("")

    lines.append("## 8. 评测脚本\n")
    lines.append("预测文件必须是 `id -> 单个选项索引`：")
    lines.append("```json")
    lines.append("{")
    lines.append('  "sample_id_1": 2,')
    lines.append('  "sample_id_2": 0')
    lines.append("}")
    lines.append("```")
    lines.append("```bash")
    lines.append("python evaluate_multiple_choice.py \\")
    lines.append("  --dataset multiple_choice_all_varying_reference.json \\")
    lines.append("  --prediction pred.json")
    lines.append("```")
    lines.append("")

    lines.append("## 9. Reference 规则\n")
    lines.append("- reference 条件数至少 `2` 条、至多 `5` 条。")
    lines.append("- 参考条件只从低到中等表现区间抽取，不暴露最高分条件。")
    lines.append("- 去掉 reference 后，剩余候选必须至少还有 `5` 个有效选项。")
    lines.append("- 仅当 reference 成功生成时，额外配对生成无 reference 的 `*_reference_control` 题。")
    lines.append("")

    lines.append("## 10. 构建设置\n")
    lines.append("| 设置项 | 当前值 | 说明 |")
    lines.append("|------|------|------|")
    for name, value in [
        ("YEAR_FILTER", YEAR_FILTER),
        ("CONDITION_KEYS", CONDITION_KEYS),
        ("CONDITION_REQUIRE", CONDITION_REQUIRE),
        ("SMILES_EXEMPT_KEYS", SMILES_EXEMPT_KEYS),
        ("REACTANT_PRODUCT_REQUIRE", REACTANT_PRODUCT_REQUIRE),
        ("INPUT_PRODUCTS_MODE", INPUT_PRODUCTS_MODE),
        ("YIELD_FIELD_PRIORITY", YIELD_FIELD_PRIORITY),
        ("EXPAND_CURRENT_CONDITIONS", EXPAND_CURRENT_CONDITIONS),
        ("CONDITION_KEYS_BLACKLIST", CONDITION_KEYS_BLACKLIST),
        ("NORMALIZE_CONDITIONS", NORMALIZE_CONDITIONS),
        ("BALANCE_ALL_VARYING_OPTIONS", BALANCE_ALL_VARYING_OPTIONS),
        ("BALANCE_MAX_IMBALANCE_RATIO", BALANCE_MAX_IMBALANCE_RATIO),
        ("BALANCE_MIN_OPTIONS", BALANCE_MIN_OPTIONS),
        ("MIN_REFERENCE_OPTIONS", MIN_REFERENCE_OPTIONS),
        ("MAX_REFERENCE_OPTIONS", MAX_REFERENCE_OPTIONS),
        ("MIN_REFERENCE_REMAINING_EFFECTIVE_OPTIONS", MIN_REFERENCE_REMAINING_EFFECTIVE_OPTIONS),
        ("REFERENCE_LOW_MID_FRACTION", REFERENCE_LOW_MID_FRACTION),
    ]:
        lines.append(f"| `{name}` | `{value}` | {SETTINGS_DESCRIPTIONS.get(name, '')} |")
    lines.append("")

    lines.append("## 11. 统计摘要\n")
    lines.append(f"- table 文件总数: `{stats['total_tables']}`")
    lines.append(f"- 使用的表格数: `{stats['tables_used']}`")
    lines.append(f"- 生成题目总数: `{stats['questions_generated']}`")
    lines.append(f"- reference 尝试数: `{stats['reference_attempts']}`")
    lines.append(f"- reference 生成数: `{stats['reference_generated']}`")
    if stats["reference_skipped"]:
        lines.append(f"- reference 跳过原因: `{stats['reference_skipped']}`")
    for filt in get_enabled_filters():
        lines.append(f"- 被 `{filt.__name__}` 淘汰: `{filter_stats[filt.__name__]}`")
    lines.append("")

    lines.append("## 12. 样本展示")
    for question_type in [
        "single_varying",
        "single_varying_reference",
        "single_varying_reference_control",
        "all_varying_reference",
        "all_varying_reference_control",
    ]:
        if not dataset_by_type[question_type]:
            continue
        sample = dataset_by_type[question_type][0]
        lines.append(f"### `{question_type}` 示例：`{sample['id']}`\n")
        inp = sample["input"]
        lines.append(core._fmt_species(inp["reactants"], "反应物"))
        lines.append(core._fmt_species(inp["products"], "产物"))
        lines.append("")
        constant_conditions = inp.get("constant_conditions", {})
        if any(value for value in constant_conditions.values()):
            lines.append("**不变条件：**")
            lines.append(core._fmt_condition(constant_conditions))
            lines.append("")
        fixed_conditions = inp.get("fixed_conditions", {})
        if fixed_conditions:
            lines.append("**固定条件：**")
            lines.append(core._fmt_condition(fixed_conditions))
            lines.append("")
        if inp.get("reference_conditions"):
            lines.append("**Reference 条件：**")
            lines.extend(_format_reference_block(inp["reference_conditions"]))
        lines.append("**选项：**")
        for idx, option in enumerate(sample["options"]):
            lines.append(f"*Option {idx}:*")
            lines.append(core._fmt_condition(option))
            lines.append("")
        lines.append(f"**最佳索引:** `{sample['answer']}`")
        lines.append(f"**相对分数:** `{sample['meta']['option_relative_scores']}`")
        lines.append("")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


PROMPT_SINGLE_VARYING = r"""You are an expert organic chemist. Given the following chemical reaction and one real fixed condition group taken from the source table, choose the best value for **{target_key}** from the provided options.

### Reaction

**Reactants:**
{reactants}

**Target Product:**
{products}

### Constant Conditions (do not change)

{constant_conditions}

### Fixed Conditions

{fixed_conditions}

### Candidate Values For {target_key}

{options}

### Output Format

Respond with ONLY the option index (0-based integer).
"""

PROMPT_ALL_VARYING = r"""You are an expert organic chemist. Given the following chemical reaction, choose the best reaction-condition option from the provided candidates.

### Reaction

**Reactants:**
{reactants}

**Target Product:**
{products}

### Constant Conditions (do not change)

{constant_conditions}

### Candidate Conditions

{options}

### Output Format

Respond with ONLY the option index (0-based integer).
"""

PROMPT_SINGLE_VARYING_REFERENCE = r"""You are an expert organic chemist. Given the following reaction, one real fixed condition group from the source table, and several reference condition sets with observed outcomes, choose the best value for **{target_key}** from the remaining candidate options.

### Reaction

**Reactants:**
{reactants}

**Target Product:**
{products}

### Constant Conditions (do not change)

{constant_conditions}

### Fixed Conditions

{fixed_conditions}

### Reference Conditions And Outcomes

{references}

### Candidate Values For {target_key}

{options}

### Output Format

Respond with ONLY the option index (0-based integer).
"""

PROMPT_ALL_VARYING_REFERENCE = r"""You are an expert organic chemist. Given the following reaction and several reference condition sets with observed outcomes, choose the best remaining reaction-condition option.

### Reaction

**Reactants:**
{reactants}

**Target Product:**
{products}

### Constant Conditions (do not change)

{constant_conditions}

### Reference Conditions And Outcomes

{references}

### Candidate Conditions

{options}

### Output Format

Respond with ONLY the option index (0-based integer).
"""


def generate_prompt_templates(output_dir):
    templates = {
        "prompt_single_varying.txt": PROMPT_SINGLE_VARYING,
        "prompt_all_varying.txt": PROMPT_ALL_VARYING,
        "prompt_single_varying_reference.txt": PROMPT_SINGLE_VARYING_REFERENCE,
        "prompt_all_varying_reference.txt": PROMPT_ALL_VARYING_REFERENCE,
    }
    for filename, template in templates.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(template.strip() + "\n")


def generate_example_prediction(dataset, output_dir, filename="example_prediction.json"):
    rng = random.Random(42)
    predictions = {}
    for sample in dataset:
        n_options = len(sample["options"])
        if rng.random() < 0.65:
            predictions[sample["id"]] = sample["answer"][0]
        else:
            wrong = [idx for idx in range(n_options) if idx not in sample["answer"]]
            predictions[sample["id"]] = rng.choice(wrong) if wrong else 0
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    return out_path


def process_source(article_dir, metadata_dir, build_kwargs, dataset_out, filter_stats, stats, filename_doi_map=None):
    article_dir = os.path.normpath(article_dir)
    metadata_dir = os.path.normpath(metadata_dir) if metadata_dir else ""
    if filename_doi_map is None:
        filename_doi_map = {}

    table_files = sorted(
        filename for filename in os.listdir(article_dir)
        if "_table_" in filename and filename.endswith(".json")
    )
    stats["total_tables"] += len(table_files)
    print(f"  [INFO] {article_dir}")
    print(f"         找到 {len(table_files)} 个 table 文件")

    seen_ids = {
        sample["meta"]["parent_table_id"]
        for questions in dataset_out.values()
        for sample in questions
        if "parent_table_id" in sample.get("meta", {})
    }

    for filename in table_files:
        filepath = os.path.join(article_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            table_data = json.load(f)

        core.canonicalize_all_smiles(table_data)
        context = core.build_context(filename, filepath, metadata_dir, filename_doi_map)

        passed = True
        for filt in get_enabled_filters():
            if not filt(table_data, context):
                filter_stats[filt.__name__] += 1
                passed = False
                break
        if not passed:
            continue

        common = core._build_common_fields(table_data, context, **build_kwargs)
        if all(core._is_empty_condition(cond) for cond in common["best_conditions"]):
            stats["empty_skipped"] += 1
            continue

        sample_id = table_data[0].get("meta_info", {}).get("root_img_name", "") or os.path.splitext(filename)[0]
        if sample_id in seen_ids:
            stats["dup_skipped"] += 1
            continue

        questions = build_questions(common, table_data, context, stats)
        classic_questions = [
            question for question in questions
            if question["question_type"] in {"single_varying", "all_varying"}
        ]
        if not classic_questions:
            stats["no_varying_skipped"] += 1
            continue

        seen_ids.add(sample_id)
        for question in questions:
            dataset_out[question["question_type"]].append(question)
        stats["tables_used"] += 1
        stats["questions_generated"] += len(questions)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the submitted V6 multiple-choice benchmark from table JSON files."
    )
    parser.add_argument(
        "--article-dir",
        default=DATA_SOURCES[0][0],
        help="Directory containing source '*_table_*.json' files "
        "(env: RXNOPTBENCH_ARTICLE_DIR).",
    )
    parser.add_argument(
        "--metadata-dir",
        default=DATA_SOURCES[0][1],
        help="Optional companion metadata directory "
        "(env: RXNOPTBENCH_METADATA_DIR).",
    )
    parser.add_argument(
        "--sioc-mapping-csv",
        default=SIOC_MAPPING_CSV,
        help="Optional filename-to-DOI mapping CSV "
        "(env: RXNOPTBENCH_SIOC_MAPPING_CSV).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Parent directory for the timestamped build "
        "(env: RXNOPTBENCH_BUILD_OUTPUT_DIR).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global DATA_SOURCES, OUTPUT_DIR, SIOC_MAPPING_CSV

    args = parse_args(argv)
    article_dir = os.path.abspath(os.path.expanduser(args.article_dir))
    metadata_dir = (
        os.path.abspath(os.path.expanduser(args.metadata_dir))
        if args.metadata_dir
        else ""
    )
    if not os.path.isdir(article_dir):
        raise FileNotFoundError(
            f"Article directory not found: {article_dir}. "
            "Pass --article-dir or set RXNOPTBENCH_ARTICLE_DIR."
        )
    if metadata_dir and not os.path.isdir(metadata_dir):
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    DATA_SOURCES = [(article_dir, metadata_dir)]
    OUTPUT_DIR = os.path.abspath(os.path.expanduser(args.output_dir))
    SIOC_MAPPING_CSV = (
        os.path.abspath(os.path.expanduser(args.sioc_mapping_csv))
        if args.sioc_mapping_csv
        else ""
    )

    _sync_core_globals()

    base_dir = os.path.normpath(OUTPUT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{timestamp}_multiple_choice")
    os.makedirs(output_dir, exist_ok=True)

    build_kwargs = {
        "condition_keys": CONDITION_KEYS,
        "yield_fields": YIELD_FIELD_PRIORITY,
        "condition_require": CONDITION_REQUIRE,
        "smiles_exempt_keys": SMILES_EXEMPT_KEYS,
        "input_products_mode": INPUT_PRODUCTS_MODE,
    }

    filter_stats = {filt.__name__: 0 for filt in get_enabled_filters()}
    stats = {
        "total_tables": 0,
        "empty_skipped": 0,
        "dup_skipped": 0,
        "no_varying_skipped": 0,
        "tables_used": 0,
        "questions_generated": 0,
        "reference_attempts": 0,
        "reference_generated": 0,
        "reference_skipped": {},
    }
    dataset_by_type = {question_type: [] for question_type in QUESTION_TYPE_TO_FILE}

    filename_doi_map = {}
    if os.path.isfile(SIOC_MAPPING_CSV):
        filename_doi_map = core.load_filename_doi_mapping(SIOC_MAPPING_CSV)
        print(f"[选择题 V6] 已加载 SIOC 文件名→DOI 映射表: {len(filename_doi_map)} 条")

    print(f"[选择题 V6] 共 {len(DATA_SOURCES)} 组数据源")
    for idx, (article_dir, metadata_dir) in enumerate(DATA_SOURCES, start=1):
        print(f"\n  --- 数据源 {idx}/{len(DATA_SOURCES)} ---")
        process_source(
            article_dir,
            metadata_dir,
            build_kwargs,
            dataset_by_type,
            filter_stats,
            stats,
            filename_doi_map,
        )

    for question_type, filename in QUESTION_TYPE_TO_FILE.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(dataset_by_type[question_type], f, ensure_ascii=False, indent=2)

    generate_report(output_dir, dataset_by_type, filter_stats, stats)
    generate_prompt_templates(output_dir)
    for question_type, dataset in dataset_by_type.items():
        stem = os.path.splitext(QUESTION_TYPE_TO_FILE[question_type])[0]
        generate_example_prediction(dataset, output_dir, f"example_prediction_{stem}.json")

    readme_path = os.path.join(SCRIPT_DIR, "README.md")
    if os.path.isfile(readme_path):
        shutil.copy2(readme_path, output_dir)
    shutil.copy2(os.path.abspath(__file__), output_dir)
    shutil.copy2(os.path.join(EVALUATION_DIR, "common_v6.py"), output_dir)
    eval_script = os.path.join(EVALUATION_DIR, "evaluate_multiple_choice.py")
    if os.path.isfile(eval_script):
        shutil.copy2(eval_script, output_dir)

    print("\n============================================================")
    print("  选择题评测集构建统计 (V6)")
    print("============================================================")
    print(f"  数据源组数:                  {len(DATA_SOURCES)}")
    print(f"  table 文件总数:              {stats['total_tables']}")
    for filt in get_enabled_filters():
        print(f"  被 {filt.__name__:36s} 淘汰: {filter_stats[filt.__name__]}")
    print(f"  最优条件为空跳过:            {stats['empty_skipped']}")
    print(f"  无变化条件跳过:              {stats['no_varying_skipped']}")
    if stats.get("all_varying_single_key_skipped"):
        print(f"  all_varying仅1键变化跳过:    {stats['all_varying_single_key_skipped']}")
    if stats["dup_skipped"] > 0:
        print(f"  跨源 id 重复跳过:            {stats['dup_skipped']}")
    print(f"  使用的表格数:                {stats['tables_used']}")
    print(f"  生成题目总数:                {stats['questions_generated']}")
    for question_type in QUESTION_TYPE_TO_FILE:
        print(f"    - {question_type:30s}: {len(dataset_by_type[question_type])}")
    print(f"  reference 尝试数:            {stats['reference_attempts']}")
    print(f"  reference 生成数:            {stats['reference_generated']}")
    for reason, count in sorted(stats["reference_skipped"].items()):
        print(f"    - 跳过({reason}): {count}")
    print("============================================================")


if __name__ == "__main__":
    main()
