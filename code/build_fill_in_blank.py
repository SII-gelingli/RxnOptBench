"""
填空题评测集构建脚本 (V6)

本脚本负责：
1. 从原始 table 数据构建填空题数据集
2. 为每个变化条件键生成候选编号表
3. 生成 prompt 模板、示例预测和 `report.md`
4. 为输出目录复制构建/评测脚本与共享 helper 快照

填空题的核心规则是：
1. 模型不直接填写条件文本，而是按条件键选择候选编号
2. 只有命中真实存在的条件组合时才计分
3. reference 模式只提供额外参考案例，不改变实测联合行命中规则
"""

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Union

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
    "all_blank": "fill_in_blank_all_blank.json",
    "all_blank_reference": "fill_in_blank_all_blank_reference.json",
}

QUESTION_TYPE_TO_PROMPT = {
    "all_blank": "prompt_all_blank.txt",
    "all_blank_reference": "prompt_all_blank_reference.txt",
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
    "BALANCE_ALL_VARYING_OPTIONS": "是否对 all_blank 候选组合做均衡化",
    "BALANCE_MAX_IMBALANCE_RATIO": "均衡化目标上界",
    "BALANCE_MIN_OPTIONS": "均衡化保留的最少选项数",
    "MIN_REFERENCE_OPTIONS": "reference 模式下至少提供几个 reference 条件",
    "MAX_REFERENCE_OPTIONS": "reference 模式下最多提供几个 reference 条件",
    "MIN_REFERENCE_REMAINING_EFFECTIVE_OPTIONS": "去掉 reference 后至少保留多少个有效组合",
    "REFERENCE_LOW_MID_FRACTION": "reference 仅从低到中等表现区间抽取的比例",
}

SCORING_RULE_SUMMARY = (
    "仅当模型选择出的候选编号组合命中表格中真实存在的条件组合时才计分；命中后按 effective = "
    "yield * enantio_major_fraction * dr_major_fraction * rr_major_fraction 计算相对得分，"
    "对映选择性优先用 ee，ee 缺失时用 er；缺失 ee/er/dr/rr 时按 1.0 处理。"
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


def _record_reference_outcome(stats: dict, reference_result: dict):
    stats["reference_attempts"] += 1
    reason = reference_result["info"].get("reason", "")
    if reason == "ok":
        stats["reference_generated"] += 1
    else:
        stats["reference_skipped"][reason] = stats["reference_skipped"].get(reason, 0) + 1


def _build_candidate_values(option_bundles: List[dict], query_keys: List[str]):
    candidate_values = {}
    value_index = {}
    for key in query_keys:
        values = []
        lookup = {}
        for bundle in option_bundles:
            items = bundle["option"].get(key, [])
            serialized = core._normalize_for_comparison(items)
            if serialized in lookup:
                continue
            idx = len(values)
            lookup[serialized] = idx
            values.append({
                "index": idx,
                "value": core._clone_condition_items(items),
            })
        candidate_values[key] = values
        value_index[key] = lookup
    return candidate_values, value_index


def _build_legal_combinations(option_bundles: List[dict], query_keys: List[str], value_index: dict):
    scores = [bundle["score"] for bundle in option_bundles]
    relative_scores = compute_relative_scores(scores)
    combinations = []
    best_selections = []
    for bundle, relative_score in zip(option_bundles, relative_scores):
        selection = {
            key: value_index[key][core._normalize_for_comparison(bundle["option"].get(key, []))]
            for key in query_keys
        }
        record = {
            "selection": selection,
            "relative_score": relative_score,
            "yield": bundle["yield"],
            "effective_score": round(core.compute_effective_product_score(bundle["score"]), 6),
            "score": core.score_to_dict(bundle["score"]),
            "source_entry": bundle["entry_info"],
        }
        combinations.append(record)
        if relative_score == 1.0:
            best_selections.append(selection)
    return combinations, best_selections


def _build_meta(table_data, context, common, table_id, legal_combinations):
    return {
        "source_file": context["filename"],
        "doi": context.get("doi", ""),
        "best_yield": common["max_yield"],
        "best_effective_score": common["best_effective_score"],
        "best_entry_indices": common["best_entry_indices"],
        "total_entries_in_source_table": len(table_data),
        "parent_table_id": table_id,
        "main_product_smiles": common["main_product_smiles"],
        "legal_combinations": legal_combinations,
    }


def _make_all_blank_question(
    qid,
    question_type,
    query_keys,
    reactants_info,
    products_info,
    constant_conditions,
    candidate_values,
    answer,
    meta,
    reference_records=None,
):
    input_payload = {
        "reactants": reactants_info,
        "products": products_info,
        "constant_conditions": constant_conditions,
        "candidate_values": candidate_values,
    }
    if reference_records:
        input_payload["reference_conditions"] = reference_records
    return {
        "id": qid,
        "question_type": question_type,
        "query_keys": list(query_keys),
        "input": input_payload,
        "answer": answer,
        "meta": meta,
    }


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

    bundles, balance_info = core.build_all_option_bundles(common, table_data, varying_keys)
    if len(bundles) < 2:
        return []

    prepared = core.prepare_condition_view(
        bundles,
        common["condition_keys"],
        varying_keys,
        constant_conditions,
    )
    query_keys = prepared["visible_varying_keys"]
    if not query_keys:
        return []

    option_bundles = prepared["option_bundles"]

    if core.count_effectively_varying_keys(option_bundles, query_keys) <= 1:
        stats.setdefault("all_blank_single_key_skipped", 0)
        stats["all_blank_single_key_skipped"] += 1
        return []

    candidate_values, value_index = _build_candidate_values(option_bundles, query_keys)
    legal_combinations, best_selections = _build_legal_combinations(option_bundles, query_keys, value_index)
    if not core.has_score_variation([b["score"] for b in option_bundles]):
        return []

    qid = f"{table_id}_q1"
    questions = [
        _make_all_blank_question(
            qid,
            "all_blank",
            query_keys,
            common["reactants_info"],
            common["products_info"],
            prepared["constant_conditions"],
            candidate_values,
            best_selections,
            _build_meta(table_data, context, common, table_id, legal_combinations),
        )
    ]

    reference_result = core.select_reference_bundles(
        option_bundles,
        min_references=MIN_REFERENCE_OPTIONS,
        max_references=MAX_REFERENCE_OPTIONS,
        min_remaining_effective_options=MIN_REFERENCE_REMAINING_EFFECTIVE_OPTIONS,
        low_mid_fraction=REFERENCE_LOW_MID_FRACTION,
    )
    _record_reference_outcome(stats, reference_result)
    if reference_result["info"]["reason"] == "ok":
        reference_candidate_values, reference_value_index = _build_candidate_values(
            reference_result["remaining"],
            query_keys,
        )
        reference_legal_combinations, reference_best_selections = _build_legal_combinations(
            reference_result["remaining"],
            query_keys,
            reference_value_index,
        )
        if not core.has_score_variation([b["score"] for b in reference_result["remaining"]]):
            return questions
        questions.append(
            _make_all_blank_question(
                f"{qid}_ref",
                "all_blank_reference",
                query_keys,
                common["reactants_info"],
                common["products_info"],
                prepared["constant_conditions"],
                reference_candidate_values,
                reference_best_selections,
                _build_meta(
                    table_data,
                    context,
                    common,
                    table_id,
                    reference_legal_combinations,
                ),
                reference_records=build_reference_records(reference_result["references"]),
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
    lines.append("# 填空题评测集构建报告 (V6)\n")
    lines.append(f"> 生成时间: {ts}")
    lines.append(f"> 输出目录: `{os.path.basename(output_dir)}`")
    lines.append(f"> 题目总数: `{total_questions}`\n")
    lines.append("## 1. 先看什么\n")
    lines.append("如果你是第一次接手这个目录，推荐按下面顺序阅读：")
    lines.append("1. 先读 `report.md` 当前节与后面的“文件导航 / 最小接入路径 / JSON 结构”。")
    lines.append("2. 再打开一个目标数据集 JSON，例如 `fill_in_blank_all_blank_reference.json` 看真实样本。")
    lines.append("3. 根据样本的 `question_type` 选择对应 prompt 文件。")
    lines.append("4. 参考 `example_prediction_*.json` 确认输出格式。")
    lines.append("5. 最后用 `evaluate_fill_in_blank.py` 跑评测；如果要改构建或评分，再看 `build_fill_in_blank.py` 与 `common_v6.py`。")
    lines.append("")
    lines.append("## 2. 文件导航\n")
    lines.append("| 你现在要做什么 | 优先读取哪些文件 | 为什么 |")
    lines.append("|------|------|------|")
    lines.append("| 快速理解这个目录 | `report.md` | 总览题型、字段、命中规则和命令 |")
    lines.append("| 看真实输入样本 | `fill_in_blank_all_blank.json` / `fill_in_blank_all_blank_reference.json` | 真实 schema 以这里为准 |")
    lines.append("| 渲染题目给模型 | `prompt_all_blank.txt` / `prompt_all_blank_reference.txt` + 对应数据集 JSON | prompt 模板与样本字段需要一起看 |")
    lines.append("| 对接模型输出格式 | `example_prediction_fill_in_blank_*.json` | 最快确认 `id -> {key: candidate_index}` 契约 |")
    lines.append("| 运行评测 | `evaluate_fill_in_blank.py` | 命令入口与指标口径都在这里 |")
    lines.append("| 修改构题逻辑 | `build_fill_in_blank.py` | 题型、report、prompt 都由这里导出 |")
    lines.append("| 修改共享评分/解析逻辑 | `common_v6.py` | 选择题与填空题共用 helper |")
    lines.append("")
    lines.append("## 3. 最小接入路径\n")
    lines.append(
        "把本目录当成自包含评测包使用即可：读取数据集 JSON，按 `input.candidate_values` 渲染 prompt，"
        "生成 `id -> {condition_key: candidate_index}` 的预测文件，再用 `evaluate_fill_in_blank.py` 评分。"
    )
    lines.append("")
    lines.append("### 3.1 最短工作流")
    lines.append("1. 选一个数据集文件。")
    lines.append("2. 根据 `question_type` 读取对应 prompt 模板。")
    lines.append("3. 用样本里的 `input` 字段渲染 prompt。")
    lines.append("4. 让模型只输出编号，不要输出条件名。")
    lines.append("5. 把所有结果保存成 `prediction.json`。")
    lines.append("6. 运行 `evaluate_fill_in_blank.py --dataset ... --prediction prediction.json`。")
    lines.append("")
    lines.append("## 4. 题型与命中规则\n")
    lines.append("| question_type | 你会看到什么 | 评测时要注意什么 |")
    lines.append("|------|------|------|")
    lines.append("| `all_blank` | 为所有变化条件键提供候选编号表 | 只有命中 `meta.legal_combinations` 中的实测联合组合才计分 |")
    lines.append("| `all_blank_reference` | 在 `all_blank` 基础上额外给 reference 条件组与结果 | reference 只提供额外线索，不会成为可命中的实测答案 |")
    lines.append("")
    lines.append("### 4.1 命中式评分规则")
    lines.append(f"- {SCORING_RULE_SUMMARY}")
    lines.append("- 若预测的编号组合未命中任何实测联合组合，则记为 `unhit`；它不表示化学上无效，也不进入 `Score | Hit` 分母。")
    lines.append("- 评测同时报告 `coverage`、`hit_rate`、`strict_accuracy`、`avg_relative_score`（`Score | Hit`）、`num_unhit` 和 `num_invalid`。")
    lines.append("")
    lines.append("## 5. 输出文件\n")
    lines.append("### 5.1 数据集")
    lines.append("| 文件 | question_type | 样本数 |")
    lines.append("|------|------|------|")
    for question_type, filename in QUESTION_TYPE_TO_FILE.items():
        lines.append(f"| `{filename}` | `{question_type}` | {len(dataset_by_type[question_type])} |")
    lines.append("")
    lines.append("### 5.2 Prompt")
    lines.append("| 文件 | 对应题型 | 用途 |")
    lines.append("|------|------|------|")
    for question_type, filename in QUESTION_TYPE_TO_PROMPT.items():
        lines.append(f"| `{filename}` | `{question_type}` | 渲染给模型的指令模板 |")
    lines.append("")
    lines.append("### 5.3 示例预测")
    lines.append("| 文件 | 对应数据集 | 用途 |")
    lines.append("|------|------|------|")
    for question_type, filename in QUESTION_TYPE_TO_FILE.items():
        lines.append(
            f"| `{example_prediction_files[question_type]}` | `{filename}` | 用于快速验证预测 JSON 结构与评测链路 |"
        )
    lines.append("")
    lines.append("### 5.4 辅助文件")
    lines.append("| 文件 | 作用 | 什么时候读 |")
    lines.append("|------|------|------|")
    lines.append("| `report.md` | 当前目录的总览与接入指南 | 第一次接手输出目录时先读 |")
    lines.append("| `README.md` | benchmark 数据根目录说明快照 | 想快速了解整个数据目录而不仅是本批输出时读 |")
    lines.append("| `build_fill_in_blank.py` | 填空题构建脚本快照 | 需要理解题目是怎么生成时读 |")
    lines.append("| `evaluate_fill_in_blank.py` | 填空题评测脚本 | 需要跑分、看指标口径时读 |")
    lines.append("| `common_v6.py` | 共享解析与评分 helper | 需要修改公共逻辑或核对评分时读 |")
    lines.append("")
    lines.append("## 6. JSON 结构\n")
    lines.append("### 6.1 顶层字段")
    lines.append("| 字段 | 说明 |")
    lines.append("|------|------|")
    lines.append("| `id` | 题目唯一标识，格式为 `{table_id}_q1[_ref]` |")
    lines.append("| `question_type` | 题型标识（`all_blank` 或 `all_blank_reference`） |")
    lines.append("| `query_keys` | 需要填写的条件键列表 |")
    lines.append("| `input` | 渲染 prompt 所需的全部信息 |")
    lines.append("| `answer` | 最优条件组合对应的编号选择列表，仅用于诊断/示例 |")
    lines.append("| `meta` | 题目构建元信息，详见下方 |")
    lines.append("")
    lines.append("### 6.2 `input` 子字段")
    lines.append("| 字段 | 说明 |")
    lines.append("|------|------|")
    lines.append("| `input.reactants` | 反应物列表，每项含 `content`（名称）和 `SMILES` |")
    lines.append("| `input.products` | 产物列表，同上 |")
    lines.append("| `input.constant_conditions` | 所有选项共享的不变条件 |")
    lines.append("| `input.candidate_values` | 每个 `query_key` 的候选编号表，模型只能从这里选编号 |")
    lines.append("| `input.reference_conditions` | 仅 `all_blank_reference` 题出现；提供额外线索的参考条件组与结果 |")
    lines.append("")
    lines.append("### 6.3 `meta` 子字段")
    lines.append("| 字段 | 说明 |")
    lines.append("|------|------|")
    lines.append("| `meta.source_file` | 该题来源的原始 JSON 文件名 |")
    lines.append("| `meta.doi` | 来源论文的 DOI |")
    lines.append("| `meta.best_yield` | 所有条件组合中的最佳产率 |")
    lines.append("| `meta.best_effective_score` | 所有条件组合中的最佳 effective score（综合 yield、ee、dr 等） |")
    lines.append("| `meta.best_entry_indices` | 最优条目在原表中的行索引 |")
    lines.append("| `meta.total_entries_in_source_table` | 源表（原始 JSON 文件）中的总条目数，不是候选值或选项的数量 |")
    lines.append("| `meta.parent_table_id` | 源表标识（通常为 root_img_name 或文件名去扩展名） |")
    lines.append("| `meta.main_product_smiles` | 用于评分的主产物 SMILES |")
    lines.append("| `meta.legal_combinations` | 历史字段名；内容是来源表实测联合组合及其相对分数，每项也带 `effective_score` |")
    lines.append("")
    lines.append("## 7. 评测脚本")
    lines.append("预测文件格式：")
    lines.append("```json")
    lines.append("{")
    lines.append('  "sample_id_1": {"solvents": 2, "catalysts": 0},')
    lines.append('  "sample_id_2": {"solvents": 1, "ligand": 3}')
    lines.append("}")
    lines.append("```")
    lines.append("```bash")
    lines.append("python evaluate_fill_in_blank.py \\")
    lines.append("  --dataset fill_in_blank_all_blank_reference.json \\")
    lines.append("  --prediction pred.json")
    lines.append("```")
    lines.append("")
    lines.append("## 8. Reference 规则")
    lines.append("- reference 条件数至少 `2` 条、至多 `5` 条。")
    lines.append("- `all_blank_reference` 的 `candidate_values` 与 `legal_combinations` 仅来自去掉 reference 后的剩余组合。")
    lines.append("- 因此，被展示成 reference 的条件组合不再属于可命中的实测答案集合。")
    lines.append("- 不满足 reference 约束的样本只保留 `all_blank`。")
    lines.append("")
    lines.append("## 9. 构建设置\n")
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
    lines.append("## 10. 统计摘要\n")
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
    lines.append("## 11. 样本展示")
    for question_type in ["all_blank", "all_blank_reference"]:
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
        if inp.get("reference_conditions"):
            lines.append("**Reference 条件：**")
            lines.extend(_format_reference_block(inp["reference_conditions"]))
        lines.append("**候选编号：**")
        for key in sample["query_keys"]:
            lines.append(f"- `{key}`")
            for candidate in inp["candidate_values"][key]:
                lines.append(f"  - {candidate['index']}: {core._fmt_condition_items(candidate['value'])}")
        lines.append("")
        lines.append(f"**最优组合:** `{sample['answer']}`")
        lines.append(f"**实测联合组合数:** `{len(sample['meta']['legal_combinations'])}`\n")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


PROMPT_ALL_BLANK = r"""You are an expert organic chemist. Given the following reaction, choose one candidate index for each required condition category.

### Reaction

**Reactants:**
{reactants}

**Target Product:**
{products}

### Constant Conditions

{constant_conditions}

### Candidate Values

{candidate_values}

### Output Format

Respond strictly in the following JSON format (no extra text):

```json
{{
{query_keys_json}
}}
```

Each value must be a single integer candidate index, not a condition name.
"""

PROMPT_ALL_BLANK_REFERENCE = r"""You are an expert organic chemist. Given the following reaction, several reference condition sets with observed outcomes, and a candidate index table for each condition category, choose one candidate index for each required condition category.

### Reaction

**Reactants:**
{reactants}

**Target Product:**
{products}

### Constant Conditions

{constant_conditions}

### Reference Conditions And Outcomes

{references}

Important: the reference condition sets shown above are NOT valid answers for this question. Do NOT copy any reference combination directly. You must propose a NEW combination by selecting candidate indices from the table below. Only combinations formed from the candidate index table are considered valid outputs.

### Candidate Values

{candidate_values}

### Output Format

Respond strictly in the following JSON format (no extra text):

```json
{{
{query_keys_json}
}}
```

Each value must be a single integer candidate index, not a condition name.
"""


def generate_prompt_templates(output_dir):
    templates = {
        "prompt_all_blank.txt": PROMPT_ALL_BLANK,
        "prompt_all_blank_reference.txt": PROMPT_ALL_BLANK_REFERENCE,
    }
    for filename, template in templates.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(template.strip() + "\n")


def generate_example_prediction(dataset, output_dir, filename="example_prediction.json"):
    rng = random.Random(42)
    predictions = {}
    for sample in dataset:
        answer = sample["answer"][0]
        prediction = dict(answer)
        if rng.random() < 0.35:
            key = sample["query_keys"][0]
            n_candidates = len(sample["input"]["candidate_values"][key])
            prediction[key] = n_candidates + 5 if rng.random() < 0.5 else max(0, prediction[key] - 1)
        predictions[sample["id"]] = prediction
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
            if question["question_type"] in {"all_blank"}
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
        description="Build the submitted V6 fill-in-the-blank benchmark from table JSON files."
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
    output_dir = os.path.join(base_dir, f"{timestamp}_fill_in_blank")
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
        print(f"[填空题 V6] 已加载 SIOC 文件名→DOI 映射表: {len(filename_doi_map)} 条")

    print(f"[填空题 V6] 共 {len(DATA_SOURCES)} 组数据源")
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
    eval_script = os.path.join(EVALUATION_DIR, "evaluate_fill_in_blank.py")
    if os.path.isfile(eval_script):
        shutil.copy2(eval_script, output_dir)

    print("\n============================================================")
    print("  填空题评测集构建统计 (V6)")
    print("============================================================")
    print(f"  数据源组数:                  {len(DATA_SOURCES)}")
    print(f"  table 文件总数:              {stats['total_tables']}")
    for filt in get_enabled_filters():
        print(f"  被 {filt.__name__:36s} 淘汰: {filter_stats[filt.__name__]}")
    print(f"  最优条件为空跳过:            {stats['empty_skipped']}")
    print(f"  无变化条件跳过:              {stats['no_varying_skipped']}")
    if stats.get("all_blank_single_key_skipped"):
        print(f"  all_blank仅1键变化跳过:      {stats['all_blank_single_key_skipped']}")
    if stats["dup_skipped"] > 0:
        print(f"  跨源 id 重复跳过:            {stats['dup_skipped']}")
    print(f"  使用的表格数:                {stats['tables_used']}")
    print(f"  生成题目总数:                {stats['questions_generated']}")
    for question_type in QUESTION_TYPE_TO_FILE:
        print(f"    - {question_type:24s}: {len(dataset_by_type[question_type])}")
    print(f"  reference 尝试数:            {stats['reference_attempts']}")
    print(f"  reference 生成数:            {stats['reference_generated']}")
    for reason, count in sorted(stats["reference_skipped"].items()):
        print(f"    - 跳过({reason}): {count}")
    print("============================================================")


if __name__ == "__main__":
    main()
