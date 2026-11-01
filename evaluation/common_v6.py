import csv
import json
import math
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

YEAR_FILTER = None
CONDITION_KEYS = None
CONDITION_REQUIRE = "content"
SMILES_EXEMPT_KEYS = ["solvents"]
REACTANT_PRODUCT_REQUIRE = "smiles"
INPUT_PRODUCTS_MODE = "main_only"
YIELD_FIELD_PRIORITY = ["isolated_yield", "yield_ratio"]
EXPAND_CURRENT_CONDITIONS = True
CONDITION_KEYS_BLACKLIST = None
NORMALIZE_CONDITIONS = True
BALANCE_ALL_VARYING_OPTIONS = False
BALANCE_MAX_IMBALANCE_RATIO = 1.5
BALANCE_MIN_OPTIONS = 4


def configure_runtime(**kwargs):
    allowed = {
        "YEAR_FILTER",
        "CONDITION_KEYS",
        "CONDITION_REQUIRE",
        "SMILES_EXEMPT_KEYS",
        "REACTANT_PRODUCT_REQUIRE",
        "INPUT_PRODUCTS_MODE",
        "YIELD_FIELD_PRIORITY",
        "EXPAND_CURRENT_CONDITIONS",
        "CONDITION_KEYS_BLACKLIST",
        "NORMALIZE_CONDITIONS",
        "BALANCE_ALL_VARYING_OPTIONS",
        "BALANCE_MAX_IMBALANCE_RATIO",
        "BALANCE_MIN_OPTIONS",
    }
    for key, value in kwargs.items():
        if key in allowed:
            globals()[key] = value


_TABLE_FILENAME_RE = re.compile(
    r"^(?P<doi_token>.+)_(?P<page>\d+)_(?P<kind>table|scope)_(?P<table_idx>\d+)\.json$"
)

_SUPER_SUB_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
    "ⁿ": "n", "ⁱ": "i",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
    "ₐ": "a", "ₑ": "e", "ₒ": "o", "ₓ": "x", "ₕ": "h",
    "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₚ": "p",
    "ₛ": "s", "ₜ": "t",
    "ᵃ": "a", "ᵇ": "b", "ᵈ": "d", "ᵉ": "e", "ᶠ": "f",
    "ᵍ": "g", "ᵏ": "k", "ᵐ": "m", "ᵒ": "o", "ᵖ": "p",
    "ᵗ": "t", "ᵘ": "u", "ᵛ": "v", "ʳ": "r", "ˢ": "s",
    "×": "x",
})
_REMOVE_CHARS = "℃℉°™®©"
_SMILES_KEYS = {"SMILES", "smiles", "main_product_smiles"}
_YIELD_NODATA = {
    "n.d.", "nd", "n,d,", "notdetected", "not detected", "notobtained",
    "not obtained", "n.r.", "nr", "noreaction", "no reaction", "n.p.",
    "n.a.", "messy", "—", "–", "-",
}
_YIELD_TRACE = {"trace", "traces"}

CONDITION_LABEL = {
    "reagents": "试剂 (reagents)",
    "solvents": "溶剂 (solvents)",
    "catalysts": "催化剂 (catalysts)",
    "ligand": "配体 (ligand)",
}


def normalize_condition_string(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s.strip())
    s = s.translate(_SUPER_SUB_MAP)
    for ch in _REMOVE_CHARS:
        s = s.replace(ch, "")
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    return s


def _normalize_for_comparison(val: list) -> str:
    if NORMALIZE_CONDITIONS:
        normalized = [
            {
                "content": normalize_condition_string(item.get("content", "")),
                "SMILES": (item.get("SMILES") or "").strip(),
            }
            for item in val
        ]
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return json.dumps(val, sort_keys=True, ensure_ascii=False)


def try_canonicalize(smi: str) -> str:
    if not smi or not smi.strip():
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol)


def canonicalize_all_smiles(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _SMILES_KEYS and isinstance(value, str) and value.strip():
                obj[key] = try_canonicalize(value)
            else:
                canonicalize_all_smiles(value)
    elif isinstance(obj, list):
        for item in obj:
            canonicalize_all_smiles(item)


def load_filename_doi_mapping(csv_path: str) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_num = row.get("文件命名", "").strip()
            doi = row.get("DOI", "").strip()
            if file_num and doi:
                mapping[file_num] = {
                    "doi": doi,
                    "doi_prefix": doi.replace("/", "_"),
                }
    return mapping


def resolve_metadata_path(doi_prefix: str, metadata_dir: str) -> Tuple[Optional[str], str]:
    candidate = os.path.join(metadata_dir, f"{doi_prefix}.json")
    if os.path.isfile(candidate):
        return candidate, doi_prefix
    dot_idx = doi_prefix.find(".")
    if dot_idx > 0:
        fixed = "1" + doi_prefix[dot_idx - 1:]
        candidate = os.path.join(metadata_dir, f"{fixed}.json")
        if os.path.isfile(candidate):
            return candidate, fixed
    return None, doi_prefix


def load_metadata_year(metadata_path: str) -> Optional[int]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for key in ["published", "published-online", "published-print", "issued"]:
        date_parts = meta.get(key, {}).get("date-parts")
        if date_parts and date_parts[0]:
            return int(date_parts[0][0])
    return None


def _parse_yield_string(raw: str) -> Optional[float]:
    s = raw.strip()
    if not s:
        return None
    low = s.lower().replace(" ", "")
    if low in _YIELD_NODATA:
        return 0.0
    if low in _YIELD_TRACE:
        return 1.0
    if low in ("quant.", "quant", "quantitative"):
        return 100.0

    s = s.replace("%", "").strip()
    s = s.replace("﹥", ">").replace("＞", ">").replace("≥", ">")
    s = s.replace("﹤", "<").replace("＜", "<").replace("≤", "<")
    s = re.sub(r"ca\.?\s*", "", s)
    s = re.sub(r"[><~≈]\s*", "", s).strip()
    if ":" in s or "/" in s:
        return None

    match = re.match(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", s)
    if match:
        return max(float(match.group(1)), float(match.group(2)))

    try:
        return float(s)
    except ValueError:
        return None


def parse_yield(product: dict, yield_fields: Optional[List[str]] = None) -> float:
    if yield_fields is None:
        yield_fields = YIELD_FIELD_PRIORITY
    for field in yield_fields:
        raw = (product.get(field) or "").strip()
        if not raw:
            continue
        value = _parse_yield_string(raw)
        if value is not None:
            return value
    return 0.0


def is_yield_available(product: dict, yield_fields: Optional[List[str]] = None) -> bool:
    if yield_fields is None:
        yield_fields = YIELD_FIELD_PRIORITY
    for field in yield_fields:
        raw = (product.get(field) or "").strip()
        if not raw:
            continue
        if _parse_yield_string(raw) is not None:
            return True
    return False


def _parse_ee_string(raw: str) -> float:
    s = raw.strip()
    if not s:
        return -1.0
    low = s.lower().replace(" ", "")
    if low in ("n.a.", "nd", "n.d.", "na", "—", "-", "n.a"):
        return -1.0

    s = s.replace("﹥", ">").replace("＞", ">").replace("≥", ">")
    parts = re.split(r"[/,;]", s)
    max_abs = -1.0
    for part in parts:
        has_percent = "%" in part
        value = re.sub(r"[><~≈≥≤%]\s*", "", part.strip()).strip()
        try:
            parsed = abs(float(value))
            if not has_percent and 0.0 <= parsed <= 1.0:
                parsed *= 100.0
            max_abs = max(max_abs, parsed)
        except ValueError:
            continue
    return max_abs


def parse_ee(product: dict) -> float:
    raw = (product.get("ee") or product.get("enantiomeric_excess") or "").strip()
    return _parse_ee_string(raw)


def _parse_dr_string(raw: str) -> float:
    s = raw.strip()
    if not s:
        return -1.0
    low = s.lower().replace(" ", "")
    if low in ("n.a.", "nd", "n.d.", "na", "—", "-", "n.a"):
        return -1.0

    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return -1.0
    try:
        values = [float(num) for num in nums]
    except ValueError:
        return -1.0
    if len(values) == 1:
        value = values[0]
        if "%" in s:
            major_fraction = max(0.0, min(value / 100.0, 1.0))
            if major_fraction >= 1.0:
                return math.inf
            return major_fraction / (1.0 - major_fraction) if major_fraction > 0 else 0.0
        if 0.0 < value < 1.0 and "." in s:
            return value / (1.0 - value)
        return value
    values.sort(reverse=True)
    return values[0] / values[1] if values[1] != 0 else values[0]


def parse_dr(product: dict) -> float:
    raw = (product.get("dr") or product.get("diastereomeric_ratio") or "").strip()
    return _parse_dr_string(raw)


def parse_er(product: dict) -> float:
    raw = (product.get("er") or product.get("enantiomeric_ratio") or "").strip()
    return _parse_dr_string(raw)


def parse_rr(product: dict) -> float:
    raw = (product.get("rr") or product.get("regioisomeric_ratio") or "").strip()
    return _parse_dr_string(raw)


def product_score(product: dict, yield_fields: Optional[List[str]] = None) -> Tuple[float, float, float, float, float]:
    return (
        parse_yield(product, yield_fields),
        parse_ee(product),
        parse_dr(product),
        parse_er(product),
        parse_rr(product),
    )


def ee_to_major_fraction(ee: float) -> float:
    if ee < 0:
        return 1.0
    clamped = max(0.0, min(ee, 100.0))
    return (100.0 + clamped) / 200.0


def dr_to_major_fraction(dr: float) -> float:
    if dr < 0:
        return 1.0
    ratio = max(dr, 0.0)
    if not math.isfinite(ratio):
        return 1.0
    return ratio / (ratio + 1.0) if ratio > 0 else 0.0


def er_to_major_fraction(er: float) -> float:
    if er < 0:
        return 1.0
    ratio = max(er, 0.0)
    if not math.isfinite(ratio):
        return 1.0
    return ratio / (ratio + 1.0) if ratio > 0 else 0.0


def rr_to_major_fraction(rr: float) -> float:
    if rr < 0:
        return 1.0
    ratio = max(rr, 0.0)
    if not math.isfinite(ratio):
        return 1.0
    return ratio / (ratio + 1.0) if ratio > 0 else 0.0


def _enantio_major_fraction(ee: float, er: float) -> float:
    if ee >= 0:
        return ee_to_major_fraction(ee)
    if er >= 0:
        return er_to_major_fraction(er)
    return 1.0


def compute_effective_product_score(score: Tuple[float, ...]) -> float:
    yield_value, ee, dr = score[0], score[1], score[2]
    er = score[3] if len(score) > 3 else -1.0
    rr = score[4] if len(score) > 4 else -1.0
    if yield_value <= 0:
        return 0.0
    return (
        yield_value
        * _enantio_major_fraction(ee, er)
        * dr_to_major_fraction(dr)
        * rr_to_major_fraction(rr)
    )


def _score_group_key(score: Tuple[float, ...]) -> float:
    return round(compute_effective_product_score(score), 12)


def _score_isclose(lhs: Tuple[float, ...], rhs: Tuple[float, ...]) -> bool:
    return math.isclose(
        compute_effective_product_score(lhs),
        compute_effective_product_score(rhs),
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _is_better_score(candidate: Tuple[float, ...], current: Tuple[float, ...]) -> bool:
    candidate_effective = compute_effective_product_score(candidate)
    current_effective = compute_effective_product_score(current)
    if not math.isclose(candidate_effective, current_effective, rel_tol=0.0, abs_tol=1e-9):
        return candidate_effective > current_effective
    return candidate > current


def discover_condition_keys(table_data: list) -> List[str]:
    keys: set = set()
    for entry in table_data:
        stages = entry.get("stages", [])
        if stages:
            keys.update(stages[0].keys())
    if EXPAND_CURRENT_CONDITIONS and "current_conditions" in keys:
        keys.discard("current_conditions")
        for entry in table_data:
            stages = entry.get("stages", [])
            if not stages:
                continue
            current_conditions = stages[0].get("current_conditions")
            if isinstance(current_conditions, dict):
                keys.update(current_conditions.keys())
    return sorted(keys)


def _collect_current_condition_subkeys(table_data: list) -> List[str]:
    keys: set = set()
    for entry in table_data:
        stages = entry.get("stages", [])
        if not stages:
            continue
        current_conditions = stages[0].get("current_conditions")
        if isinstance(current_conditions, dict):
            keys.update(current_conditions.keys())
    return sorted(keys)


def _resolve_condition_keys(condition_keys: List[str], table_data: list) -> List[str]:
    if "current_conditions" not in condition_keys:
        return list(condition_keys)
    resolved: List[str] = []
    current_subkeys = _collect_current_condition_subkeys(table_data)
    for key in condition_keys:
        if key == "current_conditions":
            for subkey in current_subkeys:
                if subkey not in resolved:
                    resolved.append(subkey)
        elif key not in resolved:
            resolved.append(key)
    return resolved


def extract_conditions(
    entry: dict,
    condition_keys: Optional[List[str]] = None,
    condition_require: str = "none",
    smiles_exempt_keys: Optional[List[str]] = None,
) -> dict:
    if condition_keys is None:
        condition_keys = discover_condition_keys([entry])
    if smiles_exempt_keys is None:
        smiles_exempt_keys = SMILES_EXEMPT_KEYS

    result: Dict[str, list] = {key: [] for key in condition_keys}
    stages = entry.get("stages", [])
    if not stages:
        return result
    stage = stages[0]

    for key in condition_keys:
        value = stage.get(key)
        if value is None and EXPAND_CURRENT_CONDITIONS:
            current_conditions = stage.get("current_conditions")
            if isinstance(current_conditions, dict):
                value = current_conditions.get(key)
        if value is None:
            continue

        if isinstance(value, list):
            if not value:
                continue
            is_standard = all(
                isinstance(item, dict) and ("content" in item or "SMILES" in item)
                for item in value
            )
            if is_standard:
                for item in value:
                    content = (item.get("content") or "").strip()
                    smiles = (item.get("SMILES") or "").strip()
                    if condition_require == "content" and not content:
                        continue
                    if condition_require == "smiles" and key not in smiles_exempt_keys and not smiles:
                        continue
                    result[key].append({"content": content, "SMILES": smiles})
            else:
                serialized = json.dumps(value, ensure_ascii=False)
                if condition_require != "content" or serialized:
                    result[key].append({"content": serialized, "SMILES": ""})
        else:
            text = str(value).strip()
            if text and (condition_require != "content" or text):
                result[key].append({"content": text, "SMILES": ""})
    return result


def extract_species_info(species_list: list) -> list:
    return [
        {
            "content": (item.get("content") or "").strip(),
            "SMILES": (item.get("SMILES") or "").strip(),
        }
        for item in species_list
    ]


def reactant_product_fingerprint(entry: dict) -> Tuple[tuple, tuple]:
    reactants = tuple(sorted(item.get("SMILES", "") for item in entry.get("reactants", [])))
    products = tuple(sorted(item.get("SMILES", "") for item in entry.get("products", [])))
    return reactants, products


def identify_main_product(table_data: list, yield_fields: Optional[List[str]] = None) -> str:
    best_score = (-1.0, -1.0, -1.0, -1.0, -1.0)
    best_effective = -1.0
    main_smiles = ""
    for entry in table_data:
        for product in entry.get("products", []):
            score = product_score(product, yield_fields)
            effective_score = compute_effective_product_score(score)
            if effective_score > best_effective or (
                math.isclose(effective_score, best_effective, rel_tol=0.0, abs_tol=1e-9)
                and score > best_score
            ):
                best_effective = effective_score
                best_score = score
                main_smiles = (product.get("SMILES") or "").strip()
    return main_smiles


def get_product_score_by_smiles(
    entry: dict,
    target_smiles: str,
    yield_fields: Optional[List[str]] = None,
) -> Tuple[float, float, float, float, float]:
    for product in entry.get("products", []):
        if (product.get("SMILES") or "").strip() == target_smiles:
            return product_score(product, yield_fields)
    return 0.0, -1.0, -1.0, -1.0, -1.0


def has_yield_for_product(entry: dict, target_smiles: str, yield_fields: Optional[List[str]] = None) -> bool:
    """Return True if *target_smiles* product in *entry* has a parseable yield value.

    Empty / unparseable yield fields indicate data-extraction failure and should
    cause the entire row to be excluded from question generation.
    """
    for product in entry.get("products", []):
        if (product.get("SMILES") or "").strip() == target_smiles:
            return is_yield_available(product, yield_fields)
    return False


def filter_is_table(table_data: list, context: dict) -> bool:
    return "_table_" in context["filename"]


def filter_reactant_product_require(table_data: list, context: dict) -> bool:
    requirement = REACTANT_PRODUCT_REQUIRE
    if requirement == "none":
        return True
    for entry in table_data:
        for reactant in entry.get("reactants", []):
            if requirement == "smiles" and not (reactant.get("SMILES") or "").strip():
                return False
            if requirement == "content" and not (reactant.get("content") or "").strip():
                return False
        for product in entry.get("products", []):
            if requirement == "smiles" and not (product.get("SMILES") or "").strip():
                return False
            if requirement == "content" and not (product.get("content") or "").strip():
                return False
    return True


def filter_same_reactants_products(table_data: list, context: dict) -> bool:
    if not table_data:
        return False
    fingerprint = reactant_product_fingerprint(table_data[0])
    return all(reactant_product_fingerprint(entry) == fingerprint for entry in table_data[1:])


def filter_year_range(table_data: list, context: dict) -> bool:
    year = context.get("year")
    if year is None:
        return False
    if YEAR_FILTER is None:
        return True
    if isinstance(YEAR_FILTER, int):
        return year >= YEAR_FILTER
    if isinstance(YEAR_FILTER, list):
        return year in YEAR_FILTER
    return False


def _condition_to_key(condition: dict) -> str:
    if NORMALIZE_CONDITIONS:
        normalized = {}
        for key, value in condition.items():
            normalized[key] = [
                {
                    "content": normalize_condition_string(item.get("content", "")),
                    "SMILES": (item.get("SMILES") or "").strip(),
                }
                for item in value
            ]
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return json.dumps(condition, sort_keys=True, ensure_ascii=False)


def _is_empty_condition(condition: dict) -> bool:
    return all(len(value) == 0 for value in condition.values())


def _clone_condition_items(items: Sequence[dict]) -> List[dict]:
    return [dict(item) for item in items]


def project_condition_dict(condition: Dict[str, List[dict]], keep_keys: Sequence[str]) -> Dict[str, List[dict]]:
    keep = set(keep_keys)
    return {
        key: _clone_condition_items(value)
        for key, value in condition.items()
        if key in keep
    }


def select_active_condition_keys(
    condition_dicts: Sequence[Dict[str, List[dict]]],
    candidate_keys: Sequence[str],
) -> List[str]:
    active: List[str] = []
    for key in candidate_keys:
        if any(condition.get(key, []) for condition in condition_dicts):
            active.append(key)
    return active


def detect_varying_conditions(all_conditions: list, condition_keys: List[str]) -> List[str]:
    varying = []
    for key in condition_keys:
        values = {_normalize_for_comparison(cond.get(key, [])) for cond in all_conditions}
        if len(values) > 1:
            varying.append(key)
    return varying


def _build_common_fields(
    table_data,
    context,
    condition_keys=None,
    yield_fields=None,
    condition_require=None,
    smiles_exempt_keys=None,
    input_products_mode=None,
):
    if condition_keys is None:
        condition_keys = discover_condition_keys(table_data)
    elif EXPAND_CURRENT_CONDITIONS:
        condition_keys = _resolve_condition_keys(condition_keys, table_data)

    if CONDITION_KEYS_BLACKLIST:
        blacklist = set(CONDITION_KEYS_BLACKLIST)
        if EXPAND_CURRENT_CONDITIONS and "current_conditions" in blacklist:
            blacklist.update(_collect_current_condition_subkeys(table_data))
        condition_keys = [key for key in condition_keys if key not in blacklist]

    if yield_fields is None:
        yield_fields = YIELD_FIELD_PRIORITY
    if condition_require is None:
        condition_require = CONDITION_REQUIRE
    if smiles_exempt_keys is None:
        smiles_exempt_keys = SMILES_EXEMPT_KEYS
    if input_products_mode is None:
        input_products_mode = INPUT_PRODUCTS_MODE

    first = table_data[0]
    main_product_smiles = identify_main_product(table_data, yield_fields)
    reactants_info = extract_species_info(first["reactants"])

    if input_products_mode == "main_only":
        products_info = []
        for product in first["products"]:
            if (product.get("SMILES") or "").strip() == main_product_smiles:
                products_info = [{
                    "content": (product.get("content") or "").strip(),
                    "SMILES": main_product_smiles,
                }]
                break
        if not products_info:
            products_info = extract_species_info(first["products"][:1])
    else:
        products_info = extract_species_info(first["products"])

    yields = []
    scores = []
    effective_scores = []
    all_conditions = []
    for entry in table_data:
        if not has_yield_for_product(entry, main_product_smiles, yield_fields):
            yields.append(None)
            scores.append(None)
            effective_scores.append(None)
        else:
            score = get_product_score_by_smiles(entry, main_product_smiles, yield_fields)
            yields.append(score[0])
            scores.append(score)
            effective_scores.append(compute_effective_product_score(score))
        all_conditions.append(
            extract_conditions(entry, condition_keys, condition_require, smiles_exempt_keys)
        )

    valid_effectives = [e for e in effective_scores if e is not None]
    if not valid_effectives:
        max_effective_score = 0.0
        best_indices = []
    else:
        max_effective_score = max(valid_effectives)
        best_indices = [
            idx
            for idx, effective_score in enumerate(effective_scores)
            if effective_score is not None
            and math.isclose(effective_score, max_effective_score, rel_tol=0.0, abs_tol=1e-9)
        ]
    best_conditions = [all_conditions[idx] for idx in best_indices]
    best_entry_indices = [
        table_data[idx]["meta_info"].get("entry_idx", str(idx))
        for idx in best_indices
    ]
    return {
        "reactants_info": reactants_info,
        "products_info": products_info,
        "main_product_smiles": main_product_smiles,
        "yields": yields,
        "scores": scores,
        "effective_scores": effective_scores,
        "best_effective_score": max_effective_score,
        "max_yield": max((y for y in yields if y is not None), default=0.0),
        "best_indices": best_indices,
        "best_conditions": best_conditions,
        "best_entry_indices": best_entry_indices,
        "all_conditions": all_conditions,
        "condition_keys": condition_keys,
    }


def _extract_entry_info(table_data, row_idx):
    meta_info = table_data[row_idx].get("meta_info", {})
    return {
        "row_index": row_idx,
        "entry_idx": meta_info.get("entry_idx", str(row_idx)),
        "rxn_id": meta_info.get("rxn_id", ""),
    }


def _compute_key_imbalance(options, varying_keys):
    total = len(options)
    if total <= 1:
        return 1.0, {}
    key_ratios = {}
    for key in varying_keys:
        freq: Dict[str, int] = {}
        for option in options:
            value = _normalize_for_comparison(option.get(key, []))
            freq[value] = freq.get(value, 0) + 1
        n_values = len(freq)
        if n_values <= 1:
            key_ratios[key] = 1.0
            continue
        ideal = total / n_values
        key_ratios[key] = max(freq.values()) / ideal
    worst = max(key_ratios.values()) if key_ratios else 1.0
    return worst, key_ratios


def _balance_all_varying_options(option_list, yield_list, score_list, entry_info_list, varying_keys):
    total = len(option_list)
    original_imbalance, _ = _compute_key_imbalance(option_list, varying_keys)

    def _make_info(balanced_count, balanced_imbalance):
        return {
            "enabled": True,
            "original_count": total,
            "balanced_count": balanced_count,
            "removed_count": total - balanced_count,
            "original_max_imbalance": round(original_imbalance, 3),
            "balanced_max_imbalance": round(balanced_imbalance, 3),
        }

    if total <= max(BALANCE_MIN_OPTIONS, 2) or original_imbalance <= BALANCE_MAX_IMBALANCE_RATIO:
        return option_list, yield_list, score_list, entry_info_list, _make_info(total, original_imbalance)

    effective_scores = [compute_effective_product_score(score) for score in score_list]
    max_effective = max(effective_scores)
    protected = {
        idx
        for idx, effective_score in enumerate(effective_scores)
        if math.isclose(effective_score, max_effective, rel_tol=0.0, abs_tol=1e-9)
    }
    remaining = list(range(total))
    while len(remaining) > max(BALANCE_MIN_OPTIONS, 2):
        current_options = [option_list[idx] for idx in remaining]
        worst, _ = _compute_key_imbalance(current_options, varying_keys)
        if worst <= BALANCE_MAX_IMBALANCE_RATIO:
            break
        removable = [idx for idx in remaining if idx not in protected]
        if not removable:
            break
        best_candidate = None
        best_new_worst = worst
        for candidate in removable:
            trial = [idx for idx in remaining if idx != candidate]
            trial_options = [option_list[idx] for idx in trial]
            new_worst, _ = _compute_key_imbalance(trial_options, varying_keys)
            if new_worst < best_new_worst:
                best_new_worst = new_worst
                best_candidate = candidate
        if best_candidate is None:
            break
        remaining.remove(best_candidate)

    final_options = [option_list[idx] for idx in remaining]
    final_yields = [yield_list[idx] for idx in remaining]
    final_scores = [score_list[idx] for idx in remaining]
    final_entries = [entry_info_list[idx] for idx in remaining]
    balanced_imbalance, _ = _compute_key_imbalance(final_options, varying_keys)
    return final_options, final_yields, final_scores, final_entries, _make_info(len(remaining), balanced_imbalance)


def _fmt_species(species_list, label):
    parts = []
    for species in species_list:
        name = species.get("content", "")
        smiles = species.get("SMILES", "")
        if name and smiles:
            parts.append(f"{name} (`{smiles}`)")
        elif name:
            parts.append(name)
        elif smiles:
            parts.append(f"`{smiles}`")
        else:
            parts.append("(空)")
    return f"- **{label}**: {' + '.join(parts)}"


def _fmt_condition_items(items):
    if not items:
        return "—"
    parts = []
    for item in items:
        name = item.get("content", "")
        smiles = item.get("SMILES", "")
        if name and smiles:
            parts.append(f"{name} (`{smiles}`)")
        elif name:
            parts.append(name)
        elif smiles:
            parts.append(f"`{smiles}`")
        else:
            parts.append("(空)")
    return ", ".join(parts)


def _fmt_condition(cond):
    lines = []
    for key, items in cond.items():
        label = CONDITION_LABEL.get(key, key)
        lines.append(f"  - **{label}**: {_fmt_condition_items(items)}")
    return "\n".join(lines)


def parse_table_filename(filename: str) -> Optional[Dict[str, Any]]:
    match = _TABLE_FILENAME_RE.match(filename)
    if not match:
        return None
    return {
        "doi_token": match.group("doi_token"),
        "page": int(match.group("page")),
        "kind": match.group("kind"),
        "table_idx": int(match.group("table_idx")),
    }


def extract_doi_token(filename: str) -> Optional[str]:
    parsed = parse_table_filename(filename)
    if not parsed:
        return None
    doi_token = parsed["doi_token"]
    # Strip supplementary information marker (_si or _si_<digits>) from DOI token
    doi_token = re.sub(r"_si(?:_\d+)*$", "", doi_token)
    return doi_token


def score_to_dict(score: Tuple[float, ...]) -> Dict[str, Optional[float]]:
    er = score[3] if len(score) > 3 else -1.0
    rr = score[4] if len(score) > 4 else -1.0
    return {
        "yield": score[0],
        "ee": score[1] if score[1] >= 0 else None,
        "dr": score[2] if score[2] >= 0 else None,
        "er": er if er >= 0 else None,
        "rr": rr if rr >= 0 else None,
    }


def build_context(filename: str, filepath: str, metadata_dir: str, filename_doi_map: dict) -> dict:
    doi_token = extract_doi_token(filename)
    if doi_token and doi_token in filename_doi_map:
        mapped = filename_doi_map[doi_token]
        meta_path, normalized_prefix = resolve_metadata_path(mapped["doi_prefix"], metadata_dir)
        doi_display = mapped["doi"]
        doi_prefix = normalized_prefix or mapped["doi_prefix"]
    elif doi_token:
        meta_path, normalized_prefix = resolve_metadata_path(doi_token, metadata_dir)
        doi_display = normalized_prefix or doi_token
        doi_prefix = normalized_prefix or doi_token
    else:
        meta_path = None
        doi_display = ""
        doi_prefix = ""

    year = load_metadata_year(meta_path) if meta_path else None
    return {
        "filename": filename,
        "filepath": filepath,
        "doi_prefix": doi_prefix,
        "doi": doi_display,
        "year": year,
    }


def _subset_condition_key(condition: Dict[str, List[dict]], keys: Sequence[str]) -> str:
    return json.dumps(
        {key: _normalize_for_comparison(condition.get(key, [])) for key in keys},
        sort_keys=True,
        ensure_ascii=False,
    )


def _clone_condition_dict(condition: Dict[str, List[dict]], keys: Sequence[str]) -> Dict[str, List[dict]]:
    return {key: _clone_condition_items(condition.get(key, [])) for key in keys}


def _best_bundle_score(bundles: Sequence[Dict[str, Any]]):
    if not bundles:
        return 0.0, -1.0, -1.0, -1.0, -1.0
    return max(
        bundles,
        key=lambda bundle: (compute_effective_product_score(bundle["score"]), bundle["score"]),
    )["score"]


def _mean_effective_score(bundles: Sequence[Dict[str, Any]]) -> float:
    if not bundles:
        return 0.0
    return sum(compute_effective_product_score(bundle["score"]) for bundle in bundles) / len(bundles)


def _build_distinct_varying_bundles(common, table_data, varying_keys: Sequence[str]) -> List[Dict[str, Any]]:
    all_conditions = common["all_conditions"]
    yields = common["yields"]
    scores = common["scores"]

    seen: Dict[str, Dict[str, Any]] = {}
    for row_idx, condition in enumerate(all_conditions):
        score = scores[row_idx]
        if score is None:
            continue
        varying_condition = {key: condition[key] for key in varying_keys}
        serialized = _subset_condition_key(condition, varying_keys)
        if serialized not in seen or _is_better_score(score, seen[serialized]["score"]):
            seen[serialized] = {
                "option": varying_condition,
                "full_conditions": condition,
                "yield": yields[row_idx],
                "score": score,
                "entry_info": _extract_entry_info(table_data, row_idx),
            }
    return list(seen.values())


def build_single_option_bundles(common, table_data, varying_keys, target_key):
    fixed_keys = [key for key in varying_keys if key != target_key]
    full_bundles = _build_distinct_varying_bundles(common, table_data, varying_keys)

    grouped: Dict[str, Dict[str, Any]] = {}
    for bundle in full_bundles:
        full_conditions = bundle["full_conditions"]
        fixed_group_key = _subset_condition_key(full_conditions, fixed_keys)
        target_value = full_conditions.get(target_key, [])
        target_value_key = _normalize_for_comparison(target_value)

        if fixed_group_key not in grouped:
            grouped[fixed_group_key] = {
                "fixed_conditions": _clone_condition_dict(full_conditions, fixed_keys),
                "option_map": {},
            }

        existing = grouped[fixed_group_key]["option_map"].get(target_value_key)
        if existing is None or _is_better_score(bundle["score"], existing["score"]):
            grouped[fixed_group_key]["option_map"][target_value_key] = {
                "option": {target_key: _clone_condition_items(target_value)},
                "full_conditions": full_conditions,
                "yield": bundle["yield"],
                "score": bundle["score"],
                "entry_info": bundle["entry_info"],
            }

    candidate_groups = []
    for fixed_group_key, group in grouped.items():
        option_bundles = sorted(
            group["option_map"].values(),
            key=lambda bundle: _normalize_for_comparison(bundle["option"].get(target_key, [])),
        )
        if len(option_bundles) < 2:
            continue
        best_score = _best_bundle_score(option_bundles)
        candidate_groups.append({
            "group_key": fixed_group_key,
            "fixed_conditions": group["fixed_conditions"],
            "option_bundles": option_bundles,
            "coverage": len(option_bundles),
            "best_score": best_score,
            "best_effective_score": compute_effective_product_score(best_score),
            "mean_effective_score": _mean_effective_score(option_bundles),
            "sort_key": _condition_to_key(group["fixed_conditions"]),
        })

    if not candidate_groups:
        return {
            "fixed_conditions": {},
            "option_bundles": [],
            "local_best_score": (0.0, -1.0, -1.0, -1.0, -1.0),
            "reference_candidates": [],
            "selection_info": {
                "strategy": "max_target_value_coverage",
                "target_key": target_key,
                "candidate_fixed_condition_groups": len(grouped),
                "selected_coverage": 0,
                "reason": "not_enough_real_options",
            },
        }

    selected_group = max(
        candidate_groups,
        key=lambda group: (
            group["coverage"],
            group["best_effective_score"],
            group["mean_effective_score"],
            group["sort_key"],
        ),
    )
    selected_option_keys = {
        _subset_condition_key(bundle["full_conditions"], varying_keys)
        for bundle in selected_group["option_bundles"]
    }
    local_best_effective = selected_group["best_effective_score"]

    reference_candidates = []
    excluded_overlap_count = 0
    excluded_above_local_best_count = 0
    for bundle in full_bundles:
        combo_key = _subset_condition_key(bundle["full_conditions"], varying_keys)
        if combo_key in selected_option_keys:
            excluded_overlap_count += 1
            continue
        if compute_effective_product_score(bundle["score"]) > local_best_effective + 1e-9:
            excluded_above_local_best_count += 1
            continue
        reference_candidates.append(bundle)

    return {
        "fixed_conditions": selected_group["fixed_conditions"],
        "option_bundles": selected_group["option_bundles"],
        "local_best_score": selected_group["best_score"],
        "reference_candidates": reference_candidates,
        "selection_info": {
            "strategy": "max_target_value_coverage",
            "target_key": target_key,
            "candidate_fixed_condition_groups": len(grouped),
            "selected_coverage": selected_group["coverage"],
            "selected_best_effective_score": round(local_best_effective, 6),
            "selected_mean_effective_score": round(selected_group["mean_effective_score"], 6),
            "reference_candidate_count": len(reference_candidates),
            "excluded_reference_overlap_count": excluded_overlap_count,
            "excluded_reference_above_local_best_count": excluded_above_local_best_count,
            "reason": "ok",
        },
    }


def build_all_option_bundles(common, table_data, varying_keys):
    all_conditions = common["all_conditions"]
    yields = common["yields"]
    scores = common["scores"]

    seen = {}
    for row_idx, condition in enumerate(all_conditions):
        score = scores[row_idx]
        if score is None:
            continue
        varying_condition = {key: condition[key] for key in varying_keys}
        serialized = json.dumps(
            {key: _normalize_for_comparison(condition[key]) for key in varying_keys},
            sort_keys=True,
            ensure_ascii=False,
        )
        if serialized not in seen or _is_better_score(score, seen[serialized]["score"]):
            seen[serialized] = {
                "option": varying_condition,
                "full_conditions": condition,
                "yield": yields[row_idx],
                "score": score,
                "entry_info": _extract_entry_info(table_data, row_idx),
            }

    bundles = list(seen.values())
    if BALANCE_ALL_VARYING_OPTIONS and len(bundles) >= 2:
        full_condition_map = {
            json.dumps(bundle["option"], sort_keys=True, ensure_ascii=False): bundle["full_conditions"]
            for bundle in bundles
        }
        option_list = [bundle["option"] for bundle in bundles]
        yield_list = [bundle["yield"] for bundle in bundles]
        score_list = [bundle["score"] for bundle in bundles]
        entry_info_list = [bundle["entry_info"] for bundle in bundles]
        option_list, yield_list, score_list, entry_info_list, balance_info = _balance_all_varying_options(
            option_list,
            yield_list,
            score_list,
            entry_info_list,
            varying_keys,
        )
        bundles = [
            {
                "option": option,
                "full_conditions": full_condition_map.get(
                    json.dumps(option, sort_keys=True, ensure_ascii=False),
                    option,
                ),
                "yield": option_yield,
                "score": score,
                "entry_info": entry_info,
            }
            for option, option_yield, score, entry_info in zip(option_list, yield_list, score_list, entry_info_list)
        ]
    else:
        balance_info = {"enabled": False}
    return bundles, balance_info


def prune_option_bundles(option_bundles: List[dict], visible_keys: List[str]) -> List[dict]:
    pruned = []
    for bundle in option_bundles:
        item = dict(bundle)
        item["option"] = project_condition_dict(bundle["option"], visible_keys)
        item["full_conditions"] = project_condition_dict(bundle["full_conditions"], visible_keys)
        pruned.append(item)
    return pruned


def prepare_condition_view(
    option_bundles: List[dict],
    candidate_keys: List[str],
    varying_keys: List[str],
    constant_conditions: dict,
    fixed_conditions: Optional[dict] = None,
):
    visible_keys = select_active_condition_keys(
        [bundle["full_conditions"] for bundle in option_bundles],
        candidate_keys,
    )
    visible_varying_keys = [key for key in varying_keys if key in visible_keys]
    prepared = {
        "visible_keys": visible_keys,
        "visible_varying_keys": visible_varying_keys,
        "constant_conditions": project_condition_dict(constant_conditions, visible_keys),
        "option_bundles": prune_option_bundles(option_bundles, visible_keys),
    }
    if fixed_conditions is not None:
        prepared["fixed_conditions"] = project_condition_dict(fixed_conditions, visible_keys)
    return prepared


def count_effectively_varying_keys(option_bundles: List[dict], keys: List[str]) -> int:
    count = 0
    for key in keys:
        values = set()
        for bundle in option_bundles:
            val = bundle["option"].get(key, [])
            values.add(_normalize_for_comparison(val))
        if len(values) > 1:
            count += 1
    return count


def has_score_variation(scores: Sequence[Tuple[float, ...]]) -> bool:
    """Return True when there is at least one pair of scores with different effective values.

    Used to discard questions where every option yields the same effective score
    (model picks anything and it is always correct -> no evaluation signal).
    """
    if len(scores) < 2:
        return False
    effs = [compute_effective_product_score(s) for s in scores]
    best = max(effs)
    worst = min(effs)
    return not math.isclose(best, worst, rel_tol=0.0, abs_tol=1e-9)


def get_top_answer_indices(scores: Sequence[Tuple[float, ...]]) -> List[int]:
    if not scores:
        return []
    best_effective = max(compute_effective_product_score(score) for score in scores)
    return [
        idx
        for idx, score in enumerate(scores)
        if math.isclose(
            compute_effective_product_score(score),
            best_effective,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ]


def compute_relative_score(score: Tuple[float, ...], best_score: Tuple[float, ...]) -> float:
    if _score_isclose(score, best_score):
        return 1.0

    best_effective = compute_effective_product_score(best_score)
    current_effective = compute_effective_product_score(score)
    if best_effective > 0:
        base = max(0.0, min(current_effective / best_effective, 1.0))
    else:
        base = 1.0 if math.isclose(current_effective, best_effective, rel_tol=0.0, abs_tol=1e-9) else 0.0
    return round(base, 6)


def compute_relative_scores(scores: Sequence[Tuple[float, ...]]) -> List[float]:
    if not scores:
        return []
    best_score = max(scores, key=lambda score: (compute_effective_product_score(score), score))
    return [compute_relative_score(score, best_score) for score in scores]


def _spread_positions(n_items: int, n_pick: int) -> List[int]:
    if n_pick <= 0 or n_items <= 0:
        return []
    if n_pick >= n_items:
        return list(range(n_items))
    if n_pick == 1:
        return [0]
    positions = []
    for idx in range(n_pick):
        position = round(idx * (n_items - 1) / (n_pick - 1))
        if not positions or position != positions[-1]:
            positions.append(position)
    while len(positions) < n_pick:
        for position in range(n_items):
            if position not in positions:
                positions.append(position)
                if len(positions) == n_pick:
                    break
    return sorted(positions)


def _bundle_sort_key(bundle: Dict[str, Any]) -> str:
    option = bundle.get("option")
    if option is None:
        option = bundle.get("full_conditions", {})
    return json.dumps(option, sort_keys=True, ensure_ascii=False)


def _effective_option_count(total_options: int, top_score_option_count: int) -> int:
    if total_options <= 0:
        return 0
    collapsed_top = max(1, top_score_option_count)
    return total_options - max(0, collapsed_top - 1)


def select_reference_bundles(
    bundles: Sequence[Dict[str, Any]],
    min_references: int = 2,
    max_references: int = 5,
    min_remaining_effective_options: int = 5,
    low_mid_fraction: float = 2.0 / 3.0,
) -> Dict[str, Any]:
    total = len(bundles)
    info = {
        "enabled": True,
        "total_options": total,
        "effective_total_options": total,
        "min_references": min_references,
        "max_references": max_references,
        "target_reference_count": 0,
        "selected_count": 0,
        "remaining_count": total,
        "remaining_effective_count": total,
        "reason": "",
        "top_score_option_count": 0,
        "eligible_score_groups": 0,
        "candidate_score_groups": 0,
        "eligible_bundle_count": 0,
    }

    score_groups: Dict[float, List[int]] = {}
    for idx, bundle in enumerate(bundles):
        score_groups.setdefault(_score_group_key(bundle["score"]), []).append(idx)

    ordered_scores = sorted(score_groups.keys())
    if ordered_scores:
        top_group = score_groups[ordered_scores[-1]]
        info["top_score_option_count"] = len(top_group)
        info["effective_total_options"] = _effective_option_count(total, len(top_group))
        info["remaining_effective_count"] = info["effective_total_options"]
    else:
        top_group = []

    if len(ordered_scores) <= 1:
        info["reason"] = "no_non_top_score_group"
        return {"references": [], "remaining": list(bundles), "info": info}

    info["target_reference_count"] = min(
        max_references,
        max(0, info["effective_total_options"] - min_remaining_effective_options),
    )
    if info["target_reference_count"] < min_references:
        info["reason"] = "not_enough_effective_options"
        return {"references": [], "remaining": list(bundles), "info": info}

    candidate_scores = ordered_scores[:-1]
    info["candidate_score_groups"] = len(candidate_scores)
    eligible_count = max(1, math.ceil(len(candidate_scores) * low_mid_fraction))
    eligible_scores = candidate_scores[:eligible_count]
    info["eligible_score_groups"] = len(eligible_scores)

    eligible_groups: List[List[int]] = []
    for score in eligible_scores:
        grouped = sorted(score_groups[score], key=lambda idx: _bundle_sort_key(bundles[idx]))
        eligible_groups.append(grouped)

    eligible_indices = [idx for group in eligible_groups for idx in group]
    info["eligible_bundle_count"] = len(eligible_indices)
    if not eligible_indices:
        info["reason"] = "no_reference_candidate"
        return {"references": [], "remaining": list(bundles), "info": info}

    n_pick = min(info["target_reference_count"], len(eligible_indices))
    if n_pick < min_references:
        info["reason"] = "not_enough_reference_candidates"
        return {"references": [], "remaining": list(bundles), "info": info}

    representative_indices = [group[0] for group in eligible_groups]
    if n_pick <= len(representative_indices):
        chosen_positions = _spread_positions(len(representative_indices), n_pick)
        chosen_indices = [representative_indices[pos] for pos in chosen_positions]
    else:
        chosen_indices = list(representative_indices)
        extra_candidates = [idx for group in eligible_groups for idx in group[1:]]
        extra_needed = n_pick - len(chosen_indices)
        extra_positions = _spread_positions(len(extra_candidates), extra_needed)
        chosen_indices.extend(extra_candidates[pos] for pos in extra_positions)

    chosen_indices = sorted(
        chosen_indices,
        key=lambda idx: (_score_group_key(bundles[idx]["score"]), _bundle_sort_key(bundles[idx])),
    )
    chosen_set = set(chosen_indices)
    references = [bundles[idx] for idx in chosen_indices]
    remaining = [bundle for idx, bundle in enumerate(bundles) if idx not in chosen_set]
    remaining_effective = _effective_option_count(len(remaining), len(top_group))

    if remaining_effective < min_remaining_effective_options:
        info["reason"] = "remaining_effective_options_below_threshold"
        return {"references": [], "remaining": list(bundles), "info": info}

    info["selected_count"] = len(references)
    info["remaining_count"] = len(remaining)
    info["remaining_effective_count"] = remaining_effective
    info["reason"] = "ok"
    return {"references": references, "remaining": remaining, "info": info}


def select_single_reference_bundles(
    candidate_bundles: Sequence[Dict[str, Any]],
    ceiling_score,
    min_references: int = 2,
    max_references: int = 5,
    low_mid_fraction: float = 2.0 / 3.0,
) -> Dict[str, Any]:
    info = {
        "enabled": True,
        "strategy": "external_real_combinations_below_local_best",
        "min_references": min_references,
        "max_references": max_references,
        "total_candidates": len(candidate_bundles),
        "candidate_score_groups": 0,
        "eligible_score_groups": 0,
        "eligible_bundle_count": 0,
        "selected_count": 0,
        "ceiling_effective_score": round(compute_effective_product_score(ceiling_score), 6),
        "selection_scope": "",
        "reason": "",
    }
    if not candidate_bundles:
        info["reason"] = "no_reference_candidate"
        return {"references": [], "info": info}

    score_groups: Dict[float, List[int]] = {}
    for idx, bundle in enumerate(candidate_bundles):
        score_groups.setdefault(_score_group_key(bundle["score"]), []).append(idx)

    ordered_scores = sorted(score_groups.keys())
    info["candidate_score_groups"] = len(ordered_scores)
    if not ordered_scores:
        info["reason"] = "no_reference_candidate"
        return {"references": [], "info": info}

    eligible_group_count = max(1, math.ceil(len(ordered_scores) * low_mid_fraction))
    eligible_scores = ordered_scores[:eligible_group_count]

    def _flatten_groups(score_values: Sequence[float]) -> List[List[int]]:
        return [
            sorted(score_groups[score], key=lambda idx: _bundle_sort_key(candidate_bundles[idx]))
            for score in score_values
        ]

    eligible_groups = _flatten_groups(eligible_scores)
    eligible_indices = [idx for group in eligible_groups for idx in group]
    info["eligible_score_groups"] = len(eligible_scores)
    info["eligible_bundle_count"] = len(eligible_indices)

    if len(eligible_indices) < min_references:
        eligible_scores = ordered_scores
        eligible_groups = _flatten_groups(eligible_scores)
        eligible_indices = [idx for group in eligible_groups for idx in group]
        info["eligible_score_groups"] = len(eligible_scores)
        info["eligible_bundle_count"] = len(eligible_indices)
        info["selection_scope"] = "all_allowed_scores"
    else:
        info["selection_scope"] = "low_mid_allowed_scores"

    n_pick = min(max_references, len(eligible_indices))
    if n_pick < min_references:
        info["reason"] = "not_enough_reference_candidates"
        return {"references": [], "info": info}

    representative_indices = [group[0] for group in eligible_groups]
    if n_pick <= len(representative_indices):
        chosen_positions = _spread_positions(len(representative_indices), n_pick)
        chosen_indices = [representative_indices[pos] for pos in chosen_positions]
    else:
        chosen_indices = list(representative_indices)
        extra_candidates = [idx for group in eligible_groups for idx in group[1:]]
        extra_needed = n_pick - len(chosen_indices)
        extra_positions = _spread_positions(len(extra_candidates), extra_needed)
        chosen_indices.extend(extra_candidates[pos] for pos in extra_positions)

    chosen_indices = sorted(
        chosen_indices,
        key=lambda idx: (_score_group_key(candidate_bundles[idx]["score"]), _bundle_sort_key(candidate_bundles[idx])),
    )
    references = [candidate_bundles[idx] for idx in chosen_indices]
    info["selected_count"] = len(references)
    info["reason"] = "ok"
    return {"references": references, "info": info}


def build_reference_records(reference_bundles: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for bundle in reference_bundles:
        records.append({
            "conditions": bundle["full_conditions"],
            "outcome": score_to_dict(bundle["score"]),
            "source_entry": bundle["entry_info"],
        })
    return records
