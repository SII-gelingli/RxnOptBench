"""
Reaction condition optimization inference for locally served open-source models.

This script keeps the submitted benchmark task schema, but
switches the backend to a configurable OpenAI-compatible endpoint such as vLLM.
"""

import argparse
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import requests

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


DEFAULT_BASE_URL = "http://localhost:8008/v1"
DEFAULT_CREDENTIAL = "EMPTY"
DEFAULT_MAX_WORKERS = 8
DEFAULT_RETRY_DELAY = 3.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_CHECKPOINT_EVERY = 50
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "original_json"
DEFAULT_PROMPT_DIR = REPO_ROOT / "prompts"
PROMPT_TEMPLATE_FILENAMES = {
    ("mc_single", "classic"): "prompt_single_varying.txt",
    ("mc_single", "reference"): "prompt_single_varying_reference.txt",
    ("mc_single", "reference_control"): "prompt_single_varying.txt",
    ("mc_all", "classic"): "prompt_all_varying.txt",
    ("mc_all", "reference"): "prompt_all_varying_reference.txt",
    ("mc_all", "reference_control"): "prompt_all_varying.txt",
    ("fib_all", "classic"): "prompt_all_blank.txt",
    ("fib_all", "reference"): "prompt_all_blank_reference.txt",
}
DATASET_FILENAMES = {
    ("mc_single", "classic"): "multiple_choice_single_varying.json",
    ("mc_single", "reference"): "multiple_choice_single_varying_reference.json",
    ("mc_single", "reference_control"): "multiple_choice_single_varying_reference_control.json",
    ("mc_all", "classic"): "multiple_choice_all_varying.json",
    ("mc_all", "reference"): "multiple_choice_all_varying_reference.json",
    ("mc_all", "reference_control"): "multiple_choice_all_varying_reference_control.json",
    ("fib_all", "classic"): "fill_in_blank_all_blank.json",
    ("fib_all", "reference"): "fill_in_blank_all_blank_reference.json",
}
TASK_SETTING_TO_QUESTION_TYPE = {
    ("mc_single", "classic"): "single_varying",
    ("mc_single", "reference"): "single_varying_reference",
    ("mc_single", "reference_control"): "single_varying_reference_control",
    ("mc_all", "classic"): "all_varying",
    ("mc_all", "reference"): "all_varying_reference",
    ("mc_all", "reference_control"): "all_varying_reference_control",
    ("fib_all", "classic"): "all_blank",
    ("fib_all", "reference"): "all_blank_reference",
}
QUESTION_TYPE_TO_TASK_SETTING = {
    question_type: pair for pair, question_type in TASK_SETTING_TO_QUESTION_TYPE.items()
}

client = None
MODEL_NAME = None

progress_lock = threading.Lock()
progress_count = 0
progress_total = 0

token_lock = threading.Lock()
total_prompt_tokens = 0
total_completion_tokens = 0
total_total_tokens = 0

parse_lock = threading.Lock()
parse_stats = Counter()


def latest_timestamped_dir(root_dir, suffix):
    candidates = sorted(path for path in root_dir.glob(f"*_{suffix}") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No '*_{suffix}' directory found under {root_dir}")
    return candidates[-1]


def default_prompt_dir_for_task(task_type):
    if task_type not in {"mc_single", "mc_all", "fib_all"}:
        raise ValueError(f"Unknown task_type: {task_type}")
    return DEFAULT_PROMPT_DIR


def default_testset_path(task_type, setting, data_dir=DEFAULT_DATA_DIR):
    effective_setting = "classic" if setting == "auto" else setting
    filename = DATASET_FILENAMES.get((task_type, effective_setting))
    if filename is None:
        raise ValueError(f"No bundled dataset for task={task_type}, setting={effective_setting}")
    return Path(data_dir) / filename


def normalize_base_url(base_url):
    base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def setup_logging(log_dir, model_name, task):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    tag = f"{safe_model}_{task}"
    log_file = os.path.join(log_dir, f"{tag}_{timestamp}.log")

    logger = logging.getLogger(f"rxn_open_source_{tag}_{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    logger.info(f"Log file: {log_file}")
    return logger


def format_species_list(species_list):
    parts = []
    for item in species_list:
        name = item.get("content", "").strip()
        smiles = item.get("SMILES", "").strip()
        if name and smiles:
            parts.append(f"{name} (SMILES: {smiles})")
        elif smiles:
            parts.append(f"SMILES: {smiles}")
        elif name:
            parts.append(name)
    return "\n".join(f"  - {part}" for part in parts) if parts else "  (none)"


def format_condition_dict(cond_dict):
    lines = []
    for key, items in cond_dict.items():
        if items:
            parts = []
            for item in items:
                name = item.get("content", "").strip()
                smiles = item.get("SMILES", "").strip()
                if name and smiles:
                    parts.append(f"{name} (SMILES: {smiles})")
                elif name:
                    parts.append(name)
                elif smiles:
                    parts.append(f"SMILES: {smiles}")
            lines.append(f"  - {key}: {', '.join(parts)}" if parts else f"  - {key}: (none)")
        else:
            lines.append(f"  - {key}: (none)")
    return "\n".join(lines) if lines else "  (none)"


def format_mc_options(options):
    blocks = []
    for index, option in enumerate(options):
        lines = [f"**Option {index}:**"]
        for key, items in option.items():
            if items:
                parts = []
                for item in items:
                    name = item.get("content", "").strip()
                    smiles = item.get("SMILES", "").strip()
                    if name and smiles:
                        parts.append(f"{name} (SMILES: {smiles})")
                    elif name:
                        parts.append(name)
                    elif smiles:
                        parts.append(f"SMILES: {smiles}")
                lines.append(f"  - {key}: {', '.join(parts)}" if parts else f"  - {key}: (none)")
            else:
                lines.append(f"  - {key}: (none)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_fib_candidates(candidate_values):
    blocks = []
    for key, candidates in candidate_values.items():
        lines = [f"**{key}:**"]
        for candidate in candidates:
            idx = candidate["index"]
            values = candidate.get("value", [])
            if values:
                parts = []
                for value in values:
                    name = value.get("content", "").strip()
                    smiles = value.get("SMILES", "").strip()
                    if name and smiles:
                        parts.append(f"{name} (SMILES: {smiles})")
                    elif name:
                        parts.append(name)
                    elif smiles:
                        parts.append(f"SMILES: {smiles}")
                lines.append(f"  {idx}: {', '.join(parts)}" if parts else f"  {idx}: (none)")
            else:
                lines.append(f"  {idx}: (none)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@lru_cache(maxsize=None)
def resolve_prompt_template_path(task_type, setting, dataset_path_str, prompt_dir_str=None):
    filename = PROMPT_TEMPLATE_FILENAMES[(task_type, setting)]

    candidate_dirs = [
        Path(prompt_dir_str).expanduser().resolve()
        if prompt_dir_str
        else default_prompt_dir_for_task(task_type)
    ]
    if dataset_path_str:
        dataset_dir = Path(dataset_path_str).resolve().parent
        candidate_dirs.append(dataset_dir)

    seen = set()
    for candidate_dir in candidate_dirs:
        if candidate_dir in seen:
            continue
        seen.add(candidate_dir)
        template_path = candidate_dir / filename
        if template_path.exists():
            return template_path

    searched = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(
        f"Cannot find prompt template '{filename}' for task={task_type}, setting={setting}. "
        f"Searched in: {searched}"
    )


@lru_cache(maxsize=None)
def load_prompt_template(task_type, setting, dataset_path_str, prompt_dir_str=None):
    return resolve_prompt_template_path(
        task_type,
        setting,
        dataset_path_str,
        prompt_dir_str,
    ).read_text(encoding="utf-8")


def format_reference_conditions(reference_items):
    blocks = []
    for index, ref in enumerate(reference_items, start=1):
        outcome = ref.get("outcome", {})
        score_parts = []
        if outcome.get("yield") is not None:
            score_parts.append(f"yield={outcome['yield']}%")
        if outcome.get("ee") is not None:
            score_parts.append(f"ee={outcome['ee']}")
        if outcome.get("dr") is not None:
            score_parts.append(f"dr={outcome['dr']}")
        if outcome.get("er") is not None:
            score_parts.append(f"er={outcome['er']}")
        if outcome.get("rr") is not None:
            score_parts.append(f"rr={outcome['rr']}")

        header = f"*Reference {index} ({', '.join(score_parts)}):*" if score_parts else f"*Reference {index}:*"
        blocks.append("\n".join([header, format_condition_dict(ref.get("conditions", {}))]))
    return "\n\n".join(blocks) if blocks else "  (none)"


def task_label(task_type, setting):
    return task_type if setting == "classic" else f"{task_type}_{setting}"


def resolve_task_and_setting(task_type, requested_setting, dataset, logger):
    sample_question_types = {
        sample.get("question_type")
        for sample in dataset
        if isinstance(sample, dict) and sample.get("question_type")
    }

    if not sample_question_types:
        if requested_setting == "auto":
            logger.warning("Dataset has no question_type field; defaulting setting to classic.")
            return "classic"
        return requested_setting

    if len(sample_question_types) != 1:
        raise ValueError(f"Dataset contains mixed question_type values: {sorted(sample_question_types)}")

    question_type = next(iter(sample_question_types))
    inferred = QUESTION_TYPE_TO_TASK_SETTING.get(question_type)
    if inferred is None:
        raise ValueError(f"Unsupported question_type in dataset: {question_type}")

    inferred_task_type, inferred_setting = inferred
    if inferred_task_type != task_type:
        raise ValueError(
            f"Dataset question_type '{question_type}' maps to task '{inferred_task_type}', "
            f"but requested task is '{task_type}'."
        )

    if requested_setting == "auto":
        return inferred_setting

    if requested_setting != inferred_setting:
        raise ValueError(
            f"Requested setting '{requested_setting}' does not match dataset question_type "
            f"'{question_type}' (expected setting '{inferred_setting}')."
        )

    return requested_setting


def reasoning_instruction(reasoning_mode):
    if reasoning_mode == "required":
        return """

### Reasoning Requirement

Think carefully about the reaction before answering. Your final answer must still follow the required output format exactly, with no extra wrapper tags or commentary."""
    return ""


def mc_answer_instruction(reasoning_mode):
    return "Provide ONLY the option index (0-based integer)."


def fib_answer_instruction(reasoning_mode):
    return "Provide ONLY a JSON object in the following format:"


def build_mc_single_prompt(sample, reasoning_mode, setting, dataset_path, prompt_dir=None):
    inp = sample["input"]
    reactants_str = format_species_list(inp["reactants"])
    products_str = format_species_list(inp["products"])
    target_key = sample["target_key"]
    constant_str = format_condition_dict(inp.get("constant_conditions", {}))
    fixed_str = format_condition_dict(inp.get("fixed_conditions", {}))
    options_str = format_mc_options(sample["options"])
    references_str = format_reference_conditions(inp.get("reference_conditions", []))
    if reasoning_mode == "none":
        template_kwargs = {
            "target_key": target_key,
            "reactants": reactants_str,
            "products": products_str,
            "constant_conditions": constant_str,
            "fixed_conditions": fixed_str,
            "references": references_str,
            "options": options_str,
        }
        return load_prompt_template(
            "mc_single", setting, dataset_path, prompt_dir
        ).format(**template_kwargs)

    answer_instruction = mc_answer_instruction(reasoning_mode)
    if setting == "reference":
        return f"""You are an expert organic chemist. Given the following reaction, one real fixed condition group from the source table, and several reference condition sets with observed outcomes, choose the best value for **{target_key}** from the remaining candidate options.

### Reaction

**Reactants:**
{reactants_str}

**Target Product:**
{products_str}

### Constant Conditions (do not change)

{constant_str}

### Fixed Conditions (from source table)

{fixed_str}

### Reference Conditions And Outcomes

{references_str}

### Candidate Values For {target_key}

{options_str}
{reasoning_instruction(reasoning_mode)}

{answer_instruction} For example:

0"""
    return f"""You are an expert organic chemist. Given the following chemical reaction and one real fixed condition group taken from the source table, choose the best value for **{target_key}** from the provided options.

### Reaction

**Reactants:**
{reactants_str}

**Target Product:**
{products_str}

### Constant Conditions (do not change)

{constant_str}

### Fixed Conditions (from source table)

{fixed_str}

### Candidate Values For {target_key}

{options_str}
{reasoning_instruction(reasoning_mode)}

{answer_instruction} For example:

0"""


def build_mc_all_prompt(sample, reasoning_mode, setting, dataset_path, prompt_dir=None):
    inp = sample["input"]
    reactants_str = format_species_list(inp["reactants"])
    products_str = format_species_list(inp["products"])
    constant_str = format_condition_dict(inp.get("constant_conditions", {}))
    options_str = format_mc_options(sample["options"])
    references_str = format_reference_conditions(inp.get("reference_conditions", []))
    if reasoning_mode == "none":
        template_kwargs = {
            "reactants": reactants_str,
            "products": products_str,
            "constant_conditions": constant_str,
            "references": references_str,
            "options": options_str,
        }
        return load_prompt_template(
            "mc_all", setting, dataset_path, prompt_dir
        ).format(**template_kwargs)

    answer_instruction = mc_answer_instruction(reasoning_mode)
    if setting == "reference":
        return f"""You are an expert organic chemist. Given the following reaction and several reference condition sets with observed outcomes, choose the best remaining reaction-condition option.

### Reaction

**Reactants:**
{reactants_str}

**Target Product:**
{products_str}

### Constant Conditions (do not change)

{constant_str}

### Reference Conditions And Outcomes

{references_str}

### Candidate Conditions

{options_str}
{reasoning_instruction(reasoning_mode)}

{answer_instruction} For example:

0"""
    return f"""You are an expert organic chemist. Given the following chemical reaction, choose the best reaction-condition option from the provided candidates.

### Reaction

**Reactants:**
{reactants_str}

**Target Product:**
{products_str}

### Constant Conditions (do not change)

{constant_str}

### Candidate Conditions

{options_str}
{reasoning_instruction(reasoning_mode)}

{answer_instruction} For example:

0"""


def build_fib_all_prompt(sample, reasoning_mode, setting, dataset_path, prompt_dir=None):
    inp = sample["input"]
    reactants_str = format_species_list(inp["reactants"])
    products_str = format_species_list(inp["products"])
    constant_str = format_condition_dict(inp.get("constant_conditions", {}))
    candidates_str = format_fib_candidates(inp["candidate_values"])
    query_keys = sample["query_keys"]
    query_keys_json = ",\n".join(f'  "{key}": <candidate_index>' for key in query_keys)
    references_str = format_reference_conditions(inp.get("reference_conditions", []))
    if reasoning_mode == "none":
        template_kwargs = {
            "reactants": reactants_str,
            "products": products_str,
            "constant_conditions": constant_str,
            "references": references_str,
            "candidate_values": candidates_str,
            "query_keys_json": query_keys_json,
        }
        return load_prompt_template(
            "fib_all", setting, dataset_path, prompt_dir
        ).format(**template_kwargs)

    answer_instruction = fib_answer_instruction(reasoning_mode)
    if setting == "reference":
        return f"""You are an expert organic chemist. Given the following reaction, several reference condition sets with observed outcomes, and a candidate index table for each condition category, choose one candidate index for each required condition category.

### Reaction

**Reactants:**
{reactants_str}

**Target Product:**
{products_str}

### Constant Conditions (do not change)

{constant_str}

### Reference Conditions And Outcomes

{references_str}

Important: the reference condition sets shown above are NOT valid answers for this question. Do NOT copy any reference combination directly. You must propose a NEW combination by selecting candidate indices from the table below. Only combinations formed from the candidate index table are considered valid outputs.

### Candidate Values

{candidates_str}
{reasoning_instruction(reasoning_mode)}

{answer_instruction}

{{
{query_keys_json}
}}

Each value must be a single integer candidate index, not a condition name."""
    return f"""You are an expert organic chemist. Given the following reaction, choose one candidate index for each required condition category.

### Reaction

**Reactants:**
{reactants_str}

**Target Product:**
{products_str}

### Constant Conditions (do not change)

{constant_str}

### Candidate Values

{candidates_str}
{reasoning_instruction(reasoning_mode)}

{answer_instruction}

{{
{query_keys_json}
}}

Each value must be a single integer candidate index, not a condition name."""


def extract_answer_tag(text):
    if not text:
        return text
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def extract_reasoning(text):
    if not text:
        return ""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer_candidate(text):
    if not text:
        return ""

    answer_text = extract_answer_tag(text).strip()
    for closing_tag in ("</think>", "</reasoning>"):
        if closing_tag in answer_text:
            answer_text = answer_text.rsplit(closing_tag, 1)[-1].strip()

    if "```" in answer_text:
        lines = answer_text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        answer_text = "\n".join(lines).strip()

    return answer_text


def parse_reasonable_int(number_str, max_digits=12):
    if not number_str or len(number_str) > max_digits:
        return None
    try:
        return int(number_str)
    except ValueError:
        return None


def build_mc_structured_outputs(n_options):
    choices = "|".join(str(index) for index in range(n_options))
    return {"regex": f"(?:{choices})"}


def build_fib_structured_outputs(sample):
    candidate_values = sample["input"]["candidate_values"]
    properties = {}
    required = []

    for key in sample["query_keys"]:
        n_candidates = len(candidate_values.get(key, []))
        property_schema = {
            "type": "integer",
            "minimum": 0,
        }
        if n_candidates > 0:
            property_schema["maximum"] = n_candidates - 1
        else:
            property_schema["maximum"] = 0
        properties[key] = property_schema
        required.append(key)

    return {
        "json": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "disable_additional_properties": True,
    }


def resolve_chat_template_thinking_key(requested_key, model_name):
    if requested_key and requested_key != "auto":
        return requested_key

    model_name = (model_name or "").lower()
    if "kimi" in model_name:
        return "thinking"
    return "enable_thinking"


def build_request_extra_body(sample, task_type, reasoning_mode, chat_template_thinking_key):
    thinking_key = resolve_chat_template_thinking_key(
        chat_template_thinking_key,
        MODEL_NAME,
    )
    extra_body = {
        "chat_template_kwargs": {
            thinking_key: reasoning_mode == "required",
        }
    }

    if task_type.startswith("mc_"):
        extra_body["structured_outputs"] = build_mc_structured_outputs(len(sample["options"]))
    else:
        extra_body["structured_outputs"] = build_fib_structured_outputs(sample)

    return extra_body


def parse_mc_response(response_text, n_options, logger):
    if not response_text:
        with parse_lock:
            parse_stats["failed_empty"] += 1
        return 0, "failed_empty"

    text = extract_answer_candidate(response_text)

    try:
        idx = int(text)
        if 0 <= idx < n_options:
            with parse_lock:
                parse_stats["success_direct"] += 1
            return idx, "success_direct"
        logger.warning(f"Option {idx} out of range [0,{n_options}), clamping")
        with parse_lock:
            parse_stats["success_clamped"] += 1
        return max(0, min(idx, n_options - 1)), "success_clamped"
    except ValueError:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            idx = int(line)
            if 0 <= idx < n_options:
                with parse_lock:
                    parse_stats["fallback_last_line_int"] += 1
                return idx, "fallback_last_line_int"
        except ValueError:
            pass

    numbers = re.findall(r"\b(\d+)\b", text)
    for number_str in reversed(numbers):
        idx = parse_reasonable_int(number_str)
        if idx is None:
            continue
        if 0 <= idx < n_options:
            with parse_lock:
                parse_stats["fallback_last_regex"] += 1
            return idx, "fallback_last_regex"

    for number_str in reversed(numbers):
        idx = parse_reasonable_int(number_str)
        if idx is None:
            continue
        clamped = max(0, min(idx, n_options - 1))
        logger.warning(f"Using last number {idx} (clamped to {clamped}): {text[:200]}")
        with parse_lock:
            parse_stats["fallback_last_int_clamped"] += 1
        return clamped, "fallback_last_int_clamped"

    logger.warning(f"Cannot parse MC response, defaulting to 0: {text[:300]}")
    with parse_lock:
        parse_stats["failed_no_int"] += 1
    return 0, "failed_no_int"


def parse_fib_response(response_text, query_keys, candidate_values, logger):
    default = {key: 0 for key in query_keys}

    if not response_text:
        with parse_lock:
            parse_stats["failed_empty"] += 1
        return default, "failed_empty"

    text = extract_answer_candidate(response_text)

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            parsed = {}
            valid = True
            for key in query_keys:
                value = result.get(key)
                n_candidates = len(candidate_values.get(key, []))
                if isinstance(value, int) and 0 <= value < n_candidates:
                    parsed[key] = value
                elif isinstance(value, (int, float)):
                    parsed[key] = max(0, min(int(value), n_candidates - 1))
                    valid = False
                else:
                    parsed[key] = 0
                    valid = False
            method = "success_direct" if valid else "success_partial"
            with parse_lock:
                parse_stats[method] += 1
            return parsed, method
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, dict):
                parsed = {}
                for key in query_keys:
                    value = result.get(key)
                    n_candidates = len(candidate_values.get(key, []))
                    if isinstance(value, int) and 0 <= value < n_candidates:
                        parsed[key] = value
                    elif isinstance(value, (int, float)):
                        parsed[key] = max(0, min(int(value), n_candidates - 1))
                    else:
                        parsed[key] = 0
                with parse_lock:
                    parse_stats["fallback_regex_json"] += 1
                return parsed, "fallback_regex_json"
        except json.JSONDecodeError:
            pass

    parsed = {}
    any_found = False
    for key in query_keys:
        n_candidates = len(candidate_values.get(key, []))
        pattern = rf'"{re.escape(key)}"\s*:\s*(\d+)'
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            parsed[key] = max(0, min(value, n_candidates - 1))
            any_found = True
        else:
            parsed[key] = 0

    if any_found:
        with parse_lock:
            parse_stats["fallback_per_key_regex"] += 1
        return parsed, "fallback_per_key_regex"

    logger.warning(f"All FIB parsing failed, defaulting to 0s: {text[:300]}")
    with parse_lock:
        parse_stats["failed_no_parse"] += 1
    return default, "failed_no_parse"


def load_json_if_exists(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def atomic_dump(path, obj):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def sidecar_path(output_path, label):
    path = Path(output_path)
    if path.suffix.lower() == ".json":
        filename = f"{path.stem}_{label}.json"
    else:
        filename = f"{path.name}_{label}.json"
    return str(path.with_name(filename))


def ordered_mapping(dataset, mapping):
    return {sample["id"]: mapping[sample["id"]] for sample in dataset if sample["id"] in mapping}


def auth_headers(api_key):
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def fetch_served_model_ids(base_url, api_key, request_timeout):
    response = requests.get(
        f"{base_url}/models",
        headers=auth_headers(api_key),
        timeout=(10, request_timeout),
    )
    response.raise_for_status()
    payload = response.json()
    return [item["id"] for item in payload.get("data", []) if "id" in item]


def wait_for_server_ready(base_url, api_key, timeout_seconds, poll_interval, logger):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            model_ids = fetch_served_model_ids(base_url, api_key, request_timeout=30)
            if model_ids:
                logger.info(f"Server is ready at {base_url}; models: {model_ids}")
                return model_ids
            last_error = "server returned an empty model list"
        except Exception as exc:  # requests exceptions vary by transport layer
            last_error = str(exc)
        logger.info(f"Waiting for server at {base_url} ...")
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for {base_url}. Last error: {last_error}"
    )


def resolve_model_name(requested_model, served_model_ids, logger):
    if served_model_ids:
        if requested_model == "auto":
            resolved = served_model_ids[0]
            logger.info(f"Model resolved automatically to: {resolved}")
            return resolved
        if requested_model in served_model_ids:
            return requested_model
        if len(served_model_ids) == 1:
            resolved = served_model_ids[0]
            logger.warning(
                f"Requested model '{requested_model}' not found in served list {served_model_ids}; "
                f"falling back to '{resolved}'."
            )
            return resolved
        raise ValueError(
            f"Requested model '{requested_model}' not found in served list: {served_model_ids}"
        )

    if requested_model == "auto":
        raise ValueError("Model is set to 'auto', but the server model list is unavailable.")
    logger.warning("Server model list unavailable; using the requested model name directly.")
    return requested_model


def call_llm(prompt, sample, task_type, logger, args):
    global total_prompt_tokens, total_completion_tokens, total_total_tokens

    if getattr(args, "disable_vllm_extensions", False):
        extra_body = None
    else:
        extra_body = build_request_extra_body(
            sample,
            task_type,
            args.reasoning_mode,
            args.chat_template_thinking_key,
        )
    request_kwargs = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.request_timeout,
    }
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body
    if args.top_p is not None:
        request_kwargs["top_p"] = args.top_p

    for attempt in range(args.max_retries):
        try:
            response = client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message
            result = (message.content or "").strip()
            reasoning = (getattr(message, "reasoning", None) or "").strip()
            usage = response.usage
            if usage:
                with token_lock:
                    total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                    total_total_tokens += getattr(usage, "total_tokens", 0) or 0
            logger.debug(
                f"API response content ({len(result)} chars): {result[:500]}"
            )
            if reasoning:
                logger.debug(
                    f"API response reasoning ({len(reasoning)} chars): {reasoning[:500]}"
                )
            return result, reasoning
        except Exception as exc:
            logger.warning(f"API error (attempt {attempt + 1}/{args.max_retries}): {exc}")
            if attempt < args.max_retries - 1:
                wait_seconds = args.retry_delay * (2 ** attempt)
                logger.info(f"  Retrying in {wait_seconds:.1f}s ...")
                time.sleep(wait_seconds)
    return None, ""


def process_sample(args_tuple):
    sample, task_type, reasoning_mode, logger, args = args_tuple
    global progress_count, progress_total

    sample_id = sample["id"]

    if task_type == "mc_single":
        prompt = build_mc_single_prompt(
            sample, reasoning_mode, args.setting, args.testset, args.prompt_dir
        )
    elif task_type == "mc_all":
        prompt = build_mc_all_prompt(
            sample, reasoning_mode, args.setting, args.testset, args.prompt_dir
        )
    elif task_type == "fib_all":
        prompt = build_fib_all_prompt(
            sample, reasoning_mode, args.setting, args.testset, args.prompt_dir
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    response, reasoning = call_llm(prompt, sample, task_type, logger, args)
    if not reasoning and response:
        reasoning = extract_reasoning(response)

    if task_type.startswith("mc_"):
        n_options = len(sample["options"])
        result, parse_method = parse_mc_response(response, n_options, logger)
    else:
        query_keys = sample["query_keys"]
        candidate_values = sample["input"]["candidate_values"]
        result, parse_method = parse_fib_response(response, query_keys, candidate_values, logger)

    with progress_lock:
        progress_count += 1
        if progress_count % 50 == 0 or progress_count == progress_total:
            logger.info(f"  [{progress_count}/{progress_total}] {sample_id}: {result} ({parse_method})")

    return sample_id, result, response, reasoning, parse_method


def main():
    global client, MODEL_NAME
    global progress_count, progress_total
    global total_prompt_tokens, total_completion_tokens, total_total_tokens

    parser = argparse.ArgumentParser(
        description="Open-source inference engine for reaction condition optimization"
    )
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--task-type", choices=["mc_single", "mc_all", "fib_all"], required=True)
    parser.add_argument(
        "--setting",
        choices=["auto", "classic", "reference", "reference_control"],
        default=os.environ.get("SETTING", "auto"),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("RXNOPTBENCH_DATA_DIR", str(DEFAULT_DATA_DIR)),
        help="Directory containing the bundled benchmark JSON files.",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default=None,
        help="Dataset JSON path. If omitted, select it from --data-dir using "
        "--task-type and --setting (auto defaults to the classic dataset).",
    )
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default=os.environ.get("RXNOPTBENCH_PROMPT_DIR", str(DEFAULT_PROMPT_DIR)),
        help="Prompt template directory (default: repository-root prompts/).",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", DEFAULT_CREDENTIAL))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", DEFAULT_MAX_WORKERS)))
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "4096")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0")))
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--reasoning-mode", choices=["required", "none"], default=os.environ.get("REASONING_MODE", "required"))
    parser.add_argument(
        "--chat-template-thinking-key",
        default=os.environ.get("CHAT_TEMPLATE_THINKING_KEY", "auto"),
        help="Chat template kwarg name used to toggle thinking, e.g. enable_thinking or thinking",
    )
    parser.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("CHECKPOINT_EVERY", str(DEFAULT_CHECKPOINT_EVERY))))
    parser.add_argument(
        "--disable-vllm-extensions",
        action="store_true",
        help="Do not send extra_body (structured_outputs / chat_template_kwargs). "
        "Use for non-vLLM OpenAI-compatible cloud APIs.",
    )
    parser.add_argument("--wait-for-server", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=int(os.environ.get("SERVER_TIMEOUT", "1800")))
    parser.add_argument("--server-poll-interval", type=int, default=int(os.environ.get("SERVER_POLL_INTERVAL", "10")))
    args = parser.parse_args()

    if args.testset is None:
        try:
            args.testset = str(
                default_testset_path(
                    args.task_type,
                    args.setting,
                    Path(args.data_dir).expanduser(),
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
    args.testset = str(Path(args.testset).expanduser().resolve())
    args.prompt_dir = str(Path(args.prompt_dir).expanduser().resolve())
    args.base_url = normalize_base_url(args.base_url)

    if args.log_dir is None:
        args.log_dir = os.path.join(os.path.dirname(args.output) or ".", "..", "logs_open_source")

    requested_model_for_log = args.model if args.model != "auto" else "auto_model"

    with open(args.testset, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    args.setting = resolve_task_and_setting(args.task_type, args.setting, dataset, logging.getLogger("bootstrap"))
    log_task_label = task_label(args.task_type, args.setting)
    logger = setup_logging(args.log_dir, requested_model_for_log, log_task_label)

    logger.info("=== Open-Source Inference ===")
    logger.info(f"Requested model: {args.model}")
    logger.info(f"Task: {args.task_type}")
    logger.info(f"Setting: {args.setting}")
    logger.info(f"Testset: {args.testset}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Reasoning mode: {args.reasoning_mode}")
    if getattr(args, "disable_vllm_extensions", False):
        logger.info("Structured outputs: off (no extra_body; --disable-vllm-extensions)")
    else:
        logger.info("Structured outputs: enabled (via extra_body)")

    logger.info(f"Loaded {len(dataset)} samples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    existing_preds = {}
    if args.resume:
        existing_preds = load_json_if_exists(args.output)
        if existing_preds:
            logger.info(f"Resuming: found {len(existing_preds)} existing predictions")

    todo_samples = [sample for sample in dataset if sample["id"] not in existing_preds]
    logger.info(f"Samples to process: {len(todo_samples)} (skipped {len(dataset) - len(todo_samples)})")

    if not todo_samples:
        logger.info("All samples already done. Exiting without contacting the server.")
        return

    served_model_ids = None
    if args.wait_for_server:
        served_model_ids = wait_for_server_ready(
            args.base_url,
            args.api_key,
            timeout_seconds=args.server_timeout,
            poll_interval=args.server_poll_interval,
            logger=logger,
        )
    else:
        try:
            served_model_ids = fetch_served_model_ids(
                args.base_url,
                args.api_key,
                request_timeout=min(args.request_timeout, 30),
            )
        except Exception as exc:
            logger.warning(f"Unable to fetch served model list immediately: {exc}")

    MODEL_NAME = resolve_model_name(args.model, served_model_ids, logger)
    logger.info(f"Resolved model name: {MODEL_NAME}")
    args.chat_template_thinking_key = resolve_chat_template_thinking_key(
        args.chat_template_thinking_key,
        MODEL_NAME,
    )
    if getattr(args, "disable_vllm_extensions", False):
        logger.info(
            "Chat template thinking key (unused with --disable-vllm-extensions): %s",
            args.chat_template_thinking_key,
        )
    else:
        logger.info(f"Chat template thinking key: {args.chat_template_thinking_key}")
    if OpenAI is None:
        raise ModuleNotFoundError(
            "The 'openai' package is required for model inference. "
            "Prompt inspection remains available without it."
        )
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0
    progress_count = 0
    progress_total = len(todo_samples)
    parse_stats.clear()

    predictions = dict(existing_preds)
    raw_output = sidecar_path(args.output, "raw")
    parse_methods_output = sidecar_path(args.output, "parse_methods")

    raw_responses = load_json_if_exists(raw_output) if args.resume else {}
    parse_methods = load_json_if_exists(parse_methods_output) if args.resume else {}

    def flush_checkpoint(elapsed_seconds):
        try:
            atomic_dump(args.output, ordered_mapping(dataset, predictions))
            atomic_dump(raw_output, ordered_mapping(dataset, raw_responses))
            atomic_dump(parse_methods_output, ordered_mapping(dataset, parse_methods))

            new_samples = len(predictions) - len(existing_preds)
            logger.info(
                f"  [CKPT] flushed at {len(predictions)} total predictions "
                f"({new_samples} new this run, elapsed {elapsed_seconds:.1f}s)"
            )
        except Exception as exc:
            logger.error(f"  [CKPT] flush failed: {exc}")

    task_args = [
        (sample, args.task_type, args.reasoning_mode, logger, args)
        for sample in todo_samples
    ]

    start_time = time.time()
    completed_since_flush = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, task_arg): task_arg for task_arg in task_args}
        for future in as_completed(futures):
            try:
                sample_id, result, response, _reasoning, parse_method = future.result()
            except Exception as exc:
                task_arg = futures[future]
                sample = task_arg[0]
                sample_id = sample["id"]
                logger.error(f"Exception processing {sample_id}: {exc}")
                if args.task_type.startswith("mc_"):
                    result = 0
                else:
                    result = {key: 0 for key in sample["query_keys"]}
                response = None
                _reasoning = ""
                parse_method = "exception"

            predictions[sample_id] = result
            raw_responses[sample_id] = response
            parse_methods[sample_id] = parse_method

            completed_since_flush += 1
            if completed_since_flush >= max(args.checkpoint_every, 1):
                flush_checkpoint(time.time() - start_time)
                completed_since_flush = 0

    elapsed = time.time() - start_time
    flush_checkpoint(elapsed)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Open-Source Inference Summary")
    logger.info("=" * 60)
    logger.info(f"  Model:                {MODEL_NAME}")
    logger.info(f"  Task:                 {args.task_type}")
    logger.info(f"  Setting:              {args.setting}")
    logger.info(f"  Samples processed:    {len(todo_samples)}")
    logger.info(f"  Total predictions:    {len(predictions)}")
    logger.info(f"  Elapsed:              {elapsed:.1f}s")
    logger.info(f"  Prompt tokens:        {total_prompt_tokens:,}")
    logger.info(f"  Completion tokens:    {total_completion_tokens:,}")
    logger.info(f"  Total tokens:         {total_total_tokens:,}")
    logger.info(f"  Parse stats:          {dict(parse_stats)}")


if __name__ == "__main__":
    main()
