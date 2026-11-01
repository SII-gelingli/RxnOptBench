#!/usr/bin/env python3
"""Convert clean benchmark rows to the human-review schema with a VLM.

This stage is the portable integration of the workspace conversion path used
between clean benchmark construction and human review.  It consolidates the
row grouping/API/output behavior of ``benchmark/convert_benchmark_to_schema``
with the final v2 field semantics in
``benchmark/convert_benchmark_to_schema_v2.py``.

Input layout:
    <input-dir>/<journal>/<table>.json

Each table file contains metadata plus a ``rows`` list produced by
``utils/build_clean_benchmark.py``. Output mirrors the same relative paths and
maps each row index to a list of schema-format reactions.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# Make the sibling ``pipeline`` package importable for direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.utils.api_usage import (
    DEFAULT_USAGE_LOG,
    build_headers,
    log_usage,
    resolve_api_key,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "benchmark_release" / "by_journal"
DEFAULT_OUTPUT = BASE_DIR / "schema_output"
DEFAULT_SCHEMA = Path(__file__).resolve().parent / "schema" / "review_schema.json"

SYSTEM_PROMPT = (
    "You parse organic-chemistry screening and optimization tables. "
    "Return valid JSON only, without reasoning or markdown fences."
)

PROMPT_TEMPLATE = """\
## Task
Convert the following cleaned benchmark row into the target schema. Use the
original page image as the authoritative source for amounts, equivalents,
mol %, solvent ratios, temperatures, times, atmosphere, electrochemistry, and
yields.

## Cleaned benchmark row
```json
{row_json}
```

## Table context
- Journal: {journal}
- DOI: {doi}
- Page: {page}
- Table title: {table_title}
- Conditions on arrow: {conditions_on_arrow}

## Target schema
```json
{schema_json}
```

## Rules
1. Return a JSON array containing exactly one reaction object.
2. Preserve every supplied reactant, product, and ligand SMILES exactly.
3. Classify conditions into reagents, catalysts, solvents, ligands, or
   indeterminate molecules. Do not create all-empty list items.
4. Keep conversion, selectivity/ratio, and yield distinct. Conversion belongs
   in reactants[].conversion_rate. Populate product yield fields only when the
   source reports a yield for that product. Do not turn a selectivity ratio
   into a yield. If exactly one product is present, no product yield is
   reported, and conversion is reported, copy that conversion to yield_ratio
   and record "yield taken from conversion; single product reported" in
   other_info.
5. Use multiple stage objects only for genuinely sequential operations.
6. Use empty strings or empty lists for fields not reported. Do not add keys
   outside the supplied schema, and do not emit all-empty list items.
7. Set is_multi_reactions to "True" only when one row encodes multiple
   distinct reactions, and explain their unpacking in R-info.
8. In each stage, selected_conditions must list exactly the non-empty
   condition keys in that same stage. Do not select current_conditions for
   non-electrochemical reactions.
9. Keep box_id empty because bounding-box metadata is not supplied here.
10. For electrochemistry, fill current_conditions only when reported and keep
    Constant_* and Alternating_* modes mutually consistent.
11. Remove any meta_info wrapper from the result.
"""


def doi_dirnames(doi: str) -> list[str]:
    """Return directory-name variants used by the rendered-page trees."""
    double = doi.replace("/", "__")
    single = doi.replace("/", "_")
    names = [f"{double}_si", double, f"{single}_si", single]
    seen = set()
    return [name for name in names if not (name in seen or seen.add(name))]


def find_page_image(
    image_roots: list[Path],
    journal: str,
    doi: str,
    page: int | str,
) -> Path | None:
    page_name = f"p{page}.png"
    for root in image_roots:
        for journal_dir in (journal, "new_batch"):
            for doi_dir in doi_dirnames(doi):
                path = root / journal_dir / doi_dir / page_name
                if path.exists():
                    return path
    return None


def image_to_b64(path: str | Path, max_side: int) -> str:
    image = Image.open(path).convert("RGB")
    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_response(text: str):
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        for match in reversed(list(re.finditer(pattern, cleaned))):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return None


def resolve_model(
    api_url: str,
    headers: dict,
    requested: str | None,
) -> str:
    if requested:
        return requested
    base = api_url.rsplit("/v1/", 1)[0] + "/v1"
    response = requests.get(f"{base}/models", headers=headers, timeout=10)
    response.raise_for_status()
    models = response.json().get("data") or []
    if not models:
        raise RuntimeError("The API /models response contains no model")
    return models[0]["id"]


def build_prompt(row: dict, table: dict, schema: dict) -> str:
    prompt_row = {
        "substrates": row.get("substrates") or [],
        "products": row.get("products") or [],
        "conditions": row.get("conditions") or [],
        "notes": row.get("notes", ""),
        "row_index": row.get("row_index"),
    }
    return PROMPT_TEMPLATE.format(
        row_json=json.dumps(prompt_row, ensure_ascii=False, indent=2),
        journal=table.get("journal", ""),
        doi=table.get("doi", ""),
        page=table.get("page", ""),
        table_title=table.get("table_title", ""),
        conditions_on_arrow=json.dumps(
            row.get("conditions_on_arrow") or [], ensure_ascii=False
        ),
        schema_json=json.dumps(schema, ensure_ascii=False, indent=2),
    )


def call_api(
    *,
    api_url: str,
    model: str,
    headers: dict,
    image_b64: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    usage_log: str,
    task: dict,
):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.time()
    response = requests.post(
        api_url,
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    log_usage(
        usage_log,
        stage="schema_convert",
        doi=task["table"].get("doi"),
        page=str(task["table"].get("page", "")),
        model=model,
        usage=data.get("usage"),
        elapsed=time.time() - started,
        message=message,
        extra={"task_key": task["task_key"]},
    )
    return parse_response(message.get("content", ""))


def restore_input_smiles(entries: list, row: dict) -> None:
    """Ensure the schema conversion cannot alter supplied molecular identity."""
    pairs = (
        ("reactants", row.get("substrates") or []),
        ("products", row.get("products") or []),
    )
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry.pop("meta_info", None)
        for target_key, sources in pairs:
            targets = entry.get(target_key) or []
            for target, source in zip(targets, sources):
                if isinstance(target, dict) and isinstance(source, dict):
                    target["SMILES"] = source.get(
                        "smiles", source.get("SMILES", "")
                    )


def collect_tasks(
    input_dir: Path,
    image_roots: list[Path],
) -> tuple[list[dict], int]:
    tasks = []
    missing_images = 0
    for table_file in sorted(input_dir.glob("*/*.json")):
        if table_file.name.startswith("_"):
            continue
        try:
            table = json.loads(table_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(table, dict):
            continue
        rows = table.get("rows") or []
        if not rows:
            continue
        image = find_page_image(
            image_roots,
            table.get("journal", table_file.parent.name),
            table.get("doi", ""),
            table.get("page", 0),
        )
        if image is None:
            missing_images += 1
            continue
        rel = str(table_file.relative_to(input_dir))
        for position, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            row_index = row.get("row_index", position)
            tasks.append({
                "task_key": f"{rel}::row{row_index}",
                "rel": rel,
                "row_index": row_index,
                "row": row,
                "table": table,
                "image": str(image),
            })
    return tasks, missing_images


def load_progress(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {"done": [], "failed": []}


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    temporary.replace(path)


def flush_results(
    results: dict[str, dict],
    dirty: set[str],
    output_dir: Path,
) -> int:
    written = 0
    for rel in list(dirty):
        output = output_dir / rel
        existing = {}
        if output.exists():
            try:
                existing = json.loads(output.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
        for row_index, entries in results.get(rel, {}).items():
            existing[str(row_index)] = entries
        atomic_write_json(output, existing)
        dirty.discard(rel)
        written += 1
    return written


def build_schema_all(output_dir: Path) -> list[dict]:
    """Aggregate every per-table output, including files from resumed runs."""
    flat: list[dict] = []
    for path in sorted(output_dir.glob("*/*.json")):
        if path.name.startswith("_"):
            continue
        try:
            table = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(table, dict):
            continue
        for row_index in sorted(table, key=lambda value: str(value)):
            entries = table[row_index]
            if isinstance(entries, list):
                flat.extend(
                    entry for entry in entries if isinstance(entry, dict)
                )
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert clean benchmark rows to the review schema via VLM"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--images-dir",
        type=Path,
        action="append",
        required=True,
        help="Rendered-page root; repeat for multiple roots",
    )
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--usage-log", default=DEFAULT_USAGE_LOG)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text())
    tasks, missing_images = collect_tasks(args.input_dir, args.images_dir)
    if args.test > 0:
        tasks = tasks[:args.test]
    print(
        f"Collected {len(tasks)} schema-conversion rows; "
        f"{missing_images} table files skipped without a page image"
    )
    if args.dry_run:
        for task in tasks[:10]:
            print(task["task_key"])
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = args.output_dir / "_convert_progress.json"
    progress = (
        load_progress(progress_file)
        if args.resume
        else {"done": [], "failed": []}
    )
    done = set(progress["done"])
    tasks = [task for task in tasks if task["task_key"] not in done]
    if not tasks:
        atomic_write_json(
            args.output_dir / "schema_all.json",
            build_schema_all(args.output_dir),
        )
        print("Nothing to do")
        return

    headers = build_headers(resolve_api_key(args.api_key))
    model = resolve_model(args.api_url, headers, args.api_model)
    results: dict[str, dict] = {}
    dirty: set[str] = set()
    lock = threading.Lock()

    @lru_cache(maxsize=128)
    def cached_image(path: str) -> str:
        return image_to_b64(path, args.max_side)

    def process(task: dict):
        try:
            parsed = call_api(
                api_url=args.api_url,
                model=model,
                headers=headers,
                image_b64=cached_image(task["image"]),
                prompt=build_prompt(task["row"], task["table"], schema),
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                usage_log=args.usage_log,
                task=task,
            )
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list) or not all(
                isinstance(item, dict) for item in parsed
            ):
                return task, None, "response is not a JSON object array"
            restore_input_smiles(parsed, task["row"])
            return task, parsed, None
        except Exception as exc:
            return task, None, f"{type(exc).__name__}: {str(exc)[:200]}"

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(process, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            task, parsed, error = future.result()
            with lock:
                if error is None:
                    results.setdefault(task["rel"], {})[
                        task["row_index"]
                    ] = parsed
                    dirty.add(task["rel"])
                    progress["done"].append(task["task_key"])
                else:
                    progress["failed"].append(task["task_key"])
                    print(f"FAILED {task['task_key']}: {error}")
                if count % 20 == 0:
                    flush_results(results, dirty, args.output_dir)
                    atomic_write_json(progress_file, progress)

    flush_results(results, dirty, args.output_dir)
    atomic_write_json(progress_file, progress)
    atomic_write_json(
        args.output_dir / "schema_all.json",
        build_schema_all(args.output_dir),
    )
    print(
        f"Converted {len(progress['done'])} rows; "
        f"{len(progress['failed'])} failures recorded"
    )


if __name__ == "__main__":
    main()
