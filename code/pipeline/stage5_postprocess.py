#!/usr/bin/env python3
"""
用 Kimi API 做脚注后处理。

- 输入：模型原始转录 JSON（screening_tables_recognized*）
- 让 LLM 把脚注从列名和单元格值中分离出来
- 纯文本任务，通过 OpenAI-compatible API (Kimi) 并发请求

用法:
  python3 pipeline/stage5_postprocess.py --api-url http://localhost:8000/v1/chat/completions
  python3 pipeline/stage5_postprocess.py --api-url http://localhost:8000/v1/chat/completions --concurrency 32
  python3 pipeline/stage5_postprocess.py --api-url ... --clean   # 清空旧结果重跑
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.utils.api_usage import (  # noqa: E402
    resolve_api_key, build_headers, log_usage, DEFAULT_USAGE_LOG,
)

BASE = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = [
    BASE / "screening_tables_recognized",
    BASE / "screening_tables_recognized_2025",
]
DEFAULT_OUTPUT = BASE / "screening_tables_processed"

PROMPT_TEMPLATE = """You are given a JSON object representing a chemistry screening table transcribed from a paper image. The table may contain footnote markers (superscript letters like a, b, c) embedded in column names, cell values, and entry numbers.

Your task: separate footnote markers from the actual data values.

Rules:
1. Use the "footnotes" field to know which letters are valid footnote markers.
2. For column names: remove footnote letters and create a "column_footnotes" mapping.
   - "Yield (%)b" → column "Yield (%)", footnote b
   - "Eec, drb" → column "Ee, dr", footnotes c and b
   - "Yielda,b" → column "Yield", footnotes a and b
3. For cell values: remove trailing footnote letters ONLY if they are footnote markers.
   - "85a" where a is a footnote → value "85", note a
   - "1b" where "1b" is a substrate/entry label (not a footnote usage) → keep "1b"
   - Use context: Entry/Substrate/Product/Catalyst columns often have compound labels like "1a", "2b", "3aa" — these are NOT footnotes even if the letter matches.
   - Yield/ee/dr/ratio columns with trailing letters are usually footnotes.
4. Row keys in output MUST match the cleaned column names.
5. Add "notes" field to rows: {{"column_name": "letter: footnote content"}}
6. Add "column_footnotes" field: {{"column_name": "letter: footnote content"}}

Input JSON:
{input_json}

Return ONLY the processed JSON with these changes applied. No explanation, no markdown code blocks."""


# ── API helpers ──────────────────────────────────────────────────────────────

_cached_model = None

def resolve_model(api_url: str, model_name: str = None) -> str:
    global _cached_model
    if _cached_model:
        return _cached_model
    if model_name:
        _cached_model = model_name
        return model_name
    base = api_url.rsplit("/v1/", 1)[0] + "/v1"
    try:
        resp = requests.get(f"{base}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            _cached_model = models[0]["id"]
            print(f"  Auto-detected model: {_cached_model}")
            return _cached_model
    except Exception as e:
        print(f"  Warning: could not auto-detect model: {e}")
    _cached_model = "kimi-k2.6"
    return _cached_model


def call_api(api_url: str, model: str, table_json: dict,
             max_tokens: int = 4096, timeout: int = 120,
             headers: dict | None = None,
             usage_log: str | None = None,
             doi: str | None = None, page: str | None = None) -> dict | None:
    """Send one table to the API, return parsed result or None.

    Rows in the LLM response are IGNORED — stage5 only needs LLM output for
    column names and footnote structure. Rows are always rebuilt from the
    original input applying (rename + footnote extraction), so the LLM can
    never drop rows regardless of table size.
    """
    all_rows = table_json.get("rows", [])
    # Always trim rows sent to the LLM (keeps prompt small; rows are
    # restored from `all_rows` below regardless).
    send_table = dict(table_json)
    if len(all_rows) > 10:
        send_table["rows"] = all_rows[:10]
        send_table["_note"] = f"Showing 10 of {len(all_rows)} rows. Process based on these."

    input_json = json.dumps(send_table, ensure_ascii=False, indent=2)
    prompt_text = PROMPT_TEMPLATE.format(input_json=input_json)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 1,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp_json = resp.json()
    msg = resp_json["choices"][0]["message"]
    if usage_log:
        log_usage(usage_log, stage="postprocess", doi=doi, page=page,
                  model=model, usage=resp_json.get("usage"),
                  message=msg, elapsed=time.time() - t0)
    text = msg.get("content") or ""
    result = parse_response(text)

    # ALWAYS restore rows from the original input. The LLM's returned rows
    # are discarded; we only trust its columns/footnotes. This guarantees
    # stage5 never drops rows.
    if result and isinstance(result, dict) and "columns" in result:
        old_cols = table_json.get("columns", [])
        new_cols = result.get("columns", [])
        # Build rename map from old → new column names
        rename = {}
        if len(old_cols) == len(new_cols):
            rename = {old: new for old, new in zip(old_cols, new_cols) if old != new}

        # Re-process ALL original rows: rename keys + extract footnote notes
        footnotes = result.get("footnotes", table_json.get("footnotes", {}))
        fn_keys = set(footnotes.keys()) if isinstance(footnotes, dict) else set()

        restored_rows = []
        for row in all_rows:
            new_row = {}
            notes = {}
            for col, val in row.items():
                new_col = rename.get(col, col)
                if isinstance(val, str) and fn_keys:
                    # Strip trailing footnote letters from numeric-like values
                    import re as _re
                    m = _re.match(r'^(.*[\d)/%\]])([a-z](?:,[a-z])*)$', val.strip())
                    if m:
                        matched = [l for l in m.group(2).split(',') if l.strip() in fn_keys]
                        if matched:
                            new_row[new_col] = m.group(1).strip()
                            notes[new_col] = "; ".join(
                                f"{l}: {footnotes[l]}" for l in matched if footnotes.get(l))
                            continue
                new_row[new_col] = val
            if notes:
                new_row["notes"] = notes
            restored_rows.append(new_row)
        result["rows"] = restored_rows
        result.pop("_note", None)

    return result


def parse_response(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    # Remove <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Progress tracking ────────────────────────────────────────────────────────

def load_progress(progress_file):
    if progress_file.exists():
        try:
            return json.loads(progress_file.read_text())
        except Exception:
            pass
    return {"done": [], "failed": []}


def save_progress(progress, progress_file):
    progress_file.write_text(json.dumps(progress, ensure_ascii=False, indent=2))


# ── Task collection ──────────────────────────────────────────────────────────

def collect_tasks(input_dirs):
    tasks = []
    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            continue
        for jf in sorted(input_dir.rglob("*.json")):
            if "_progress" in jf.name or "_debug" in str(jf):
                continue
            rel = jf.relative_to(input_dir)
            tasks.append((jf, rel))
    return tasks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="API-based footnote post-processing")
    parser.add_argument("--api-url", required=True,
                        help="vLLM chat/completions endpoint URL")
    parser.add_argument("--api-key", default=None,
                        help="API key (也可由 PROXY_API_KEY / OPENAI_API_KEY 等环境变量提供)")
    parser.add_argument("--usage-log", default=None,
                        help=f"Token 用量 JSONL 日志 (默认 {DEFAULT_USAGE_LOG})")
    parser.add_argument("--api-model", default=None,
                        help="Model name (auto-detected if omitted)")
    parser.add_argument("--input-dir", nargs="*", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-tokens", type=int, default=45000)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--clean", action="store_true",
                        help="Clear output dir and progress before running")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip already processed tables (default)")
    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dir] if args.input_dir \
        else [Path(d) for d in DEFAULT_INPUTS]
    output_dir = Path(args.output_dir)

    if args.clean:
        import shutil
        if output_dir.exists():
            print(f"Cleaning {output_dir} ...", flush=True)
            shutil.rmtree(output_dir)
            print(f"Cleaned.", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / "_postprocess_progress.json"
    if args.clean and progress_file.exists():
        progress_file.unlink()

    progress = load_progress(progress_file)
    if not args.resume:
        progress["failed"] = []
        save_progress(progress, progress_file)
    done_set = set(progress["done"])

    # Resolve model name
    model = resolve_model(args.api_url, args.api_model)
    api_key = resolve_api_key(args.api_key)
    headers = build_headers(api_key)
    usage_log = args.usage_log or DEFAULT_USAGE_LOG
    print(f"Auth: {'Bearer ***' if api_key else '(none)'}", flush=True)
    print(f"Usage log: {usage_log}", flush=True)

    # Collect ALL tables (with and without footnotes)
    print("Collecting input files...", flush=True)
    file_tasks = collect_tasks(input_dirs)
    print(f"Found {len(file_tasks)} input files, loading tables...", flush=True)

    # Split into: needs LLM (has footnotes) vs pass-through (no footnotes)
    llm_tasks = []        # (jf, rel, idx, table, task_key)
    passthrough = {}      # rel -> [table, ...]

    for jf, rel in file_tasks:
        try:
            data = json.load(open(jf))
        except Exception:
            continue
        for idx, table in enumerate(data):
            if not isinstance(table, dict):
                continue
            if table.get("footnotes"):
                task_key = f"{rel}::{idx}"
                if task_key not in done_set:
                    llm_tasks.append((jf, rel, idx, table, task_key))
            else:
                passthrough.setdefault(str(rel), []).append(table)

    total = len(llm_tasks)
    print(f"API: {args.api_url}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Tables needing LLM: {total}", flush=True)
    print(f"Already done: {len(done_set)}", flush=True)
    print(f"Pass-through (no footnotes): {sum(len(v) for v in passthrough.values())}", flush=True)
    print(f"Concurrency: {args.concurrency}", flush=True)
    print("=" * 60, flush=True)

    # ── Helper: write one output file (merge all tables for same rel path) ──
    import threading
    _write_lock = threading.Lock()
    _file_data = {}   # rel_str -> [tables]  (accumulated in memory, flushed per-file)

    def _flush_file(rel_str):
        """Write accumulated tables for one rel path to disk."""
        tables = _file_data.pop(rel_str, [])
        if not tables:
            return
        out_file = output_dir / rel_str
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Merge with existing file (for resume)
        existing = []
        if out_file.exists():
            try:
                existing = json.loads(out_file.read_text())
            except Exception:
                existing = []
        existing.extend(tables)
        out_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))

    # Write pass-through tables immediately.
    # IMPORTANT: merge with any existing file so that LLM-completed tables
    # written by a previous run are not clobbered when we re-run stage5 and
    # those LLM tasks are skipped via `done_set` (while pass-through tables
    # have no resume check and would otherwise overwrite the file).
    print("Writing pass-through tables...", flush=True)
    pt_files = 0
    for rel_str, tables in passthrough.items():
        out_file = output_dir / rel_str
        out_file.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if out_file.exists():
            try:
                existing = json.loads(out_file.read_text())
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        # Dedup on (page, table_title, image) so reruns don't duplicate
        # pass-through entries while still preserving previously written
        # LLM-processed tables for the same file.
        def _key(t):
            return (t.get("page", ""), t.get("table_title", ""), t.get("image", ""))
        seen = {_key(t) for t in existing if isinstance(t, dict)}
        merged = list(existing)
        for t in tables:
            if _key(t) not in seen:
                merged.append(t)
                seen.add(_key(t))
        out_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        pt_files += 1
    print(f"  Written {pt_files} pass-through files.", flush=True)

    # Group LLM tasks by rel path so we know when a file is complete
    from collections import defaultdict
    tasks_per_file = defaultdict(int)
    done_per_file = defaultdict(int)
    for _, rel, _, _, _ in llm_tasks:
        tasks_per_file[str(rel)] += 1

    stats = {"ok": 0, "failed": 0}
    start_time = time.time()
    files_written = pt_files

    if not llm_tasks:
        print("No LLM tasks to process.", flush=True)
    else:
        def process_one(task):
            jf, rel, idx, table, task_key = task
            try:
                # rel is like "JACS/10.1021__jacs.4c13321_si.json" -> doi_safe
                rel_str = str(rel)
                rel_parts = rel_str.split("/")
                doi_safe = Path(rel_parts[-1]).stem if rel_parts else None
                page_str = table.get("page") if isinstance(table, dict) else None
                result = call_api(args.api_url, model, table,
                                  args.max_tokens, args.timeout,
                                  headers=headers, usage_log=usage_log,
                                  doi=doi_safe, page=page_str)
                if result is None:
                    return task_key, str(rel), None, table
                if isinstance(result, list):
                    result = result[0] if result else None
                if not isinstance(result, dict) or "columns" not in result:
                    return task_key, str(rel), None, table
                # Preserve metadata
                result["image"] = table.get("image", "")
                result["page"] = table.get("page", "")
                if "table_title" not in result:
                    result["table_title"] = table.get("table_title", "")
                if "caption" not in result:
                    result["caption"] = table.get("caption", "")
                result["reaction"] = table.get("reaction", {})
                if "footnotes" not in result:
                    result["footnotes"] = table.get("footnotes", {})
                # Preserve cross-page / completeness flags so that stage8b and
                # stage10 can still tell the table was stitched from adjacent
                # pages and pull in neighbour images when re-prompting.
                for _k in ("is_incomplete", "incomplete_reason",
                           "completed_from_adjacent"):
                    if _k in table:
                        result[_k] = table[_k]
                return task_key, str(rel), result, table
            except Exception as e:
                return task_key, str(rel), None, table

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process_one, t): t for t in llm_tasks}
            done_count = 0

            for fut in as_completed(futures):
                task_key, rel_str, result, original = fut.result()
                done_count += 1

                with _write_lock:
                    if result is not None:
                        _file_data.setdefault(rel_str, []).append(result)
                        progress["done"].append(task_key)
                        stats["ok"] += 1
                    else:
                        _file_data.setdefault(rel_str, []).append(original)
                        progress["failed"].append(task_key)
                        stats["failed"] += 1

                    done_per_file[rel_str] += 1
                    # Flush file to disk when all its tables are done
                    if done_per_file[rel_str] >= tasks_per_file[rel_str]:
                        _flush_file(rel_str)
                        files_written += 1

                if done_count % 20 == 0 or done_count == total:
                    elapsed = (time.time() - start_time) / 60
                    speed = done_count / max(elapsed, 0.01)
                    remaining = (total - done_count) / max(speed, 0.01)
                    print(f"  [{done_count}/{total}] ok={stats['ok']} fail={stats['failed']} "
                          f"files={files_written} ({speed:.1f}/min, ~{remaining:.0f}min left)",
                          flush=True)

                if done_count % 50 == 0:
                    save_progress(progress, progress_file)

        # Flush any remaining files
        with _write_lock:
            for rel_str in list(_file_data.keys()):
                _flush_file(rel_str)
                files_written += 1
        save_progress(progress, progress_file)

    total_time = time.time() - start_time
    print(f"\nDone in {total_time/60:.1f} min", flush=True)
    print(f"  LLM OK: {stats['ok']}", flush=True)
    print(f"  LLM Failed: {stats['failed']}", flush=True)
    print(f"  Files written: {files_written}", flush=True)
    if total_time > 0 and stats['ok'] > 0:
        print(f"  Speed: {stats['ok']/(total_time/60):.1f} tables/min", flush=True)


if __name__ == "__main__":
    main()
