"""
Stage 4b: 补全不完整表格
对 Stage 4 标记为 is_incomplete 的表格，取对应的上下页，
重新提交给模型以获得完整的表格信息。

用法:
  python3 pipeline/stage4b_complete_tables.py \
    --recognized-dir screening_tables_recognized \
    --input-dirs screening_tables_kimi screening_tables_si \
    --api-url http://localhost:8000/v1/chat/completions \
    --concurrency 32 --timeout 600
"""

import json, time, os, re, argparse, base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
import threading

import requests
from PIL import Image
import filelock

from pipeline.utils.api_usage import (
    DEFAULT_USAGE_LOG,
    build_headers,
    log_usage,
    resolve_api_key,
)

PROMPT_TEMPLATE = """You are given {n_pages} consecutive pages from a chemistry paper.
{page_descriptions}

On {target_page_desc}, there are INCOMPLETE screening/optimization table(s) that need to be completed.
The following table(s) were detected as incomplete:
{incomplete_tables_info}

Your task: Complete ONLY the incomplete table(s) listed above by finding their missing parts (headers, data rows, footnotes) from the adjacent pages. Do NOT return any other tables — only the ones listed above, now completed.

Return JSON only (no other text, no markdown code blocks):

{{"tables": [
  {{
    "table_title": "Table title exactly as shown",
    "reaction": {{
      "substrates": [{{"label": "1a", "smiles": "SMILES or empty string", "description": "text description"}}],
      "products": [{{"label": "2a", "smiles": "SMILES or empty string", "description": "text description"}}],
      "conditions_on_arrow": ["Pd(OAc)2 (10 mol%)", "K2CO3 (2.0 equiv.)", "DMF, 80°C, 12h"]
    }},
    "caption": "ALL text below the table as one string, verbatim.",
    "footnotes": {{"a": "Reaction conditions: ...", "b": "Isolated yields."}},
    "columns": ["Col1", "Col2"],
    "rows": [{{"Col1": "val1", "Col2": "val2"}}, ...],
    "is_incomplete": false,
    "incomplete_reason": ""
  }}
]}}

RULES:
1. ONLY complete the table(s) listed above. Do NOT add tables from adjacent pages that are not in the list.
2. FAITHFUL TRANSCRIPTION: Copy all values exactly as they appear. Keep superscript footnote letters in column names, cell values, and Entry numbers.
3. MERGE across pages: If the table header is on one page and data rows continue on another, combine them into ONE complete table with ALL rows.
4. ALL rows must be included — do not skip any entry from any page.
5. caption/footnotes: Extract from whichever page has them. Combine if split across pages.
6. Row keys MUST exactly match the column names in "columns".
7. reaction: substrates (left of arrow), products (right of arrow). Convert structures to SMILES when possible (empty string "" if uncertain).
8. If a single table spans pages, merge into ONE entry. But do NOT return separate tables that were not listed as incomplete."""


# ─── progress helpers ────────────────────────────────────────────────

def load_progress(progress_file):
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {"done": [], "failed": []}


def save_progress(progress, progress_file):
    lock = filelock.FileLock(str(progress_file) + ".lock", timeout=10)
    with lock:
        if progress_file.exists():
            latest = json.loads(progress_file.read_text())
        else:
            latest = {"done": [], "failed": []}
        for key in ("done", "failed"):
            merged = list(dict.fromkeys(latest[key] + progress[key]))
            latest[key] = merged
        progress_file.write_text(json.dumps(latest, ensure_ascii=False, indent=2))


# ─── response parsing ────────────────────────────────────────────────

def parse_response(text):
    if not text:
        return None
    text = text.strip()
    think_end = text.find("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>"):].strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ─── image helpers ───────────────────────────────────────────────────

def image_to_b64(img_path: Path, max_side: int = 2048) -> str | None:
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if max_side > 0 and max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        img.close()
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


# ─── API helpers ─────────────────────────────────────────────────────

_cached_api_model = None


def resolve_api_model(api_url: str, headers: dict, requested: str | None = None) -> str:
    global _cached_api_model
    if requested:
        return requested
    if _cached_api_model:
        return _cached_api_model
    base = api_url.rsplit("/v1/", 1)[0] + "/v1"
    try:
        resp = requests.get(f"{base}/models", headers=headers, timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            _cached_api_model = models[0]["id"]
            return _cached_api_model
    except Exception:
        pass
    _cached_api_model = "kimi-k2.6"
    return _cached_api_model


def call_api_multi_image(api_url, model, images_b64, prompt, headers,
                         usage_log, task_key, max_tokens=45500, timeout=600):
    """Send multiple images + prompt to the API."""
    content = []
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.time()
    resp = requests.post(
        api_url, json=payload, headers=headers, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    log_usage(
        usage_log,
        stage="complete",
        model=model,
        usage=data.get("usage"),
        elapsed=time.time() - started,
        message=message,
        extra={"task_key": task_key},
    )
    return message["content"]


# ─── task collection ─────────────────────────────────────────────────

def collect_incomplete_tasks(recognized_dir: Path, input_dirs: list,
                             all_pages_dirs: list):
    """Scan _incomplete_tables/ and build task list with adjacent page paths.

    Args:
        recognized_dir: Stage 4 output directory
        input_dirs: Stage 4 input dirs (screened pages, for finding current image)
        all_pages_dirs: Full rendered page dirs (all pages, for finding prev/next)
    """
    inc_dir = recognized_dir / "_incomplete_tables"
    if not inc_dir.exists():
        print(f"未找到 _incomplete_tables 目录: {inc_dir}")
        return []

    tasks = []
    for json_file in sorted(inc_dir.rglob("*.json")):
        data = json.loads(json_file.read_text())
        rel_key = data.get("image", "")
        page = data.get("page", "")
        tables = data.get("tables", [])

        if not rel_key or not page or not tables:
            continue

        # Find source image in input_dirs (screened)
        current_img = None
        for idir in input_dirs:
            candidate = idir / rel_key
            if candidate.exists():
                current_img = candidate
                break

        if current_img is None:
            print(f"  跳过 (源图未找到): {rel_key}")
            continue

        # Parse page number
        match = re.match(r'p(\d+)\.png', page)
        if not match:
            continue
        page_num = int(match.group(1))

        # Find adjacent pages in all_pages_dirs (full, unfiltered)
        # rel_key is like "Angew/10.1002_anie.202219166/p6.png"
        # We need to find the same DOI dir in all_pages_dirs
        rel_dir = str(Path(rel_key).parent)  # "Angew/10.1002_anie.202219166"
        prev_img = None
        next_img = None
        for apdir in all_pages_dirs:
            doi_dir = apdir / rel_dir
            if doi_dir.exists():
                p = doi_dir / f"p{page_num - 1}.png"
                n = doi_dir / f"p{page_num + 1}.png"
                if p.exists():
                    prev_img = p
                if n.exists():
                    next_img = n
                break  # found the DOI dir

        # Collect incomplete table info for prompt
        incomplete_tables_info = []
        for t in tables:
            title = t.get("table_title", "untitled")
            reason = t.get("incomplete_reason", "unknown")
            incomplete_tables_info.append(f"- \"{title}\": {reason}")
        incomplete_tables_str = "\n".join(incomplete_tables_info) or "- unknown table"

        # Legacy field for backwards compatibility
        reasons = [t.get("incomplete_reason", "") for t in tables]
        reason_str = "; ".join(r for r in reasons if r) or "unknown"

        tasks.append({
            "key": rel_key,
            "page": page,
            "page_num": page_num,
            "current_img": current_img,
            "prev_img": prev_img,
            "next_img": next_img,
            "incomplete_reason": reason_str,
            "incomplete_tables_info": incomplete_tables_str,
            "json_file": json_file,
        })

    return tasks


# ─── prompt building ─────────────────────────────────────────────────

def build_prompt_and_images(task, downscale):
    """Build prompt and encode images for one task."""
    pages = []
    page_descs = []
    idx = 1
    target_idx = None

    if task["prev_img"]:
        pages.append(task["prev_img"])
        page_descs.append(f"Image {idx}: Page {task['page_num'] - 1} (previous page)")
        idx += 1

    pages.append(task["current_img"])
    target_idx = idx
    page_descs.append(f"Image {idx}: Page {task['page_num']} (page with the incomplete table)")
    idx += 1

    if task["next_img"]:
        pages.append(task["next_img"])
        page_descs.append(f"Image {idx}: Page {task['page_num'] + 1} (next page)")
        idx += 1

    prompt = PROMPT_TEMPLATE.format(
        n_pages=len(pages),
        page_descriptions="\n".join(page_descs),
        target_page_desc=f"Image {target_idx}",
        incomplete_tables_info=task.get("incomplete_tables_info", task["incomplete_reason"]),
    )

    images_b64 = []
    for pg in pages:
        b64 = image_to_b64(pg, max_side=downscale)
        if b64 is None:
            return None, None, 0
        images_b64.append(b64)

    return images_b64, prompt, len(pages)


# ─── update results ─────────────────────────────────────────────────

def update_recognized_json(recognized_dir, rel_key, new_tables):
    """Replace incomplete table entries in the main JSON with completed ones."""
    parts = rel_key.split("/")
    journal = parts[0]
    doi_dir = parts[1] if len(parts) > 1 else "unknown"
    page = parts[-1]

    out_file = recognized_dir / journal / f"{doi_dir}.json"
    if not out_file.exists():
        return False

    lock = filelock.FileLock(str(out_file) + ".lock", timeout=10)
    with lock:
        existing = json.loads(out_file.read_text())

        # Build lookup of completed tables by title for matching
        new_by_title = {}
        for t in new_tables:
            title = t.get("table_title", "").strip()
            if title:
                new_by_title[title] = t

        updated = []
        replaced_titles = set()
        for entry in existing:
            if entry.get("image") == rel_key and entry.get("is_incomplete"):
                # This is an incomplete table on the same page — replace it
                entry_title = entry.get("table_title", "").strip()
                matched_new = new_by_title.get(entry_title)
                if matched_new and entry_title not in replaced_titles:
                    new_entry = {
                        "image": rel_key,
                        "page": page,
                        "table_title": matched_new.get("table_title", ""),
                        "reaction": matched_new.get("reaction", {}),
                        "caption": matched_new.get("caption", ""),
                        "footnotes": matched_new.get("footnotes", {}),
                        "columns": matched_new.get("columns", []),
                        "rows": matched_new.get("rows", []),
                        "is_incomplete": matched_new.get("is_incomplete", False),
                        "incomplete_reason": matched_new.get("incomplete_reason", ""),
                        "completed_from_adjacent": True,
                    }
                    updated.append(new_entry)
                    replaced_titles.add(entry_title)
                else:
                    # No matching new table found, keep original
                    updated.append(entry)
            else:
                # Complete table or different page — keep as-is
                updated.append(entry)

        # Append any new tables that didn't match an existing incomplete entry
        existing_titles = {(e.get("page"), e.get("table_title", "").strip()) for e in updated}
        for title, t in new_by_title.items():
            if title not in replaced_titles and (page, title) not in existing_titles:
                new_entry = {
                    "image": rel_key,
                    "page": page,
                    "table_title": t.get("table_title", ""),
                    "reaction": t.get("reaction", {}),
                    "caption": t.get("caption", ""),
                    "footnotes": t.get("footnotes", {}),
                    "columns": t.get("columns", []),
                    "rows": t.get("rows", []),
                    "is_incomplete": t.get("is_incomplete", False),
                    "incomplete_reason": t.get("incomplete_reason", ""),
                    "completed_from_adjacent": True,
                }
                updated.append(new_entry)

        out_file.write_text(json.dumps(updated, ensure_ascii=False, indent=2))
    return bool(replaced_titles)


# ─── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4b: 补全不完整表格 (取上下页重新识别)")
    parser.add_argument("--recognized-dir", type=str, required=True,
                        help="Stage 4 输出目录 (screening_tables_recognized)")
    parser.add_argument("--input-dirs", type=str, nargs="+", required=True,
                        help="Stage 4 输入图片目录 (筛选后的，用于定位当前页)")
    parser.add_argument("--all-pages-dirs", type=str, nargs="+", required=True,
                        help="完整渲染页面目录 (未筛选，用于查找上下页)")
    parser.add_argument("--api-url", type=str, required=True,
                        help="vLLM API URL")
    parser.add_argument("--api-key", default=None,
                        help="API key, or use OPENAI_API_KEY/KIMI_API_KEY")
    parser.add_argument("--api-model", default=None,
                        help="Model name; auto-detected when omitted")
    parser.add_argument("--usage-log", default=DEFAULT_USAGE_LOG,
                        help="Append-only token-usage JSONL")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="API 并发数")
    parser.add_argument("--max-tokens", type=int, default=45500,
                        help="最大生成 token 数")
    parser.add_argument("--timeout", type=int, default=600,
                        help="API 超时秒数")
    parser.add_argument("--downscale", type=int, default=2048,
                        help="图片最大边长 (0=不缩放)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="断点续跑 (--no-resume 重新开始)")
    args = parser.parse_args()

    recognized_dir = Path(args.recognized_dir)
    input_dirs = [Path(d) for d in args.input_dirs]
    all_pages_dirs = [Path(d) for d in args.all_pages_dirs]

    progress_file = recognized_dir / "_progress_4b.json"

    if args.resume:
        progress = load_progress(progress_file)
    else:
        progress = {"done": [], "failed": []}
        save_progress(progress, progress_file)

    done_set = set(progress["done"] + progress["failed"])

    # Collect tasks
    all_tasks = collect_incomplete_tasks(recognized_dir, input_dirs, all_pages_dirs)
    todo = [t for t in all_tasks if t["key"] not in done_set]

    print(f"不完整表格总计: {len(all_tasks)}")
    print(f"已处理: {len(done_set)}")
    print(f"待处理: {len(todo)}")
    print("=" * 60)

    if not todo:
        print("全部处理完毕！")
        return

    headers = build_headers(resolve_api_key(args.api_key))
    model = resolve_api_model(args.api_url, headers, args.api_model)
    print(f"API: {args.api_url}")
    print(f"模型: {model}")
    print(f"并发: {args.concurrency}")

    stats = {"completed": 0, "failed": 0}
    start_time = time.time()
    lock = threading.Lock()
    counter = [0]

    def process_one(task):
        images_b64, prompt, n_pages = build_prompt_and_images(task, args.downscale)
        if images_b64 is None:
            with lock:
                progress["failed"].append(task["key"])
                stats["failed"] += 1
                counter[0] += 1
            return task["key"], "FAILED (image encode error)"

        try:
            text = call_api_multi_image(
                args.api_url,
                model,
                images_b64,
                prompt,
                headers,
                args.usage_log,
                task["key"],
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        except Exception as e:
            with lock:
                progress["failed"].append(task["key"])
                stats["failed"] += 1
                counter[0] += 1
            return task["key"], f"FAILED ({e})"

        result = parse_response(text)
        if result is None:
            debug_dir = recognized_dir / "_debug_4b"
            debug_dir.mkdir(exist_ok=True)
            safe_name = task["key"].replace("/", "_")
            (debug_dir / f"{safe_name}.txt").write_text(text)
            with lock:
                progress["failed"].append(task["key"])
                stats["failed"] += 1
                counter[0] += 1
            return task["key"], "FAILED (parse error)"

        tables = result.get("tables", [])
        if not tables:
            with lock:
                progress["failed"].append(task["key"])
                stats["failed"] += 1
                counter[0] += 1
            return task["key"], "FAILED (no tables in response)"

        with lock:
            update_recognized_json(recognized_dir, task["key"], tables)
            total_rows = sum(len(t.get("rows", [])) for t in tables)

            progress["done"].append(task["key"])
            stats["completed"] += 1
            counter[0] += 1

            if counter[0] % 5 == 0:
                save_progress(progress, progress_file)

        return task["key"], f"OK ({len(tables)} table(s), {total_rows} rows, {n_pages} pages)"

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process_one, t): t for t in todo}
        for future in as_completed(futures):
            key, status = future.result()
            idx = counter[0]
            elapsed_min = (time.time() - start_time) / 60
            speed = idx / max(elapsed_min, 0.01)
            remaining = (len(todo) - idx) / max(speed, 0.01) if speed > 0 else 0
            print(f"[{idx}/{len(todo)}] ({speed:.1f}/min, 剩余~{remaining:.0f}min) "
                  f"{key} ... {status}")

    save_progress(progress, progress_file)

    print("=" * 60)
    total_time = time.time() - start_time
    print(f"完成! 耗时: {total_time/60:.1f} 分钟")
    print(f"  补全成功: {stats['completed']}")
    print(f"  失败: {stats['failed']}")


if __name__ == "__main__":
    main()
