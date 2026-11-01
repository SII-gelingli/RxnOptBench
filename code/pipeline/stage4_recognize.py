"""
批量表格识别脚本
- 支持两种模式:
  1. 离线模式: 直接用 vLLM LLM 类加载模型
  2. API 模式 (默认推荐): 调用 Kimi 或其他 OpenAI-compatible API

- 断点续跑，结果按 DOI 保存为 JSON

用法:
  # API 模式 (推荐，使用 Kimi):
  python3 batch_recognize_vllm.py --api-url http://localhost:8000/v1/chat/completions
  python3 batch_recognize_vllm.py --api-url http://localhost:8000/v1/chat/completions --concurrency 16

  # 离线模式:
  python3 batch_recognize_vllm.py --model /path/to/model

  # 离线分片模式 (8卡)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 batch_recognize_vllm.py --split 0/2 &
  CUDA_VISIBLE_DEVICES=4,5,6,7 python3 batch_recognize_vllm.py --split 1/2 &
"""

import json, time, os, re, argparse, base64, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# Make sibling package importable when run as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.utils.api_usage import (  # noqa: E402
    resolve_api_key, build_headers, log_usage, DEFAULT_USAGE_LOG,
)

# ============ 配置 ============
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = os.environ.get(
    "OMB_TABLE_MODEL",
    "kimi-k2.6",
)
DEFAULT_INPUT_DIR = str(BASE_DIR / "screening_tables_si")
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "screening_tables_recognized")

PROMPT = """Analyze this image and identify all reaction condition screening/optimization tables. A page may contain multiple tables — identify all of them.

A screening table typically has: Entry/No. column, columns for catalyst/solvent/base/temp/time/yield, and a reaction scheme above it.

Return JSON only (no other text, no markdown code blocks):

If screening table(s) found:
{"tables": [
  {
    "table_title": "Table title exactly as shown",
    "reaction": {
      "substrates": [{"label": "1a", "description": "text description of substrate"}],
      "products": [{"label": "2a", "description": "text description of product"}],
      "conditions_on_arrow": ["Pd(OAc)2 (10 mol%)", "K2CO3 (2.0 equiv.)", "DMF, 80°C, 12h"]
    },
    "caption": "ALL text below the table as one string, verbatim.",
    "footnotes": {"a": "Reaction conditions: ...", "b": "Isolated yields.", "c": "Determined by HPLC."},
    "columns": ["Col1", "Col2"],
    "rows": [{"Col1": "val1", "Col2": "val2"}, ...]
  }
]}

If no screening tables: {"tables": [], "reason": "brief reason"}

RULES:

1. FAITHFUL TRANSCRIPTION: Copy all values exactly as they appear in the image. Keep superscript footnote letters in column names, cell values, and Entry numbers. Do NOT remove or process footnotes.
   - Column name "Yield (%)b" → write "Yield (%)b"
   - Cell value "85a" → write "85a"
   - Cell value "96/ (95)d" → write "96/ (95)d"
   - Entry "14b" → write "14b"
   - Entry "26i,k" → write "26i,k"

2. caption: Extract ALL text below the table verbatim as one complete string. Do not omit any sentence.
   footnotes: Also extract each labeled footnote separately into a {"letter": "content"} object.
   Example caption: "aReaction conditions: 1a (0.2 mmol)... bIsolated yields. cDetermined by HPLC."
   → footnotes: {"a": "Reaction conditions: 1a (0.2 mmol)...", "b": "Isolated yields.", "c": "Determined by HPLC."}

3. Row keys MUST exactly match the column names in "columns".

4. reaction: substrates (left of arrow), products (right of arrow). Do NOT generate SMILES — leave the smiles field out entirely. Only provide the label (compound code like "1a", "2b") and a brief text description. SMILES will be determined separately by a dedicated OCSR model. List all conditions on/below the arrow.

5. IMPORTANT — REPORT EVERY TABLE: You MUST report ALL tables on the page, including both complete and incomplete ones. Do NOT skip a table just because another table on the same page is incomplete. A page may have a complete table (with all headers and rows) alongside an incomplete table (e.g. a title-only fragment at the bottom) — both must appear in your output.

6. INCOMPLETE TABLE DETECTION: A table is COMPLETE only when ALL of the following are present on this page:
   (a) Table title (e.g. "Table S1. ...")
   (b) Column headers
   (c) ALL data rows — the first entry must start from entry 1 (or the first number in the sequence), AND the last row must NOT be at the very bottom of the page with no space for caption/footnotes
   (d) Caption/footnotes text — almost every screening table has footnotes or a caption below it describing reaction conditions (e.g. "aReaction conditions: ..."). If the table has NO caption/footnotes text below it on this page, it is very likely incomplete (the caption is on the next page).

   Add an "is_incomplete" field to each table:
   - Set "is_incomplete": true in these cases:
     * Missing column headers — data rows present but no header row (this is a continuation from a previous page)
     * First entry does NOT start from 1 — this means earlier rows are on a previous page
     * Data rows reach the bottom of the page with no caption/footnotes below — the table likely continues on the next page
     * Only the table title is visible with no data rows on this page
     * Missing footnotes — footnote markers (e.g. superscript a, b, *) appear in the table title, column headers, or cell values, but the corresponding footnote text is NOT visible anywhere on this page. IMPORTANT: do NOT put empty strings as footnote values — if you cannot find the footnote text, mark the table as incomplete.
     * Missing caption — no descriptive text appears below the table on this page. Most screening tables have caption text like "Reaction conditions: ..." or "Unless otherwise noted, ..." below them. If this text is absent, the table is likely incomplete.
   - Set "is_incomplete": false ONLY when the table has title, column headers, ALL data rows (starting from entry 1), AND caption/footnotes text visible on this page.
   - Add "incomplete_reason": describe what is missing, e.g. "missing column headers (continuation page)", "first entry is 5, earlier rows on previous page", "no caption/footnotes below table, likely on next page", "table title only, no data rows on this page", "footnote markers a,b but text not found on this page", or empty string if complete.

   CRITICAL: If you detect footnote marker letters (a, b, etc.) in the table title or column headers but cannot find the actual footnote text on this page, you MUST set is_incomplete to true. Never output footnotes with empty string values like {{"a": "", "b": ""}} — that means you could not find them and the table is incomplete."""


import filelock

def load_progress(progress_file):
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {"done": [], "skipped": [], "failed": []}


def save_progress(progress, progress_file):
    lock = filelock.FileLock(str(progress_file) + ".lock", timeout=10)
    with lock:
        if progress_file.exists():
            latest = json.loads(progress_file.read_text())
        else:
            latest = {"done": [], "skipped": [], "failed": []}
        for key in ("done", "skipped", "failed"):
            merged = list(dict.fromkeys(latest[key] + progress[key]))
            latest[key] = merged
        progress_file.write_text(json.dumps(latest, ensure_ascii=False, indent=2))


def parse_response(text):
    if not text:
        return None
    text = text.strip()
    # Strip <think>...</think> blocks
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


def get_relative_key(image_path: Path, input_dir: Path):
    return str(image_path.relative_to(input_dir))


# ============ API 模式辅助函数 ============

_cached_api_model = None


def resolve_api_model(api_url: str, requested: str | None = None) -> str:
    """Auto-detect model name from vLLM /v1/models endpoint."""
    global _cached_api_model
    if _cached_api_model:
        return _cached_api_model
    if requested:
        _cached_api_model = requested
        return requested
    base = api_url.rsplit("/v1/", 1)[0] + "/v1"
    try:
        resp = requests.get(f"{base}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            _cached_api_model = models[0]["id"]
            print(f"  Auto-detected model: {_cached_api_model}")
            return _cached_api_model
    except Exception as e:
        print(f"  Warning: could not auto-detect model: {e}")
    _cached_api_model = "kimi-k2.6"
    return _cached_api_model


def image_to_b64(img_path: Path, max_side: int = 2048) -> str | None:
    """Read image, optionally downscale, return base64 PNG."""
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


def call_api(api_url: str, model: str, img_b64: str,
             max_tokens: int = 45500, timeout: int = 600,
             headers: dict | None = None,
             usage_log: str | None = None,
             doi: str | None = None, page: str | None = None) -> str | None:
    """Send one image to vLLM API, return response text."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 1,
    }
    try:
        t0 = time.time()
        resp = requests.post(api_url, json=payload, headers=headers,
                             timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        if usage_log:
            log_usage(usage_log, stage="recognize", doi=doi, page=page,
                      model=model, usage=data.get("usage"),
                      message=msg, elapsed=time.time() - t0)
        return msg.get("content") or ""
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")


def process_result(text, rel_key, OUTPUT_DIR, progress, stats, INPUT_DIR=None):
    """Process VLM response text: parse, save result, update progress.

    For incomplete (cross-page) tables, records the page in
    OUTPUT_DIR/_incomplete_tables/ for separate handling.
    """
    parts_list = rel_key.split("/")
    journal = parts_list[0]
    doi_dir = parts_list[1] if len(parts_list) > 1 else "unknown"
    page = parts_list[-1]

    result = parse_response(text)

    if result is None:
        debug_dir = OUTPUT_DIR / "_debug"
        debug_dir.mkdir(exist_ok=True)
        (debug_dir / f"{journal}_{doi_dir}_{page}.txt").write_text(text)
        progress["failed"].append(rel_key)
        stats["failed"] += 1
        return "FAILED (parse error)"

    tables = result.get("tables", [])
    if not tables:
        reason = result.get("reason", "unknown")
        progress["skipped"].append(rel_key)
        stats["skipped"] += 1
        return f"SKIP ({reason})"

    total_rows = sum(len(t.get("rows", [])) for t in tables)
    out_dir = OUTPUT_DIR / journal
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{doi_dir}.json"

    existing = []
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text())
        except Exception:
            existing = []

    # Remove old entries for the same page before appending new results
    existing = [e for e in existing if e.get("page") != page]

    has_incomplete = False
    for t in tables:
        is_incomplete = t.get("is_incomplete", False)
        incomplete_reason = t.get("incomplete_reason", "")

        # Post-validation: auto-detect incomplete tables the model missed
        footnotes = t.get("footnotes", {})
        caption = t.get("caption", "")
        rows = t.get("rows", [])
        columns = t.get("columns", [])

        # Check 1: footnote keys with empty values
        if footnotes and all(not str(v).strip() for v in footnotes.values()):
            is_incomplete = True
            if not incomplete_reason:
                incomplete_reason = (
                    f"footnote markers {','.join(footnotes.keys())} present "
                    f"but text not found on this page (auto-detected)"
                )

        # Check 2: no caption and no footnotes — almost all screening tables have them
        if not is_incomplete and not caption.strip() and not footnotes:
            if rows and columns:  # has actual data, not just a title stub
                is_incomplete = True
                if not incomplete_reason:
                    incomplete_reason = "no caption/footnotes below table (auto-detected, likely on next page)"

        # Check 3: first entry doesn't start from 1
        if not is_incomplete and rows:
            entry_col = None
            for col in columns:
                if re.match(r'^(entry|no\.?|run|#|\s*$)', col.lower().strip()):
                    entry_col = col
                    break
            if entry_col:
                first_val = str(rows[0].get(entry_col, "")).strip().rstrip("abcdefghij*†‡§")
                if first_val.isdigit() and int(first_val) > 1:
                    is_incomplete = True
                    if not incomplete_reason:
                        incomplete_reason = f"first entry is {first_val}, earlier rows likely on previous page (auto-detected)"

        entry = {
            "image": rel_key,
            "page": page,
            "table_title": t.get("table_title", ""),
            "reaction": t.get("reaction", {}),
            "caption": t.get("caption", ""),
            "footnotes": footnotes,
            "columns": t.get("columns", []),
            "rows": t.get("rows", []),
            "is_incomplete": is_incomplete,
            "incomplete_reason": incomplete_reason,
        }
        existing.append(entry)
        if is_incomplete:
            has_incomplete = True

    out_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))

    # Record incomplete tables separately
    if has_incomplete:
        inc_dir = OUTPUT_DIR / "_incomplete_tables" / journal / doi_dir
        inc_dir.mkdir(parents=True, exist_ok=True)

        # Save the incomplete table JSON
        inc_tables = [t for t in tables if t.get("is_incomplete", False)]
        inc_file = inc_dir / f"{page}.json"
        inc_data = {
            "image": rel_key,
            "page": page,
            "tables": inc_tables,
        }
        inc_file.write_text(json.dumps(inc_data, ensure_ascii=False, indent=2))

        # Symlink the source image for easy access
        if INPUT_DIR:
            src_img = INPUT_DIR / rel_key
            link_path = inc_dir / f"{page}.png"
            if src_img.exists() and not link_path.exists():
                try:
                    link_path.symlink_to(src_img.resolve())
                except OSError:
                    pass

        stats["incomplete"] = stats.get("incomplete", 0) + 1

    progress["done"].append(rel_key)
    stats["recognized"] += 1
    status = f"OK ({len(tables)} table(s), {total_rows} rows)"
    if has_incomplete:
        status += " [INCOMPLETE]"
    return status


def run_offline(todo, args, INPUT_DIR, OUTPUT_DIR, PROGRESS_FILE, progress, done_set):
    """离线模式: 用 vLLM LLM 类直接加载模型推理。"""
    total = len(list(INPUT_DIR.rglob("*.png")))

    print("加载 vLLM 模型...")
    t0 = time.time()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    print(f"模型加载完成，耗时 {time.time()-t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=args.max_tokens,
    )

    stats = {"recognized": 0, "skipped": 0, "failed": 0}
    start_time = time.time()

    for i, img_path in enumerate(todo):
        rel_key = get_relative_key(img_path, INPUT_DIR)
        idx = len(done_set) + i + 1
        elapsed_min = (time.time() - start_time) / 60
        speed = (i) / max(elapsed_min, 0.01) if i > 0 else 0
        remaining = (len(todo) - i) / max(speed, 0.01) if speed > 0 else 0
        print(f"[{idx}/{total}] ({speed:.1f}/min, 剩余~{remaining:.0f}min) {rel_key}", end=" ... ", flush=True)

        try:
            img = Image.open(img_path).convert("RGB")
            prompt_text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
            outputs = llm.generate(
                [{"prompt": prompt_text, "multi_modal_data": {"image": img}}],
                sampling_params=sampling_params,
            )
            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            img.close()
        except Exception as e:
            print(f"FAILED ({e})")
            progress["failed"].append(rel_key)
            stats["failed"] += 1
            if (i + 1) % 10 == 0:
                save_progress(progress, PROGRESS_FILE)
            continue

        status = process_result(text, rel_key, OUTPUT_DIR, progress, stats, INPUT_DIR)
        print(status)

        if (i + 1) % 10 == 0:
            save_progress(progress, PROGRESS_FILE)

    save_progress(progress, PROGRESS_FILE)
    return stats, time.time() - start_time


def run_api(todo, args, INPUT_DIR, OUTPUT_DIR, PROGRESS_FILE, progress, done_set):
    """API 模式: 并发调用已部署的 vLLM OpenAI-compatible API。"""
    import threading

    total = len(list(INPUT_DIR.rglob("*.png")))
    model = resolve_api_model(args.api_url, args.model)
    api_key = resolve_api_key(getattr(args, "api_key", None))
    headers = build_headers(api_key)
    usage_log = getattr(args, "usage_log", None) or DEFAULT_USAGE_LOG
    print(f"API 模式: {args.api_url}")
    print(f"模型: {model}")
    print(f"并发数: {args.concurrency}")
    print(f"Auth: {'Bearer ***' if api_key else '(none)'}")
    print(f"Usage log: {usage_log}")

    stats = {"recognized": 0, "skipped": 0, "failed": 0}
    start_time = time.time()
    lock = threading.Lock()
    counter = [0]  # mutable counter for threads

    def process_one(img_path):
        rel_key = get_relative_key(img_path, INPUT_DIR)
        img_b64 = image_to_b64(img_path, max_side=args.downscale)
        if img_b64 is None:
            with lock:
                progress["failed"].append(rel_key)
                stats["failed"] += 1
                counter[0] += 1
            return rel_key, "FAILED (image read error)"

        try:
            # rel_key looks like "JACS/10.1021__jacs.4c13321_si/p18.png"
            parts = rel_key.split("/")
            doi_safe = parts[1] if len(parts) > 1 else None
            page_str = parts[-1]
            text = call_api(args.api_url, model, img_b64,
                            max_tokens=args.max_tokens, timeout=args.timeout,
                            headers=headers, usage_log=usage_log,
                            doi=doi_safe, page=page_str)
        except Exception as e:
            with lock:
                progress["failed"].append(rel_key)
                stats["failed"] += 1
                counter[0] += 1
            return rel_key, f"FAILED ({e})"

        with lock:
            status = process_result(text, rel_key, OUTPUT_DIR, progress, stats, INPUT_DIR)
            counter[0] += 1
            cnt = counter[0]
            if cnt % 10 == 0:
                save_progress(progress, PROGRESS_FILE)
        return rel_key, status

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process_one, img): img for img in todo}
        for future in as_completed(futures):
            rel_key, status = future.result()
            idx = len(done_set) + counter[0]
            elapsed_min = (time.time() - start_time) / 60
            speed = counter[0] / max(elapsed_min, 0.01)
            remaining = (len(todo) - counter[0]) / max(speed, 0.01) if speed > 0 else 0
            print(f"[{idx}/{total}] ({speed:.1f}/min, 剩余~{remaining:.0f}min) "
                  f"{rel_key} ... {status}")

    save_progress(progress, PROGRESS_FILE)
    return stats, time.time() - start_time


def main():
    parser = argparse.ArgumentParser(description="vLLM 批量识别条件筛选表 (离线/API 两种模式)")
    # 通用参数
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR, help="输入图片目录")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="输出结果目录")
    parser.add_argument("--max-tokens", type=int, default=45500, help="最大生成 token 数")
    parser.add_argument("--split", type=str, default=None, help="分片: 0/2 表示取前半, 1/2 表示取后半")
    # 离线模式参数
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型路径 (离线模式)")
    parser.add_argument("--tp", type=int, default=4, help="tensor parallel size (离线模式)")
    parser.add_argument("--max-model-len", type=int, default=8192, help="模型最大上下文长度 (离线模式)")
    # API 模式参数
    parser.add_argument("--api-url", type=str, default=None,
                        help="vLLM API URL (提供则启用 API 模式，例如 http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (若中转站需要)；也可由 PROXY_API_KEY/OPENAI_API_KEY 环境变量提供")
    parser.add_argument("--usage-log", type=str, default=None,
                        help=f"Token 使用量 JSONL 日志路径 (默认 {DEFAULT_USAGE_LOG})")
    parser.add_argument("--concurrency", type=int, default=256, help="API 并发数 (API 模式)")
    parser.add_argument("--downscale", type=int, default=2048, help="图片最大边长 (API 模式, 0=不缩放)")
    parser.add_argument("--timeout", type=int, default=600, help="API 超时秒数 (API 模式)")
    args = parser.parse_args()

    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    PROGRESS_FILE = OUTPUT_DIR / "_progress.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress(PROGRESS_FILE)
    done_set = set(progress["done"] + progress["skipped"] + progress["failed"])

    all_images = sorted(INPUT_DIR.rglob("*.png"))
    total = len(all_images)
    todo = [img for img in all_images if get_relative_key(img, INPUT_DIR) not in done_set]

    # 分片支持
    if args.split:
        idx, parts = args.split.split("/")
        idx, parts = int(idx), int(parts)
        chunk_size = (len(todo) + parts - 1) // parts
        todo = todo[idx * chunk_size : (idx + 1) * chunk_size]
        print(f"分片模式: {idx}/{parts}, 本进程处理 {len(todo)} 张")

    mode = "API" if args.api_url else "离线"
    print(f"总计: {total} 张图片")
    print(f"已处理: {len(done_set)}")
    print(f"待处理: {len(todo)}")
    print(f"模式: {mode}")
    print("=" * 60)

    if not todo:
        print("全部处理完毕！")
        return

    if args.api_url:
        stats, total_time = run_api(todo, args, INPUT_DIR, OUTPUT_DIR,
                                    PROGRESS_FILE, progress, done_set)
    else:
        print(f"模型: {args.model}")
        print(f"TP: {args.tp}")
        stats, total_time = run_offline(todo, args, INPUT_DIR, OUTPUT_DIR,
                                        PROGRESS_FILE, progress, done_set)

    print("=" * 60)
    print(f"完成! 耗时: {total_time/60:.1f} 分钟")
    print(f"  识别成功: {stats['recognized']}")
    print(f"  不完整表格(跨页): {stats.get('incomplete', 0)}")
    print(f"  跳过(非筛选表): {stats['skipped']}")
    print(f"  失败: {stats['failed']}")
    if stats.get('incomplete', 0) > 0:
        print(f"  不完整表格已记录到: {OUTPUT_DIR}/_incomplete_tables/")
    total_processed = stats['recognized'] + stats['skipped'] + stats['failed']
    if total_time > 0:
        print(f"  平均速度: {total_processed/(total_time/60):.1f} 张/分钟")


if __name__ == "__main__":
    main()
