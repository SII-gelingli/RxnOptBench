#!/usr/bin/env python3
"""
VLM screening: filter screening table pages from rendered 600 DPI images.

Reads all rendered page images from all_pages_hires_2025/, sends to
Kimi VLM API (OpenAI-compatible), and identifies pages containing
condition screening/optimization tables.  Also detects cross-page tables.

Output:
  - scope_screen_results_2025/{journal}_{doi_safe}_p{page}.json
  - scope_confirmed_index_2025.json

Usage:
    # Kimi API (default):
    python pipeline/stage2_screen_vlm.py --api-base https://api.moonshot.cn/v1
    python pipeline/stage2_screen_vlm.py --test 50
    python pipeline/stage2_screen_vlm.py --resume
    python pipeline/stage2_screen_vlm.py --concurrency 64
    python pipeline/stage2_screen_vlm.py --journal JACS

    # Or local vLLM server:
    python pipeline/stage2_screen_vlm.py --api-base http://localhost:8000/v1
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image

# 600 DPI 渲染的页面很大，禁用 PIL 解压炸弹警告
Image.MAX_IMAGE_PIXELS = None

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Error: openai package not found. Install with: pip install openai")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.utils.api_usage import log_usage, DEFAULT_USAGE_LOG  # noqa: E402


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Two image sources: main text and SI
IMAGE_SOURCES = {
    "main": {
        "images_dir": BASE_DIR / "all_pages_hires_2025",
        "results_dir": BASE_DIR / "scope_screen_results_2025",
        "index_file": BASE_DIR / "scope_confirmed_index_2025.json",
        "label": "正文 (main text)",
    },
    "si": {
        "images_dir": BASE_DIR / "all_si_pages_hires_2025",
        "results_dir": BASE_DIR / "scope_screen_results_2025_si",
        "index_file": BASE_DIR / "scope_confirmed_index_2025_si.json",
        "label": "SI (Supporting Information)",
    },
}

# These globals are set per-run by run_one_source()
IMAGES_DIR = IMAGE_SOURCES["main"]["images_dir"]
RESULTS_DIR = IMAGE_SOURCES["main"]["results_dir"]
CONFIRMED_INDEX_FILE = IMAGE_SOURCES["main"]["index_file"]

# ── VLM Prompt ─────────────────────────────────────────────────────────────────
VLM_PROMPT = """Look at this page from a chemistry research paper. Does it contain a reaction condition screening/optimization table?

A condition screening table typically shows:
- A model/benchmark reaction tested under DIFFERENT conditions
- A tabular format with entries (Entry 1, 2, 3...) or rows
- Columns for variables like: catalyst, ligand, solvent, base, additive, temperature, time, etc.
- Each row tests a different combination of conditions on the SAME substrate
- A yield/conversion/selectivity column showing the result for each entry
- The goal is to identify optimal reaction conditions

It is NOT a condition screening table if it's:
- A substrate scope table (same conditions, DIFFERENT substrates with numbered products 2a, 2b, 3a...)
- A figure showing only reaction mechanisms or energy diagrams
- A table of physical/spectral/analytical data only (NMR, MS, IR, crystal data)
- Plain text without tabular data
- A comparison of different literature methods or review table
- A table of substrate/product characterization data

Return ONLY valid JSON (no markdown fences):
{"is_screening_table": true/false, "confidence": "high"/"medium"/"low", "reason": "brief explanation"}"""


def image_to_b64(img_path: str, max_side: int = 0) -> str | None:
    """Read image, optionally downscale, return base64 PNG."""
    try:
        img = Image.open(img_path)
        if max_side > 0:
            w, h = img.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def parse_vlm_output(text: str) -> dict:
    """Parse JSON from VLM output."""
    text = text.strip()
    think_end = text.find("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>"):].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = text.find("{")
        if brace >= 0:
            try:
                return json.loads(text[brace:])
            except json.JSONDecodeError:
                pass

    is_scope = bool(re.search(r'"is_screening_table"\s*:\s*true', text, re.IGNORECASE))
    return {
        "is_screening_table": is_scope,
        "confidence": "low",
        "reason": f"parse_fallback: {text[:200]}",
        "parse_error": True,
    }


async def vlm_call(client, model_name: str, b64_image: str,
                    semaphore: asyncio.Semaphore,
                    max_retries: int = 3,
                    usage_log: str | None = None,
                    doi: str | None = None,
                    page: str | None = None) -> str | None:
    """Send one image to VLM and return the response text. Retries on failure."""
    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                            {"type": "text", "text": VLM_PROMPT},
                        ],
                    }],
                    max_tokens=45500,
                    temperature=1,
                )
                _m = resp.choices[0].message
                if usage_log and getattr(resp, "usage", None) is not None:
                    u = resp.usage
                    log_usage(
                        usage_log, stage="screen_vlm", doi=doi, page=page,
                        model=model_name,
                        usage={
                            "prompt_tokens": getattr(u, "prompt_tokens", None),
                            "completion_tokens": getattr(u, "completion_tokens", None),
                            "total_tokens": getattr(u, "total_tokens", None),
                        },
                        message={
                            "reasoning_content": getattr(_m, "reasoning_content", None) or "",
                            "content": _m.content or "",
                        },
                        elapsed=time.time() - t0,
                    )
                return _m.content
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt  # 2s, 4s, 8s
                    print(f"  [RETRY {attempt}/{max_retries}] {e} — waiting {wait}s")
                    await asyncio.sleep(wait)
                else:
                    print(f"  [API ERROR] {e} (failed after {max_retries} attempts)")
                    return None


async def run_screening(tasks: list, args):
    """Run VLM screening on all rendered page images."""
    api_key = args.api_key or os.environ.get("MOONSHOT_API_KEY", "EMPTY")
    client = AsyncOpenAI(
        base_url=args.api_base,
        api_key=api_key,
        max_retries=3,
        timeout=args.timeout,
    )

    # Detect model name
    try:
        models = await client.models.list()
        model_name = models.data[0].id
    except Exception:
        model_name = args.api_model or "kimi-k2.6"
    print(f"  Model: {model_name}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats = {"confirmed": 0, "rejected": 0, "error": 0, "parse_error": 0}
    semaphore = asyncio.Semaphore(args.concurrency)
    total = len(tasks)
    t0 = time.time()
    done_count = 0
    lock = asyncio.Lock()

    # Dedicated thread pool for image I/O — enough threads to never starve GPU
    io_pool = ThreadPoolExecutor(max_workers=args.io_workers)

    async def process_one(task):
        nonlocal done_count
        journal = task["journal"]
        doi_safe = task["doi_safe"]
        page_num = task["page"]
        img_path = task["img_path"]

        result_key = f"{journal}_{doi_safe}_p{page_num}"
        result_file = RESULTS_DIR / f"{result_key}.json"

        # Read image → base64 on dedicated I/O thread pool
        loop = asyncio.get_event_loop()
        b64 = await loop.run_in_executor(
            io_pool, image_to_b64, img_path, args.downscale
        )

        if b64 is None:
            async with lock:
                stats["error"] += 1
                done_count += 1
            return

        # Call VLM — semaphore controls in-flight requests
        raw_text = await vlm_call(client, model_name, b64, semaphore,
                                  usage_log=getattr(args, "_usage_log_resolved",
                                                    None),
                                  doi=doi_safe.replace("__", "/"),
                                  page=str(page_num))

        if raw_text is None:
            async with lock:
                stats["error"] += 1
                done_count += 1
            return

        vlm_result = parse_vlm_output(raw_text)
        is_confirmed = vlm_result.get("is_screening_table", False)

        vlm_result["_source"] = {
            "journal": journal,
            "doi_safe": doi_safe,
            "doi": doi_safe.replace("__", "/"),
            "page": page_num,
        }
        vlm_result["_meta"] = {"model": model_name, "raw": raw_text[:2000]}

        # Write result to disk (non-blocking)
        await loop.run_in_executor(
            io_pool,
            lambda: result_file.write_text(
                json.dumps(vlm_result, indent=2, ensure_ascii=False)
            ),
        )

        async with lock:
            if is_confirmed:
                stats["confirmed"] += 1
            else:
                stats["rejected"] += 1
            if vlm_result.get("parse_error"):
                stats["parse_error"] += 1
            done_count += 1
            if done_count % 50 == 0 or done_count == total:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"  [{done_count}/{total}] {rate:.1f} pg/s | "
                      f"scope:{stats['confirmed']} other:{stats['rejected']} "
                      f"err:{stats['error']}")

    # Fire all tasks — semaphore + vLLM batching handles the queuing
    await asyncio.gather(*[process_one(t) for t in tasks])

    io_pool.shutdown(wait=False)

    infer_time = time.time() - t0
    print(f"\nVLM screening done: {infer_time:.1f}s ({infer_time/60:.1f} min)")
    print(f"  Scope tables: {stats['confirmed']}")
    print(f"  Other pages: {stats['rejected']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Parse errors: {stats['parse_error']}")
    if infer_time > 0:
        print(f"  Throughput: {total/infer_time:.1f} pg/s")

    return stats, model_name


def build_confirmed_index(model_name: str = ""):
    """Build final confirmed index from VLM results."""
    print("\nBuilding confirmed index...")

    confirmed_papers = {}
    total_confirmed = 0

    for result_file in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(result_file.read_text())
        except Exception:
            continue

        if not data.get("is_screening_table", False):
            continue

        src = data.get("_source", {})
        doi = src.get("doi", "")
        journal = src.get("journal", "")
        doi_safe = src.get("doi_safe", "")
        page_num = src.get("page", 0)

        if not doi:
            continue

        image_rel = f"{journal}/{doi_safe}/p{page_num}.png"
        image_abs = IMAGES_DIR / image_rel

        if doi not in confirmed_papers:
            confirmed_papers[doi] = {
                "doi": doi, "journal": journal, "pages": [],
            }

        confirmed_papers[doi]["pages"].append({
            "page": page_num,
            "vlm_confidence": data.get("confidence", ""),
            "vlm_reason": data.get("reason", ""),
            "image": image_rel if image_abs.exists() else None,
        })
        total_confirmed += 1

    for entry in confirmed_papers.values():
        entry["pages"].sort(key=lambda x: x["page"])

    index = {
        "metadata": {
            "description": "Screening table pages confirmed by VLM (papers_2025)",
            "total_confirmed_pages": total_confirmed,
            "total_papers": len(confirmed_papers),
            "vlm_model": model_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "papers": confirmed_papers,
    }

    with open(CONFIRMED_INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"  Screening tables: {total_confirmed} pages from {len(confirmed_papers)} papers")
    print(f"  Index: {CONFIRMED_INDEX_FILE}")
    return total_confirmed, len(confirmed_papers)


def collect_tasks(images_dir: Path, results_dir: Path,
                   journal_filter: str = None, resume: bool = False,
                   test_n: int = 0) -> list:
    """Collect page image tasks from an images directory."""
    if not images_dir.exists():
        print(f"  Warning: {images_dir} not found, skipping.")
        return []

    tasks = []
    journals = sorted(
        d.name for d in images_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    )

    if journal_filter:
        journals = [j for j in journals if j == journal_filter]

    for journal in journals:
        j_dir = images_dir / journal
        for doi_dir in sorted(j_dir.iterdir()):
            if not doi_dir.is_dir():
                continue
            doi_safe = doi_dir.name
            for img_file in sorted(doi_dir.glob("p*.png")):
                page_str = img_file.stem[1:]
                try:
                    page_num = int(page_str)
                except ValueError:
                    continue

                if resume:
                    result_key = f"{journal}_{doi_safe}_p{page_num}"
                    result_file = results_dir / f"{result_key}.json"
                    if result_file.exists():
                        try:
                            existing = json.loads(result_file.read_text())
                            if "parse_error" not in existing:
                                continue
                        except Exception:
                            pass

                tasks.append({
                    "journal": journal,
                    "doi_safe": doi_safe,
                    "page": page_num,
                    "img_path": str(img_file),
                })

    if test_n > 0:
        tasks = tasks[:test_n]

    return tasks


def run_one_source(source_key: str, args):
    """Run screening for one image source (main or si)."""
    global IMAGES_DIR, RESULTS_DIR, CONFIRMED_INDEX_FILE

    src = IMAGE_SOURCES[source_key]
    IMAGES_DIR = src["images_dir"]
    RESULTS_DIR = src["results_dir"]
    CONFIRMED_INDEX_FILE = src["index_file"]

    print()
    print("=" * 60)
    print(f"  Screening: {src['label']}")
    print(f"  Images: {IMAGES_DIR}")
    print("=" * 60)

    if args.build_index_only:
        build_confirmed_index()
        return

    tasks = collect_tasks(
        IMAGES_DIR, RESULTS_DIR,
        journal_filter=args.journal,
        resume=args.resume,
        test_n=args.test,
    )

    total = len(tasks)
    if total == 0:
        print("  Nothing to process.")
        build_confirmed_index()
        return

    total_dois = len(set(t["doi_safe"] for t in tasks))
    journals = sorted(set(t["journal"] for t in tasks))
    print(f"  Pages: {total} from {total_dois} papers")
    print(f"  Journals: {', '.join(journals)}")
    print(f"  API: {args.api_base}")
    print(f"  Concurrency: {args.concurrency}")
    print()

    stats, model_name = asyncio.run(run_screening(tasks, args))
    build_confirmed_index(model_name)

    print()
    print(f"  [{src['label']}] COMPLETE")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Index: {CONFIRMED_INDEX_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Screen screening tables from rendered page images (Kimi VLM)"
    )
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: only process N pages per source")
    parser.add_argument("--journal", type=str, default=None,
                        help="Process only this journal")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip already-screened pages (default: True)")
    parser.add_argument("--concurrency", type=int, default=256,
                        help="Max concurrent VLM API requests (default: 256)")
    parser.add_argument("--io-workers", type=int, default=64,
                        help="Thread pool size for image I/O (default: 64)")
    parser.add_argument("--api-base", type=str,
                        default="http://localhost:8000/v1",
                        help="Kimi API base URL")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (default: env MOONSHOT_API_KEY, or 'EMPTY' for local)")
    parser.add_argument("--usage-log", type=str, default=None,
                        help=f"Token 用量 JSONL 日志 (默认 {DEFAULT_USAGE_LOG})")
    parser.add_argument("--api-model", type=str, default=None,
                        help="Model name (auto-detected if omitted)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="API timeout in seconds (default: 600)")
    parser.add_argument("--downscale", type=int, default=2048,
                        help="Downscale image max side for VLM input (0=no, default 2048)")
    parser.add_argument("--build-index-only", action="store_true",
                        help="Only rebuild confirmed index from existing results")
    parser.add_argument("--source", type=str, default="all",
                        choices=["all", "main", "si"],
                        help="Which image source to screen: 'all' (default) runs both main+SI, "
                             "'main' runs main text only, 'si' runs SI only")
    args = parser.parse_args()

    if args.source == "all":
        sources = ["main", "si"]
    else:
        sources = [args.source]

    # Resolve usage log path once (passed into vlm_call via args)
    args._usage_log_resolved = args.usage_log or DEFAULT_USAGE_LOG
    print(f"Usage log: {args._usage_log_resolved}")

    for src_key in sources:
        run_one_source(src_key, args)

    print()
    print("=" * 60)
    print("ALL SCREENING COMPLETE")
    print("=" * 60)
    for src_key in sources:
        src = IMAGE_SOURCES[src_key]
        print(f"  {src['label']}:")
        print(f"    Results: {src['results_dir']}")
        print(f"    Index:   {src['index_file']}")


if __name__ == "__main__":
    main()
