#!/usr/bin/env python3
"""
Stage 10: Per-row reaction expansion & correction using Kimi-K2.6 VLM.

For each row in a screening table, sends the original SI page image + stage9 JSON
to the VLM, which:
  1. Corrects errors in the JSON by cross-referencing with the original image
  2. Expands that row into a complete reaction (resolves footnotes, fills conditions)

Usage:
    python pipeline/stage10_expand_rows.py --test 3
    python pipeline/stage10_expand_rows.py --concurrency 4 --resume
    python pipeline/stage10_expand_rows.py --api-url http://localhost:8000/v1/chat/completions
"""

import argparse
import base64
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.utils.api_usage import (  # noqa: E402
    resolve_api_key, build_headers, log_usage, DEFAULT_USAGE_LOG,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "screening_tables_final"
DEFAULT_OUTPUT = BASE_DIR / "screening_tables_expanded"
DEFAULT_MOL_LABELS = BASE_DIR / "mol_labels_screening"
DEFAULT_IMAGE_DIRS = [
    BASE_DIR / "all_pages_hires_2025",
    BASE_DIR / "all_si_pages_hires_2025",
    BASE_DIR / "screening_tables_qwen72b",
    BASE_DIR / "screening_tables_si",
]

# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are an expert at parsing organic chemistry screening/optimization tables from paper SI images. You always respond with valid JSON only — no explanations, no reasoning, no markdown."

ROW_PROMPT_TEMPLATE = """\
## Input
1. Original page image(s) (authoritative source — may include adjacent pages for cross-page tables)
2. Auto-parsed JSON (reference only, may contain errors){cross_page_note}

NOTE: If multiple page images are provided, they are for completing a cross-page table. Only use data from the TABLE specified in the JSON below — do NOT mix in data from other tables that may appear on adjacent pages.

## Auto-parsed JSON
```json
{table_json}
```

## Task
For row {row_index} ({row_summary}), produce a complete expanded reaction by:

Step 1 — Verify against the original image and correct any errors:
- SMILES RULE: DO NOT generate, modify, "correct", fill in, or replace ANY SMILES string under any circumstance. SMILES are provided externally by a dedicated OCSR model and MUST NOT be touched.
  * If a smiles field is non-empty, copy it verbatim to the output (including any "*" R-group placeholders).
  * If a smiles field is empty (""), leave it empty ("") — do NOT attempt to fill it even for common reagents.
  * If a SMILES contains "*" (R-group placeholder), keep the "*" in the smiles field verbatim.
    ADDITIONALLY populate a per-mol "r_groups" object mapping each R-group
    identifier to the SMILES fragment of its substituent FOR THIS SPECIFIC ROW,
    read from the row's substituent column(s). Each value must be a valid SMILES
    fragment representing ONLY the substituent attached at the * position — do
    NOT include the "*" itself.
    Examples:
      - Scaffold "CC(=O)c1ccc(*)cc1" in a row where R=Cl → "r_groups": {{"R": "Cl"}}
      - Scaffold "[*:1]c1ccc([*:2])cc1" in a row where R1=Me, R2=OMe
        → "r_groups": {{"R1": "C", "R2": "OC"}}
      - For bidentate/symmetric scaffolds with multiple bare "*" where the row
        specifies a single substituent class, use one "R" entry applied to all
        dummies, e.g. "r_groups": {{"R": "c1ccccc1"}}.
    Rules:
      * Keys: use "R", "R1", "R2", etc. matching the scheme figure or footnotes.
      * Values: plain SMILES of the substituent fragment (omit when unsure).
      * If you cannot confidently identify the substituent from THIS row, either
        omit "r_groups" entirely or set its value to "". NEVER invent.
  * When this row uses a different substrate/product than the top-level reaction, still do NOT synthesize a new SMILES; keep smiles as "" for that slot. OCSR backfill will handle it.
- Table cell values: do they match what is printed in the image for this row?
- Footnotes: are they complete and accurate?
- Conditions on/around the reaction arrow: catalyst, solvent, temperature, etc.

Step 2 — Expand this row into a complete standalone reaction:
- Replace all footnote references (e.g. "a", "b") with their actual content
- Fill in the variable conditions from this row (catalyst, solvent, ligand, temperature, equivalents, etc.)
- Keep yield, ee, dr and other result data. IMPORTANT: when there are multiple products, report yield/ee/dr for EACH product individually in its product object (not as a single top-level field)
- IMPORTANT: The JSON contains a "label_smiles_map" section mapping compound labels to their SMILES. When a condition references a compound by label (e.g. "4CzIPN", "L1", "Cat-1"), look up its SMILES in label_smiles_map and include it in the condition string as "[SMILES: ...]". For example: "4CzIPN (1 mol%) [SMILES: N#Cc1c(-n2c3ccccc3c3ccccc32)...]". This applies to catalysts, ligands, photocatalysts, additives, and any other reagent that has a known SMILES in label_smiles_map.
- REMINDER: You MUST NOT invent SMILES for conditions either — only use SMILES that already exist in label_smiles_map. If a reagent has no SMILES in label_smiles_map, write the condition without a [SMILES: ...] tag.

## CRITICAL: Output ONLY the JSON below. No reasoning, no explanation, no markdown fences.
{{
  "row_index": {row_index},
  "corrections": ["description of correction 1", ...],
  "reaction": {{
    "substrates": [{{"label": "1", "smiles": "...", "description": "...", "r_groups": {{}}}}],
    "products": [{{"label": "2", "smiles": "...", "description": "...", "yield": "xx%", "ee": null, "dr": null, "r_groups": {{}}}}],
    "conditions": ["full condition 1 (e.g. fac-Ir(ppy)3 (1.5 mol%))", "full condition 2", ...],
    "notes": "any notes for this row"
  }}
}}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def find_page_image(journal, doi_dir, page_key):
    """Find original SI page image (no annotations)."""
    for d in DEFAULT_IMAGE_DIRS:
        p = d / journal / doi_dir / f"{page_key}.png"
        if p.exists():
            return p
    return None


def find_adjacent_images(journal, doi_dir, page_key):
    """Find previous and next page images for cross-page tables."""
    page_num = page_key.lstrip("p")
    if not page_num.isdigit():
        return []
    page_num = int(page_num)
    adjacent = []
    for offset in (-1, 1):
        adj_key = f"p{page_num + offset}"
        img = find_page_image(journal, doi_dir, adj_key)
        if img:
            adjacent.append((adj_key, img))
    return adjacent


def image_to_b64(img_path, max_side=2048):
    """Encode image as base64 PNG, optionally downscaling."""
    img = Image.open(str(img_path))
    w, h = img.size
    if max_side > 0 and max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def row_summary(row, columns):
    """One-line summary of a row for the prompt."""
    parts = []
    for col in columns[:6]:
        val = str(row.get(col, ""))
        if val:
            parts.append(f"{col}={val}")
    return ", ".join(parts)


def build_prompt(table_json, row_index, cross_page: bool = False):
    """Build the per-row prompt text.

    When `cross_page` is True (i.e. stage4b stitched this table from adjacent
    pages and stage10 is passing the neighbour images alongside the main page),
    prepend an explicit instruction so the VLM treats them as context rather
    than trying to "fix" the row count from the main page alone.
    """
    rows = table_json.get("rows", [])
    columns = table_json.get("columns", [])
    row = rows[row_index] if row_index < len(rows) else {}
    rs = row_summary(row, columns)

    # Compact JSON for prompt (no unnecessary whitespace)
    tj = json.dumps(table_json, ensure_ascii=False, indent=2)

    cross_page_note = (
        "\n\nIMPORTANT: This table was stitched from adjacent pages. "
        "The first/last image(s) are the previous/next page provided ONLY as "
        "context for headers, continuation rows, or footnotes that were split "
        "across pages. Do NOT invent or add rows that are not present in the "
        "auto-parsed JSON, and do NOT drop rows because their entry numbers "
        "do not start from 1 on the current page."
        if cross_page else ""
    )

    return ROW_PROMPT_TEMPLATE.format(
        table_json=tj,
        row_index=row_index,
        row_summary=rs,
        cross_page_note=cross_page_note,
    )


def parse_response(text):
    """Parse JSON from model response, handling various output formats."""
    if not text:
        return None
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Model may have dumped reasoning before the JSON — find the last { ... } block
    # that contains "row_index" (our expected key)
    matches = list(re.finditer(r'\{[\s\S]*\}', text))
    for match in reversed(matches):
        candidate = match.group()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and ("row_index" in obj or "reaction" in obj):
                return obj
        except json.JSONDecodeError:
            continue

    # Fallback: try any JSON object
    for match in reversed(matches):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None


# ── API ─────────────────────────────────────────────────────────────────────

_cached_model = None


def resolve_model(api_url, model_name=None):
    """Auto-detect model name from vLLM /v1/models endpoint."""
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


def call_api(api_url, model, image_b64, table_json, row_index,
             max_tokens=45500, timeout=600, extra_images=None,
             headers=None, usage_log=None, doi=None, page=None):
    """Send image(s) + table JSON + row index to VLM, return (parsed_result, raw_text).

    Args:
        extra_images: list of (page_label, b64_str) for adjacent pages (cross-page tables)
    """
    prompt_text = build_prompt(table_json, row_index,
                               cross_page=bool(extra_images))

    content_parts = []
    # Add adjacent previous page first (if any)
    if extra_images:
        for label, b64 in extra_images:
            if label.startswith("prev") or (label.lstrip("p").isdigit() and
                    int(label.lstrip("p")) < int(table_json.get("page", 0))):
                content_parts.append({"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}})
    # Main page image
    content_parts.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
    })
    # Add adjacent next page (if any)
    if extra_images:
        for label, b64 in extra_images:
            if not (label.startswith("prev") or (label.lstrip("p").isdigit() and
                    int(label.lstrip("p")) < int(table_json.get("page", 0)))):
                content_parts.append({"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content_parts.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    t0 = time.time()
    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp_json = resp.json()
    msg = resp_json["choices"][0]["message"]
    if usage_log:
        log_usage(usage_log, stage="expand_rows", doi=doi, page=page,
                  model=model, usage=resp_json.get("usage"),
                  message=msg, elapsed=time.time() - t0)
    text = msg.get("content") or ""
    return parse_response(text), text


# ── Progress tracking ───────────────────────────────────────────────────────

def load_progress(progress_file):
    if progress_file.exists():
        try:
            return json.loads(progress_file.read_text())
        except Exception:
            pass
    return {"done": [], "failed": []}


def save_progress(progress, progress_file):
    progress_file.write_text(json.dumps(progress, ensure_ascii=False, indent=2))


# ── Task collection ─────────────────────────────────────────────────────────

def collect_tasks(input_dir, image_dirs, journal_filter=None, test_n=0):
    """Collect (page_json_path, image_path, rel_path) tuples."""
    tasks = []
    input_dir = Path(input_dir)
    for jf in sorted(input_dir.rglob("*.json")):
        if jf.name.startswith("_"):
            continue
        rel = jf.relative_to(input_dir)
        parts = rel.parts
        if len(parts) != 3:
            continue
        journal, doi_dir, page_file = parts
        if journal_filter and journal not in journal_filter.split(","):
            continue
        page_key = page_file.replace(".json", "")

        # Find image
        img_path = None
        for d in image_dirs:
            p = d / journal / doi_dir / f"{page_key}.png"
            if p.exists():
                img_path = p
                break
        if not img_path:
            continue

        tasks.append((jf, img_path, str(rel), journal, doi_dir, page_key))
        if test_n > 0 and len(tasks) >= test_n:
            break
    return tasks


# ── OCSR SMILES helpers ─────────────────────────────────────────────────────

def load_ocsr_smiles(mol_labels_dir, journal, doi_dir, page_key):
    """Load OCSR SMILES from mol_labels_screening for a page."""
    if mol_labels_dir is None:
        return {}
    jf = mol_labels_dir / journal / doi_dir / f"{page_key}.json"
    if not jf.exists():
        return {}
    try:
        data = json.load(open(jf))
    except Exception:
        return {}
    label_map = {}
    from collections import defaultdict
    label_smiles = defaultdict(set)
    for mol in data.get("molecules", []):
        label = mol.get("compound_label")
        if not label:
            continue
        if not mol.get("rdkit_valid"):
            continue
        smi = mol.get("smiles", "")
        if not smi or "*" in smi:
            continue
        label_smiles[label].add(smi)
    for label, smiles_set in label_smiles.items():
        if len(smiles_set) == 1:
            label_map[label] = next(iter(smiles_set))
    return label_map


_doi_smiles_cache = {}


def load_doi_smiles(mol_labels_dir, journal, doi_dir):
    """Load OCSR SMILES from ALL pages of a DOI across mol_labels_screening dirs."""
    if mol_labels_dir is None:
        return {}
    cache_key = (str(mol_labels_dir), journal, doi_dir)
    if cache_key in _doi_smiles_cache:
        return _doi_smiles_cache[cache_key]
    from collections import defaultdict
    label_smiles = defaultdict(set)
    base = mol_labels_dir.parent
    search_dirs = [mol_labels_dir, base / "mol_labels_screening_si"]
    for d in search_dirs:
        doi_path = d / journal / doi_dir
        if not doi_path.exists():
            continue
        for jf in doi_path.glob("*.json"):
            try:
                data = json.load(open(jf))
            except Exception:
                continue
            for mol in data.get("molecules", []):
                label = mol.get("compound_label")
                if not label or not mol.get("rdkit_valid"):
                    continue
                smi = mol.get("smiles", "")
                if smi and "*" not in smi:
                    label_smiles[label].add(smi)
    result = {}
    for label, smiles_set in label_smiles.items():
        if len(smiles_set) == 1:
            result[label] = next(iter(smiles_set))
    _doi_smiles_cache[cache_key] = result
    return result


# ── R-group substitution ────────────────────────────────────────────────────

def try_r_group_substitution(smiles_with_star, r_groups):
    """Attempt to substitute "*" (R-group placeholders) in an OCSR SMILES
    with fragment SMILES provided by the VLM per-row.

    Handles three common cases:
      1. Single dummy (bare "*" or labeled [*:1]/[1*]): uses the labeled entry
         if present, else the "R" key, else the first available value.
      2. Multiple labeled dummies ([*:1]/[*:2] or [1*]/[2*]): each matched to
         its own r_groups key (R1, R2, ...). Every dummy must have a match.
      3. Multiple bare dummies with a single uniform R: substitute all dummies
         with the same fragment. Ambiguous multi-bare cases (len(r_groups)>1)
         are skipped to avoid guessing.

    Returns a canonical SMILES string with no remaining "*", or None on failure.
    """
    if not smiles_with_star or "*" not in smiles_with_star:
        return None
    if not isinstance(r_groups, dict) or not r_groups:
        return None

    try:
        from rdkit import Chem
    except ImportError:
        return None

    # Normalize r_groups keys (strip non-alnum, upper; accept "R"/"R1"/"1" etc.)
    norm_rg = {}
    for k, v in r_groups.items():
        if not isinstance(v, str) or not v.strip():
            continue
        kn = re.sub(r'[^A-Za-z0-9]', '', str(k)).upper()
        if not kn:
            continue
        if kn.isdigit():
            kn = f"R{kn}"
        elif not kn.startswith("R"):
            # allow e.g. "X", "Y" but prefer R-prefixed
            pass
        norm_rg[kn] = v.strip()
    if not norm_rg:
        return None

    mol0 = Chem.MolFromSmiles(smiles_with_star)
    if mol0 is None:
        return None

    dummies = [a for a in mol0.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummies:
        return None

    # Determine each dummy's identifier (map num or isotope); None for bare.
    dummy_labels = []
    for a in dummies:
        mn = a.GetAtomMapNum()
        iso = a.GetIsotope()
        if mn > 0:
            dummy_labels.append(f"R{mn}")
        elif iso > 0:
            dummy_labels.append(f"R{iso}")
        else:
            dummy_labels.append(None)

    n = len(dummies)
    replacements = {}  # atom_idx -> fragment SMILES

    if n == 1:
        lbl = dummy_labels[0]
        r_smi = norm_rg.get(lbl) if lbl else None
        if r_smi is None:
            r_smi = norm_rg.get("R") or next(iter(norm_rg.values()))
        replacements[dummies[0].GetIdx()] = r_smi
    elif all(l is not None for l in dummy_labels):
        for d, lbl in zip(dummies, dummy_labels):
            if lbl not in norm_rg:
                return None
            replacements[d.GetIdx()] = norm_rg[lbl]
    else:
        # Multiple bare dummies: require uniform R
        if len(norm_rg) != 1:
            return None
        r_smi = next(iter(norm_rg.values()))
        for d in dummies:
            replacements[d.GetIdx()] = r_smi

    # Tag each target dummy with a unique high isotope, replace via SMARTS.
    rw = Chem.RWMol(mol0)
    tag_base = 9000
    tag_to_frag = {}
    for idx, r_smi in replacements.items():
        tag_base += 1
        a = rw.GetAtomWithIdx(idx)
        a.SetIsotope(tag_base)
        a.SetAtomMapNum(0)
        tag_to_frag[tag_base] = r_smi

    current = rw.GetMol()
    for tag, r_smi in tag_to_frag.items():
        r_mol = Chem.MolFromSmiles(r_smi)
        if r_mol is None:
            return None
        patt = Chem.MolFromSmarts(f"[{tag}*]")
        if patt is None:
            return None
        news = Chem.ReplaceSubstructs(current, patt, r_mol, replaceAll=True)
        if not news:
            return None
        current = news[0]

    try:
        result = Chem.MolToSmiles(current)
    except Exception:
        return None
    if not result or "*" in result:
        return None
    if Chem.MolFromSmiles(result) is None:
        return None
    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 10: Per-row reaction expansion & correction (Kimi VLM)")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions",
                        help="VLM chat/completions endpoint")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: env PROXY_API_KEY / MOONSHOT_API_KEY)")
    parser.add_argument("--usage-log", default=None,
                        help=f"Token 用量 JSONL 日志 (默认 {DEFAULT_USAGE_LOG})")
    parser.add_argument("--api-model", default=None,
                        help="Model name (auto-detected if omitted)")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--image-dir", nargs="*", default=None)
    parser.add_argument("--mol-labels-dir", default=str(DEFAULT_MOL_LABELS),
                        help="OCSR mol_labels_screening dir for higher-accuracy SMILES")
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=45500,
                        help="Max output tokens (will be clamped to fit context window)")
    parser.add_argument("--max-side", type=int, default=2048,
                        help="Max image side length (0=no resize)")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--journal", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_dirs = [Path(d) for d in args.image_dir] if args.image_dir \
        else list(DEFAULT_IMAGE_DIRS)
    mol_labels_dir = Path(args.mol_labels_dir) if args.mol_labels_dir else None
    if mol_labels_dir and not mol_labels_dir.exists():
        print(f"  Warning: mol_labels_dir not found: {mol_labels_dir}", flush=True)
        mol_labels_dir = None

    if args.clean:
        import shutil
        if output_dir.exists():
            print(f"Cleaning {output_dir} ...", flush=True)
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / "_expand_progress.json"
    if args.clean and progress_file.exists():
        progress_file.unlink()
    progress = load_progress(progress_file)
    if not args.resume:
        progress["done"] = []
        save_progress(progress, progress_file)
    done_set = set(progress["done"])

    # Resolve model + auth + usage log
    model = resolve_model(args.api_url, args.api_model)
    api_key = resolve_api_key(args.api_key)
    headers = build_headers(api_key)
    usage_log = args.usage_log or DEFAULT_USAGE_LOG
    print(f"Auth: {'Bearer ***' if api_key else '(none)'}", flush=True)
    print(f"Usage log: {usage_log}", flush=True)

    # Collect page tasks
    print("Collecting tasks...", flush=True)
    page_tasks = collect_tasks(input_dir, image_dirs, args.journal, args.test)
    print(f"Found {len(page_tasks)} pages with images", flush=True)

    # Expand to row-level tasks
    row_tasks = []  # (page_json_path, img_path, rel, table_idx, row_idx, table, task_key, journal, doi_dir, page_key)
    for jf, img_path, rel, journal, doi_dir, page_key in page_tasks:
        try:
            tables = json.loads(jf.read_text())
            if not isinstance(tables, list):
                tables = [tables]
        except Exception:
            continue
        for tidx, table in enumerate(tables):
            rows = table.get("rows", [])
            for ridx in range(len(rows)):
                task_key = f"{rel}::{tidx}::row{ridx}"
                if task_key not in done_set:
                    row_tasks.append((jf, img_path, rel, tidx, ridx, table, task_key,
                                      journal, doi_dir, page_key))

    total = len(row_tasks)
    print(f"Row tasks: {total} (skipping {len(done_set)} already done)", flush=True)
    print(f"API: {args.api_url}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Concurrency: {args.concurrency}", flush=True)
    print("=" * 60, flush=True)

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {total} row tasks across {len(page_tasks)} pages")
        for t in row_tasks[:10]:
            print(f"  {t[6]}")
        if total > 10:
            print(f"  ... and {total - 10} more")
        return

    if not row_tasks:
        print("Nothing to do.", flush=True)
        return

    # Lazy image encoding with LRU cache (avoid loading all 3000 images upfront)
    from functools import lru_cache

    @lru_cache(maxsize=256)
    def get_image_b64(img_path_str):
        return image_to_b64(img_path_str, args.max_side)

    unique_imgs = len({str(t[1]) for t in row_tasks})
    print(f"Images: {unique_imgs} unique (lazy-loaded with LRU cache)", flush=True)

    # Pre-compute per-page row counts for incremental flushing
    tasks_per_file = {}   # rel -> total row count (including already done)
    done_per_file = {}    # rel -> completed count this run
    for jf, img_path, rel, journal, doi_dir, page_key in page_tasks:
        try:
            tables = json.loads(jf.read_text())
            if not isinstance(tables, list):
                tables = [tables]
        except Exception:
            continue
        n = sum(len(t.get("rows", [])) for t in tables)
        tasks_per_file[rel] = n
        # Count previously done rows for this file
        already = sum(1 for k in done_set if k.startswith(rel + "::"))
        done_per_file[rel] = already

    # Results accumulator: {rel -> {tidx -> {ridx -> result}}}
    _results = {}
    _results_lock = threading.Lock()

    stats = {"ok": 0, "failed": 0, "files_written": 0}
    start_time = time.time()

    def flush_file(rel):
        """Assemble and write output for a completed page."""
        src_file = input_dir / rel
        try:
            tables = json.loads(src_file.read_text())
            if not isinstance(tables, list):
                tables = [tables]
        except Exception:
            return

        table_results = _results.get(rel, {})

        # Load OCSR SMILES from mol_labels_screening (higher accuracy than VLM)
        rel_parts = Path(rel).parts
        if len(rel_parts) == 3:
            journal_r, doi_r, page_file_r = rel_parts
            page_key_r = page_file_r.replace(".json", "")
            ocsr_smiles = load_ocsr_smiles(mol_labels_dir, journal_r, doi_r, page_key_r)
            doi_smiles = load_doi_smiles(mol_labels_dir, journal_r, doi_r)
        else:
            ocsr_smiles = {}
            doi_smiles = {}

        # DOI-level label_smiles_map from other pages in screening_tables_final
        doi_final_smiles = {}
        if len(rel_parts) == 3:
            doi_final_dir = input_dir / journal_r / doi_r
            if doi_final_dir.exists():
                for _pf in doi_final_dir.glob("*.json"):
                    try:
                        _ptables = json.loads(_pf.read_text())
                        if not isinstance(_ptables, list):
                            _ptables = [_ptables]
                    except Exception:
                        continue
                    for _pt in _ptables:
                        for _lbl, _info in _pt.get("label_smiles_map", {}).items():
                            _smi = _info.get("smiles", "") if isinstance(_info, dict) else ""
                            if _lbl and _smi and _lbl not in doi_final_smiles:
                                doi_final_smiles[_lbl] = _smi
                        for _role in ("substrates", "products", "catalysts", "ligands"):
                            for _mol in _pt.get("reaction", {}).get(_role, []):
                                _lbl = _mol.get("label", "")
                                _smi = _mol.get("smiles", "")
                                if _lbl and _smi and _lbl not in doi_final_smiles:
                                    doi_final_smiles[_lbl] = _smi

        # Load existing output file to merge with previous expanded_rows
        out_file = output_dir / rel
        existing_expanded = {}  # tidx -> {ridx -> er}
        if out_file.exists():
            try:
                existing_tables = json.loads(out_file.read_text())
                if not isinstance(existing_tables, list):
                    existing_tables = [existing_tables]
                for etidx, etbl in enumerate(existing_tables):
                    for er in etbl.get("expanded_rows", []):
                        ri = er.get("row_index")
                        if ri is not None:
                            existing_expanded.setdefault(etidx, {})[ri] = er
            except Exception:
                pass

        out_tables = []
        for tidx, table in enumerate(tables):
            out_table = dict(table)
            row_results = table_results.get(tidx, {})

            # Priority 1: per-slot SMILES from stage9's table.reaction. This
            # includes Stage 8b per-table box alignment (authoritative at the
            # slot level). Stage 9 already strips stage4 VLM-generated SMILES,
            # so every SMILES in table.reaction.*.smiles is OCSR-sourced
            # (smiles_source in {"ocsr_label", "ocsr_table_aligned", ...}).
            # For labels appearing in multiple slots, 8b-aligned entries
            # (smiles_source == "ocsr_table_aligned") win over label-only OCSR.
            table_slot_smiles = {}  # label -> smiles
            _slot_src = {}          # label -> smiles_source (for priority tiebreak)
            for role in ("substrates", "products", "catalysts", "ligands"):
                for item in table.get("reaction", {}).get(role, []):
                    if not isinstance(item, dict):
                        continue
                    lbl = item.get("label", "")
                    smi = item.get("smiles", "")
                    src = item.get("smiles_source", "")
                    if not lbl or not smi or re.match(r'^\d+$', smi):
                        continue
                    # 8b slot alignment overrides a previously-seen label-only entry
                    if lbl not in table_slot_smiles or (
                        src == "ocsr_table_aligned"
                        and _slot_src.get(lbl) != "ocsr_table_aligned"
                    ):
                        table_slot_smiles[lbl] = smi
                        _slot_src[lbl] = src

            # Priority 2: label-level OCSR from stage9's label_smiles_map.
            # Used when a label is not present in table.reaction's slots (e.g.
            # compounds referenced only in rows or footnotes).
            orig_smiles = {}  # label -> smiles
            for lbl, info in table.get("label_smiles_map", {}).items():
                smi = info.get("smiles", "") if isinstance(info, dict) else ""
                if lbl and smi and not re.match(r'^\d+$', smi):
                    orig_smiles[lbl] = smi

            # Merge new results with existing expanded_rows
            merged_rows = dict(existing_expanded.get(tidx, {}))  # ridx -> er (from prev run)
            for ridx, er in row_results.items():
                merged_rows[ridx] = er  # new results override old

            # Build expanded_rows list, forcing SMILES to come from OCSR only.
            # Any SMILES the VLM may have produced is discarded unless it can be
            # re-sourced from OCSR lookups (stage9 label_smiles_map / stage7 OCSR
            # current page / same-DOI cross-page OCSR). Labels with no OCSR match
            # have their smiles cleared.
            expanded = []
            for ridx in sorted(merged_rows.keys()):
                er = merged_rows[ridx]
                er_rxn = er.get("reaction", {})
                for role_key in ("substrates", "products", "catalysts", "ligands"):
                    for mol in er_rxn.get(role_key, []):
                        lbl = mol.get("label", "")
                        # Always strip VLM-produced smiles first; only OCSR sources
                        # below are allowed to set it back.
                        mol.pop("smiles", None)
                        mol.pop("smiles_source", None)
                        if not lbl:
                            # No label → cannot verify via OCSR → leave smiles unset
                            mol["smiles"] = ""
                            mol["smiles_source"] = "none"
                            continue
                        # Priority 1: stage9 table.reaction slot (includes 8b per-slot alignment)
                        if lbl in table_slot_smiles:
                            mol["smiles"] = table_slot_smiles[lbl]
                            mol["smiles_source"] = "table_slot_aligned"
                            continue
                        # Priority 2: stage9 label_smiles_map (OCSR-backed on same page)
                        if lbl in orig_smiles:
                            mol["smiles"] = orig_smiles[lbl]
                            mol["smiles_source"] = "label_smiles_map"
                            continue
                        # Priority 3: OCSR SMILES from current page
                        if lbl in ocsr_smiles:
                            mol["smiles"] = ocsr_smiles[lbl]
                            mol["smiles_source"] = "ocsr"
                            continue
                        # Priority 4: OCSR SMILES from other pages in same DOI
                        if lbl in doi_smiles:
                            mol["smiles"] = doi_smiles[lbl]
                            mol["smiles_source"] = "ocsr_doi"
                            continue
                        # Priority 5: label_smiles_map from other pages in same DOI
                        if lbl in doi_final_smiles:
                            mol["smiles"] = doi_final_smiles[lbl]
                            mol["smiles_source"] = "stage9_doi"
                            continue
                        # No OCSR source available — leave smiles empty
                        mol["smiles"] = ""
                        mol["smiles_source"] = "none"

                # R-group substitution pass: if any mol's OCSR SMILES still
                # contains "*" placeholders and the VLM provided a per-row
                # r_groups map, attempt to substitute "*" with the row-specific
                # substituent fragments. Keep the original SMILES as
                # smiles_template for traceability on success; on failure, keep
                # the SMILES untouched (star preserved). VLM-provided r_groups
                # fragments are validated via RDKit; invalid fragments abort.
                for role_key in ("substrates", "products", "catalysts", "ligands"):
                    for mol in er_rxn.get(role_key, []):
                        if not isinstance(mol, dict):
                            continue
                        smi = mol.get("smiles", "")
                        if not smi or "*" not in smi:
                            continue
                        rg = mol.get("r_groups")
                        if not isinstance(rg, dict) or not rg:
                            continue
                        new_smi = try_r_group_substitution(smi, rg)
                        if not new_smi:
                            continue
                        mol["smiles_template"] = smi
                        mol["smiles"] = new_smi
                        mol["rdkit_valid"] = True
                        src = mol.get("smiles_source", "")
                        mol["smiles_source"] = (
                            f"{src}+r_subst" if src and src != "none"
                            else "r_subst"
                        )
                        mol["r_groups_applied"] = True

                # Also backfill SMILES into structured conditions — strip any
                # VLM-produced smiles first, then re-fill only from OCSR sources.
                conditions = er_rxn.get("conditions", [])
                for ci, cond in enumerate(conditions):
                    if isinstance(cond, dict):
                        cond.pop("smiles", None)
                        cond.pop("smiles_source", None)
                        lbl = cond.get("label", "")
                        if not lbl:
                            continue
                        if lbl in table_slot_smiles:
                            cond["smiles"] = table_slot_smiles[lbl]
                            cond["smiles_source"] = "table_slot_aligned"
                        elif lbl in orig_smiles:
                            cond["smiles"] = orig_smiles[lbl]
                            cond["smiles_source"] = "label_smiles_map"
                        elif lbl in ocsr_smiles:
                            cond["smiles"] = ocsr_smiles[lbl]
                            cond["smiles_source"] = "ocsr"
                        elif lbl in doi_smiles:
                            cond["smiles"] = doi_smiles[lbl]
                            cond["smiles_source"] = "ocsr_doi"
                        elif lbl in doi_final_smiles:
                            cond["smiles"] = doi_final_smiles[lbl]
                            cond["smiles_source"] = "stage9_doi"
                expanded.append(er)
            out_table["expanded_rows"] = expanded

            # Collect table-level corrections from row results
            all_corrections = []
            for r in expanded:
                corr = r.get("corrections", [])
                if corr:
                    all_corrections.extend(corr)
            seen = set()
            unique_corrections = []
            for c in all_corrections:
                if c not in seen:
                    seen.add(c)
                    unique_corrections.append(c)
            out_table["corrections"] = unique_corrections

            # Update metadata
            meta = out_table.get("metadata", {})
            meta["expansion_model"] = model
            meta["expansion_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            out_table["metadata"] = meta

            out_tables.append(out_table)

        out_file = output_dir / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(out_tables, ensure_ascii=False, indent=2))
        stats["files_written"] += 1
        print(f"  >> Wrote {rel} ({stats['files_written']} files)", flush=True)

    def process_one(task):
        jf, img_path, rel, tidx, ridx, table, task_key, journal, doi_dir, page_key = task
        try:
            img_b64 = get_image_b64(str(img_path))
            # For cross-page tables, include adjacent page images
            extra_images = None
            if table.get("completed_from_adjacent"):
                adj = find_adjacent_images(journal, doi_dir, page_key)
                if adj:
                    extra_images = [(ak, get_image_b64(str(ap))) for ak, ap in adj]
            result, raw_text = call_api(args.api_url, model, img_b64, table, ridx,
                                        args.max_tokens, args.timeout,
                                        extra_images=extra_images,
                                        headers=headers, usage_log=usage_log,
                                        doi=f"{journal}/{doi_dir}",
                                        page=str(page_key))
            if result and isinstance(result, dict):
                return task_key, rel, tidx, ridx, result, None
            # Parse failed — return raw text for debugging
            return task_key, rel, tidx, ridx, None, f"parse_failed:\n{raw_text[:500]}"
        except Exception as e:
            return task_key, rel, tidx, ridx, None, str(e)[:200]

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process_one, t): t for t in row_tasks}
        done_count = 0

        for fut in as_completed(futures):
            task_key, rel, tidx, ridx, result, error = fut.result()
            done_count += 1

            with _results_lock:
                if result is not None:
                    _results.setdefault(rel, {}).setdefault(tidx, {})[ridx] = result
                    progress["done"].append(task_key)
                    stats["ok"] += 1
                else:
                    progress["failed"].append(task_key)
                    stats["failed"] += 1
                    if stats["failed"] <= 5:
                        print(f"  FAIL [{task_key}]: {error}", flush=True)

                # Track per-file completion and flush when ready
                done_per_file[rel] = done_per_file.get(rel, 0) + 1
                if done_per_file[rel] >= tasks_per_file.get(rel, float('inf')):
                    flush_file(rel)

            elapsed = (time.time() - start_time) / 60
            speed = done_count / max(elapsed, 0.01)
            remaining = (total - done_count) / max(speed, 0.01)
            status = "OK" if result is not None else "FAIL"
            print(f"  [{done_count}/{total}] {status} {task_key} "
                  f"({speed:.1f}/min, ~{remaining:.0f}min left)", flush=True)

            if done_count % 50 == 0:
                save_progress(progress, progress_file)

    save_progress(progress, progress_file)

    # Flush all files that have any new results (including re-runs with new data)
    for rel in _results:
        flush_file(rel)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done! {stats['ok']} rows expanded, {stats['failed']} failed")
    print(f"Output files: {stats['files_written']}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
