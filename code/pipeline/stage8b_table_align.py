#!/usr/bin/env python3
"""
Per-table VLM alignment of reaction role slots -> annotated-image mol_ids.

Unlike stage8 (which aligns labels per PAGE), this stage operates per TABLE.
For each table produced by stage4/stage4b, the VLM is shown:
  - the annotated page image(s) with red-numbered molecule boxes
  - the table's reaction slots (substrate/product/catalyst/ligand) with their
    author labels and short descriptions

It returns, for each slot, the `box_id` of the best-matching red-numbered box
in the provided image(s). A `box_id` has the form "p{N}:{mol_id}" so that
cross-page tables can reference the correct page.

SMILES are then resolved from the stage8 per-page mol list
(`mol_labels_screening_si/{Journal}/{DOI}/p{N}.json`) — the VLM NEVER
generates SMILES. If a slot cannot be matched, its smiles is left empty.

Output:
  mol_table_alignments_si/{Journal}/{DOI}/p{N}.json
  -> list of alignment records (one per table anchored on that page)

Usage:
  python pipeline/stage8b_table_align.py --api-url http://host:port/v1/chat/completions \
      --doi-filter 10.1038__s41467-026-69464-3_si
  python pipeline/stage8b_table_align.py --api-url ... --journal NatCommun
"""

import argparse
import base64
import functools
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.utils.api_usage import (  # noqa: E402
    resolve_api_key, build_headers, log_usage, DEFAULT_USAGE_LOG,
)

_API_HEADERS: dict | None = None
_USAGE_LOG: str | None = None

# Flush prints line-by-line so `| tee` / log files see progress in real time
# even without python -u on the command line.
print = functools.partial(print, flush=True)  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TABLES_DIR = BASE_DIR / "screening_tables_processed"
DEFAULT_TABLES_FALLBACK = BASE_DIR / "screening_tables_recognized"
DEFAULT_LABELS_DIRS = [
    BASE_DIR / "mol_labels_screening_si",
    BASE_DIR / "mol_labels_screening",
]
DEFAULT_BOXES_DIRS = [
    BASE_DIR / "mol_boxes_decimer_si",
    BASE_DIR / "mol_boxes_decimer",
]
DEFAULT_OUTPUT_DIR = BASE_DIR / "mol_table_alignments_si"

ROLE_KEYS = ["substrates", "products", "catalysts", "ligands"]


# ══════════════════════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════════════════════

VLM_TABLE_ALIGN_PROMPT = """\
You are aligning the reaction slots of ONE chemistry screening table with
molecule structures detected in the paper's images.

INPUT IMAGES
------------
You are given one or more annotated page images. Each image has:
  - a page tag (e.g. "PAGE p12")
  - red numbered boxes drawn around every detected molecule structure
  - the red number inside each box is that molecule's `mol_id` on that page

Image page tags for this task (in order): {page_list}

INPUT TABLE
-----------
The single table you need to align (JSON):

{table_json}

KNOWN LABEL ASSIGNMENTS (from stage 8 page-level reading)
---------------------------------------------------------
A prior VLM pass already identified the author label printed under each
numbered box on each page. These assignments are STRONG priors:

{known_labels_block}

When a slot's `label` is non-null and matches a KNOWN label above (case-
insensitive, exact match after trimming), you MUST output that label's
corresponding `box_id` (p{{N}}:{{mol_id}}). Do NOT pick a different box based
on visual matching alone. Only override a KNOWN assignment when:
  (a) the KNOWN list has multiple boxes sharing the same label, in which
      case use the slot's description + visual match to disambiguate; or
  (b) the slot's label is null or does not appear in the KNOWN list.

TASK
----
For EACH slot in `reaction.substrates`, `reaction.products`,
`reaction.catalysts`, and `reaction.ligands`, identify which red-numbered
box in the images shows that exact molecule.

- A slot has a `label` (author's compound code, e.g. "1", "L1", "28",
  possibly null) and a `description` (short text hint, possibly null).
- Match based on the drawn structure, author labels printed NEAR each box,
  and the slot's label/description.
- A slot whose label also appears as a compound code somewhere else in the
  paper (e.g. substrate "1" that is used across multiple tables) can still
  be matched — pick the box whose drawn structure matches the slot's
  description.
- Unlabeled slots (label is null) can still be matched when the structure
  or role/position in the scheme makes the mapping clear.
- If no box matches, return `box_id`: null.

OUTPUT FORMAT (JSON only, no prose)
-----------------------------------
{{
  "substrates": [
    {{"slot_idx": 0, "box_id": "p12:3"}},
    {{"slot_idx": 1, "box_id": null}}
  ],
  "products":   [{{"slot_idx": 0, "box_id": "p12:5"}}],
  "catalysts":  [],
  "ligands":    []
}}

Rules:
- `box_id` MUST be either null or the string "p{{N}}:{{mol_id}}" where N is
  a page number from the tag of one of the provided images and mol_id is
  the red number drawn inside the chosen box.
- Include ONE entry per slot, preserving `slot_idx` (0-based position in
  the table's list).
- Do NOT output any SMILES, explanation, or extra keys.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

_PAGE_RX = re.compile(r"^p(\d+)")


def page_num(page_stem: str) -> int | None:
    m = _PAGE_RX.match(page_stem)
    return int(m.group(1)) if m else None


def page_key_from(page_stem: str) -> str:
    """Return 'p12' from 'p12' or 'p12.png'."""
    return page_stem[:-4] if page_stem.endswith(".png") else page_stem


def _table_page_key(table: dict) -> str:
    page = str(table.get("page", "")).replace(".png", "")
    if page:
        return page
    image = str(table.get("image", ""))
    if image:
        return Path(image).stem
    return ""


def _table_dedup_key(table: dict) -> str:
    pk = _table_page_key(table)
    title = re.sub(r"[\s^a-z*]+", "", (table.get("table_title") or "").lower().strip())
    return f"{pk}::{title}"


def load_tables(tables_root: Path, fallback_root: Path, journal: str,
                doi: str) -> list[dict]:
    """Load tables from both processed and recognized dirs, merging per table.

    Per-table dedup: identical (page, title) from processed takes priority
    over recognized; tables only present in recognized are still included.
    This matches stage9's load_table_data_indexed behaviour, so stage8b
    sees every table stage9 will assemble.
    """
    tagged = []  # (table, priority, dedup_key)
    for priority, root in enumerate((tables_root, fallback_root)):
        path = root / journal / f"{doi}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"    cannot parse {path}: {e}")
            continue
        tables = data if isinstance(data, list) else [data]
        for t in tables:
            if not isinstance(t, dict):
                continue
            tagged.append((t, priority, _table_dedup_key(t)))

    if not tagged:
        return []

    best_priority = {}
    for _t, priority, dk in tagged:
        if not dk:
            continue
        if dk not in best_priority or priority < best_priority[dk]:
            best_priority[dk] = priority

    result = []
    for t, priority, dk in tagged:
        if not dk:
            result.append(t)
            continue
        if priority == best_priority.get(dk, priority):
            result.append(t)
    return result


def load_page_mol_labels(labels_roots: list, journal: str, doi: str,
                         page_key: str) -> dict:
    """Load stage8 per-page mol list from the first matching labels root.

    Returns {mol_id(str): mol_entry}.
    """
    if isinstance(labels_roots, (str, os.PathLike)):
        labels_roots = [labels_roots]
    for root in labels_roots:
        path = Path(root) / journal / doi / f"{page_key}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        out = {}
        for m in data.get("molecules", []):
            mid = m.get("mol_id")
            if mid is None:
                continue
            out[str(mid)] = m
        if out:
            return out
    return {}


def find_annotated_image(boxes_roots: list, journal: str, doi: str,
                         page_key: str) -> Path | None:
    if isinstance(boxes_roots, (str, os.PathLike)):
        boxes_roots = [boxes_roots]
    for root in boxes_roots:
        p = Path(root) / journal / doi / f"{page_key}_annotated.png"
        if p.exists():
            return p
    return None


def list_available_pages(boxes_roots: list, journal: str, doi: str) -> set:
    """Page stems present as annotated images across all boxes dirs."""
    if isinstance(boxes_roots, (str, os.PathLike)):
        boxes_roots = [boxes_roots]
    out: set[str] = set()
    for root in boxes_roots:
        d = Path(root) / journal / doi
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.name.endswith("_annotated.png"):
                out.add(p.name[:-len("_annotated.png")])
    return out


def build_known_labels(mol_cache: dict, journal: str, doi: str,
                       pages: list) -> tuple:
    """Collect stage8 per-page (mol_id -> compound_label) assignments.

    Returns (display_list, lookup):
      - display_list: list of (page_key, mol_id, label, role) sorted for prompt
      - lookup: {normalized_label: [(page_key, mol_id), ...]} for post-hoc
        override (case-insensitive, stripped)
    """
    display = []
    lookup = defaultdict(list)
    for p in pages:
        page_mols = mol_cache.get((journal, doi, p), {})
        for mid, mol in page_mols.items():
            lbl = (mol.get("compound_label") or "").strip()
            if not lbl:
                continue
            role = mol.get("role") or ""
            display.append((p, str(mid), lbl, role))
            lookup[lbl.lower()].append((p, str(mid)))

    def _sort_key(it):
        p, mid, _, _ = it
        try:
            m = int(mid)
        except (TypeError, ValueError):
            m = 10 ** 9
        return (p, m)

    display.sort(key=_sort_key)
    return display, dict(lookup)


def format_known_labels_block(display: list) -> str:
    if not display:
        return "(no page-level label assignments available for these pages)"
    lines = []
    last_page = None
    for p, mid, lbl, role in display:
        if p != last_page:
            lines.append(f"PAGE {p}:")
            last_page = p
        role_str = f"  [{role}]" if role else ""
        lines.append(f"  mol_id={mid}  label={lbl}{role_str}")
    return "\n".join(lines)


def build_compact_table(table: dict) -> dict:
    """Strip smiles out and keep only fields useful for alignment."""
    reaction = table.get("reaction", {}) or {}
    out = {
        "table_title": table.get("table_title", ""),
        "caption": table.get("caption", ""),
        "columns": table.get("columns", []),
        "reaction": {},
        "conditions_on_arrow": reaction.get("conditions_on_arrow", []),
    }
    for role in ROLE_KEYS:
        slots = []
        for idx, s in enumerate(reaction.get(role, []) or []):
            slots.append({
                "slot_idx": idx,
                "label": s.get("label"),
                "description": s.get("description"),
            })
        out["reaction"][role] = slots
    return out


def related_pages_for(table: dict, available_pages: set[str]) -> list[str]:
    """Compute [page] or [prev, page, next] based on completed_from_adjacent."""
    page_field = table.get("page") or ""
    base = page_key_from(page_field)
    n = page_num(base)
    if n is None:
        return [base] if base else []

    if not table.get("completed_from_adjacent"):
        return [base]

    out = []
    for nn in (n - 1, n, n + 1):
        key = f"p{nn}"
        if key in available_pages:
            out.append(key)
    if not out:
        out = [base]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# VLM API
# ══════════════════════════════════════════════════════════════════════════════

_cached_api_model_name = None


def resolve_api_model_name(api_url: str, model_name: str = None) -> str:
    global _cached_api_model_name
    if _cached_api_model_name:
        return _cached_api_model_name
    if model_name:
        _cached_api_model_name = model_name
        return model_name
    import requests as _req
    base = api_url.rsplit("/v1/", 1)[0] + "/v1"
    try:
        resp = _req.get(f"{base}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            _cached_api_model_name = models[0]["id"]
            print(f"  Auto-detected API model: {_cached_api_model_name}")
            return _cached_api_model_name
    except Exception as e:
        print(f"  model-detect error: {e}")
    _cached_api_model_name = "kimi-k2.6"
    return _cached_api_model_name


def image_to_b64(img_bgr: np.ndarray, max_w: int = 2000) -> str:
    h, w = img_bgr.shape[:2]
    if w > max_w:
        scale = max_w / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def call_vlm(api_url: str, model_name: str, images: list[tuple[str, np.ndarray]],
             table_compact: dict, known_labels_block: str = "",
             timeout: int = 600,
             doi: str | None = None, page: str | None = None) -> str:
    """images: list of (page_tag, bgr_image)."""
    import requests

    resolved = resolve_api_model_name(api_url, model_name)
    page_tags = [tag for tag, _ in images]
    prompt = VLM_TABLE_ALIGN_PROMPT.format(
        page_list=", ".join(page_tags),
        table_json=json.dumps(table_compact, ensure_ascii=False, indent=2),
        known_labels_block=(known_labels_block
                            or "(no page-level label assignments available)"),
    )

    content = []
    for tag, img in images:
        content.append({"type": "text", "text": f"Image for {tag}:"})
        b64 = image_to_b64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": resolved,
        "messages": [
            {"role": "system",
             "content": "You are a chemistry image analysis assistant. "
                        "Respond with ONLY valid JSON — no reasoning, no markdown."},
            {"role": "user", "content": content},
        ],
        "max_tokens": 45500,
        "temperature": 1,
        "response_format": {"type": "json_object"},
    }
    t0 = time.time()
    resp = requests.post(api_url, json=payload, headers=_API_HEADERS, timeout=timeout)
    resp.raise_for_status()
    _resp_json = resp.json()
    _msg = _resp_json["choices"][0]["message"]
    if _USAGE_LOG:
        log_usage(_USAGE_LOG, stage="table_align", doi=doi, page=page,
                  model=resolved, usage=_resp_json.get("usage"),
                  message=_msg, elapsed=time.time() - t0)
    return _msg.get("content") or ""


def parse_vlm_response(response: str, table_compact: dict) -> dict:
    """Return {role: [{slot_idx, box_id}, ...]}."""
    out = {role: [] for role in ROLE_KEYS}
    try:
        text = _strip_thinking(response)
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return out
        data = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        return out

    for role in ROLE_KEYS:
        items = data.get(role, []) or []
        if not isinstance(items, list):
            continue
        n_expected = len(table_compact.get("reaction", {}).get(role, []))
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                si = int(item.get("slot_idx"))
            except (TypeError, ValueError):
                continue
            if si < 0 or si >= n_expected or si in seen:
                continue
            seen.add(si)
            box_id = item.get("box_id")
            if box_id is not None and not isinstance(box_id, str):
                box_id = None
            if isinstance(box_id, str) and not re.match(r"^p\d+:\S+$", box_id):
                box_id = None
            out[role].append({"slot_idx": si, "box_id": box_id})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Per-table processing
# ══════════════════════════════════════════════════════════════════════════════

def process_table(table_idx: int, table: dict, *,
                  journal: str, doi: str, args,
                  mol_cache: dict) -> dict | None:
    """Run VLM alignment for one table and resolve SMILES from stage8.

    Returns an alignment record, or None if we cannot proceed (no image).
    """
    labels_roots = [Path(d) for d in args.labels_dirs]
    boxes_roots = [Path(d) for d in args.boxes_dirs]

    # Which pages do we care about?
    available_pages = list_available_pages(boxes_roots, journal, doi)

    pages = related_pages_for(table, available_pages)
    pages = [p for p in pages if p in available_pages]
    if not pages:
        return None

    # Load page mol lookups
    for p in pages:
        if (journal, doi, p) not in mol_cache:
            mol_cache[(journal, doi, p)] = load_page_mol_labels(
                labels_roots, journal, doi, p)

    # Assemble VLM inputs
    images = []
    for p in pages:
        img_path = find_annotated_image(boxes_roots, journal, doi, p)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        images.append((f"PAGE {p}", img))
    if not images:
        return None

    table_compact = build_compact_table(table)

    # Build stage8 priors for the pages in scope
    known_display, known_lookup = build_known_labels(
        mol_cache, journal, doi, pages)
    known_block = format_known_labels_block(known_display)

    # Call VLM
    try:
        raw = call_vlm(args.api_url, args.api_model, images, table_compact,
                       known_labels_block=known_block,
                       doi=f"{journal}/{doi}",
                       page=page_key_from(table.get("page") or ""))
    except Exception as e:
        return {
            "table_idx": table_idx,
            "table_title": table.get("table_title", ""),
            "anchor_page": page_key_from(table.get("page") or ""),
            "related_pages": pages,
            "error": f"vlm_call_failed: {e}",
            "alignments": {role: [] for role in ROLE_KEYS},
        }

    parsed = parse_vlm_response(raw, table_compact)

    # Resolve SMILES from stage8 mol cache
    alignments = {}
    reaction = table.get("reaction", {}) or {}
    anchor_page_key = page_key_from(table.get("page") or "")
    for role in ROLE_KEYS:
        slots = reaction.get(role, []) or []
        role_items = {r["slot_idx"]: r for r in parsed.get(role, [])}
        aligned = []
        for idx, slot in enumerate(slots):
            box_id = role_items.get(idx, {}).get("box_id")
            forced_by_stage8 = False

            # Post-hoc override: if stage8 uniquely resolves the slot's label
            # to a (page, mol_id), force that box even if the VLM said
            # otherwise. If stage8 has multiple boxes with the same label,
            # keep the VLM's choice (descriptions disambiguate).
            slot_lbl = (slot.get("label") or "").strip().lower()
            if slot_lbl:
                candidates = known_lookup.get(slot_lbl, [])
                if len(candidates) == 1:
                    forced = f"{candidates[0][0]}:{candidates[0][1]}"
                    if box_id != forced:
                        box_id = forced
                        forced_by_stage8 = True
                elif len(candidates) > 1 and anchor_page_key:
                    # Prefer the anchor-page candidate when the VLM picked
                    # nothing or picked something not in the candidate set.
                    on_anchor = [c for c in candidates
                                 if c[0] == anchor_page_key]
                    vlm_pick_ok = False
                    if isinstance(box_id, str):
                        m = re.match(r"^p(\d+):(\S+)$", box_id)
                        if m:
                            vlm_pick = (f"p{m.group(1)}", m.group(2))
                            vlm_pick_ok = vlm_pick in candidates
                    if len(on_anchor) == 1 and not vlm_pick_ok:
                        forced = f"{on_anchor[0][0]}:{on_anchor[0][1]}"
                        if box_id != forced:
                            box_id = forced
                            forced_by_stage8 = True

            entry = {
                "slot_idx": idx,
                "label": slot.get("label"),
                "description": slot.get("description"),
                "box_id": box_id,
                "smiles": "",
                "rdkit_valid": False,
                "smiles_source": "none",
            }
            if forced_by_stage8:
                entry["forced_by_stage8"] = True
            if isinstance(box_id, str):
                m = re.match(r"^p(\d+):(\S+)$", box_id)
                if m:
                    bx_page = f"p{m.group(1)}"
                    bx_mid = m.group(2)
                    mol = mol_cache.get((journal, doi, bx_page), {}).get(bx_mid)
                    if mol and mol.get("smiles"):
                        entry["smiles"] = mol["smiles"]
                        entry["rdkit_valid"] = bool(mol.get("rdkit_valid"))
                        entry["smiles_source"] = "ocsr_table_aligned"
                        entry["bbox"] = mol.get("bbox")
            aligned.append(entry)
        alignments[role] = aligned

    return {
        "table_idx": table_idx,
        "table_title": table.get("table_title", ""),
        "anchor_page": page_key_from(table.get("page") or ""),
        "related_pages": pages,
        "alignments": alignments,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Task collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_dois(tables_root: Path, fallback_root: Path,
                 journal_filter: str | None,
                 doi_filter: str | None) -> list[tuple[str, str]]:
    """Return list of (journal, doi) pairs that have a tables JSON."""
    seen = set()
    out = []
    for root in (tables_root, fallback_root):
        if not root.is_dir():
            continue
        for jdir in sorted(root.iterdir()):
            if not jdir.is_dir():
                continue
            if journal_filter and jdir.name != journal_filter:
                continue
            for f in sorted(jdir.iterdir()):
                if not f.name.endswith(".json"):
                    continue
                doi = f.name[:-len(".json")]
                if doi_filter and doi != doi_filter:
                    continue
                key = (jdir.name, doi)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def process_doi(journal: str, doi: str, args) -> dict:
    """Process all tables for one DOI and write per-page output files."""
    tables = load_tables(Path(args.tables_dir), Path(args.tables_fallback),
                         journal, doi)
    if not tables:
        return {"doi": f"{journal}/{doi}", "n_tables": 0, "written": 0}

    output_dir = Path(args.output_dir) / journal / doi
    output_dir.mkdir(parents=True, exist_ok=True)

    mol_cache: dict = {}

    # Process tables and group by anchor page
    by_page: dict[str, list[dict]] = {}

    def _run(idx_and_table):
        idx, table = idx_and_table
        rec = process_table(idx, table, journal=journal, doi=doi,
                            args=args, mol_cache=mol_cache)
        return idx, table, rec

    n_workers = max(1, args.concurrency)
    results = []
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_run, (i, t)) for i, t in enumerate(tables)]
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"    table worker error: {e}")
    else:
        for i, t in enumerate(tables):
            results.append(_run((i, t)))

    for idx, table, rec in results:
        if rec is None:
            continue
        anchor = rec.get("anchor_page") or "p0"
        by_page.setdefault(anchor, []).append(rec)

    # Sort each page's records by table_idx for stable order
    written = 0
    for page, recs in by_page.items():
        recs.sort(key=lambda r: r.get("table_idx", 0))
        out_file = output_dir / f"{page}.json"
        out_file.write_text(json.dumps(recs, ensure_ascii=False, indent=2))
        written += 1

    return {"doi": f"{journal}/{doi}", "n_tables": len(tables), "written": written}


def main():
    ap = argparse.ArgumentParser(
        description="Per-table VLM alignment of reaction slots to mol_ids")
    ap.add_argument("--api-url", required=True,
                    help="VLM API endpoint, e.g. http://host:port/v1/chat/completions")
    ap.add_argument("--api-key", default=None,
                    help="API key (或 PROXY_API_KEY / OPENAI_API_KEY 环境变量)")
    ap.add_argument("--usage-log", default=None,
                    help=f"Token 用量 JSONL 日志 (默认 {DEFAULT_USAGE_LOG})")
    ap.add_argument("--api-model", default=None)
    ap.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR),
                    help="stage4b processed tables dir (preferred)")
    ap.add_argument("--tables-fallback", default=str(DEFAULT_TABLES_FALLBACK),
                    help="stage4 recognized tables dir (fallback)")
    ap.add_argument("--labels-dir", dest="labels_dirs", nargs="*",
                    default=[str(d) for d in DEFAULT_LABELS_DIRS],
                    help="mol_labels directories (searched in order; "
                         "default: mol_labels_screening_si + mol_labels_screening)")
    ap.add_argument("--boxes-dir", dest="boxes_dirs", nargs="*",
                    default=[str(d) for d in DEFAULT_BOXES_DIRS],
                    help="mol_boxes directories with *_annotated.png "
                         "(searched in order; default: mol_boxes_decimer_si + mol_boxes_decimer)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--journal", default=None, help="filter by journal name")
    ap.add_argument("--doi-filter", default=None,
                    help="only process this single DOI (e.g. 10.1038__s41467-026-69464-3_si)")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="concurrent tables per DOI")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="skip DOIs whose output dir already has files")
    ap.add_argument("--test", type=int, default=0,
                    help="process only the first N DOIs (0 = no limit)")
    args = ap.parse_args()

    global _API_HEADERS, _USAGE_LOG
    _api_key = resolve_api_key(args.api_key)
    _API_HEADERS = build_headers(_api_key)
    _USAGE_LOG = args.usage_log or DEFAULT_USAGE_LOG
    print(f"  auth:     {'Bearer ***' if _api_key else '(none)'}")
    print(f"  usage log:{_USAGE_LOG}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dois = collect_dois(Path(args.tables_dir), Path(args.tables_fallback),
                        args.journal, args.doi_filter)
    if args.test and args.test > 0:
        dois = dois[:args.test]
    if args.resume:
        kept = []
        skipped = 0
        for journal, doi in dois:
            out_sub = output_dir / journal / doi
            if out_sub.is_dir() and any(out_sub.glob("*.json")):
                skipped += 1
                continue
            kept.append((journal, doi))
        if skipped:
            print(f"Resume: skipping {skipped} already-processed DOIs")
        dois = kept

    print(f"stage8b per-table alignment")
    print(f"  tables:   {args.tables_dir} (fallback {args.tables_fallback})")
    print(f"  labels:   {args.labels_dirs}")
    print(f"  boxes:    {args.boxes_dirs}")
    print(f"  output:   {args.output_dir}")
    print(f"  api:      {args.api_url}")
    print(f"  DOIs:     {len(dois)}")
    if args.doi_filter:
        print(f"  filter:   {args.doi_filter}")
    print("=" * 60)

    if not dois:
        print("Nothing to process.")
        return

    t0 = time.time()
    for i, (journal, doi) in enumerate(dois, 1):
        t1 = time.time()
        try:
            summary = process_doi(journal, doi, args)
        except Exception as e:
            print(f"  [{i}/{len(dois)}] {journal}/{doi}: ERROR {e}")
            continue
        dt = time.time() - t1
        elapsed = time.time() - t0
        eta = (elapsed / i) * (len(dois) - i)
        print(f"  [{i}/{len(dois)}] {summary['doi']}: "
              f"{summary['n_tables']} tables, {summary['written']} pages "
              f"({dt:.1f}s, ETA {eta/60:.1f}min)")

    print("=" * 60)
    print(f"Done in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
