#!/usr/bin/env python3
"""
VLM label-to-structure alignment for screening-table molecules.

This stage aligns detected molecule bounding boxes (``mol_id``) with
author-assigned compound labels and roles. Structure-recognition results are
external inputs: ``--smiles-dir`` must contain precomputed per-page JSON files.
The fields ``smiles``, ``smiles_expanded``, and ``rdkit_valid`` are copied from
those files without expansion, repair, validation, or other transformation.

Usage:
    python pipeline/stage8_align_labels.py --api-url http://... --test 5
    python pipeline/stage8_align_labels.py --api-url http://... --journal JACS --resume
    python pipeline/stage8_align_labels.py --api-url http://... \
        --smiles-dir /path/to/external/results
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.utils.api_usage import (  # noqa: E402
    resolve_api_key, build_headers, log_usage, DEFAULT_USAGE_LOG,
)

# Module-level context populated in main() and consumed by the run_vlm_* helpers
# (keeps call sites compact while still recording token usage per request).
_API_HEADERS: dict | None = None
_USAGE_LOG: str | None = None
_USAGE_DOI: str | None = None
_USAGE_PAGE: str | None = None


# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_IMAGES_DIRS = [
    BASE_DIR / "screening_tables_kimi",
    BASE_DIR / "screening_tables_si",
]
DEFAULT_BOXES_INDEX = BASE_DIR / "mol_boxes_decimer_si" / "boxes_index.json"
DEFAULT_SMILES_DIR = BASE_DIR / "mol_smiles_decimer_si"
DEFAULT_OUTPUT_DIR = BASE_DIR / "mol_labels_screening_si"

# ── compound label regex ─────────────────────────────────────────────────────
LABEL_PATTERN = re.compile(
    r"^(?:"
    r"[1-9]\d{0,2}[a-z]{0,3}'?"        # numeric: 1a, 2a', 14b, 3aa
    r"|L\d{1,2}"                         # ligand: L1, L12
    r"|[Cc]at\.?\s*\d*"                  # catalyst: cat1, Cat.2
    r"|[A-Z][a-z]?\d*"                   # element-like: Pd1, Ru2
    r"|R\d*"                             # R-group: R, R1
    r"|S\d{1,2}"                         # substrate: S1, S12
    r"|P\d{1,2}"                         # product: P1
    r")$"
)

# ── VLM prompt for screening table label extraction ──────────────────────────
VLM_LABEL_PROMPT = """\
This is a chemistry paper page with a reaction condition screening table.
Red boxes (numbered 0, 1, 2, ...) mark detected molecule structures.

For each box, identify:
1. compound_label — short author-assigned code (e.g. "1a", "L1", "Pd-1")
2. role — "substrate"/"product"/"catalyst"/"ligand"/"additive"/"reagent"/"solvent"/"other"

RULES:
- Red numbers are MY annotations, NOT compound labels.
- Look BELOW and to the RIGHT of each box for labels.
- If no label visible, set compound_label to null — but STILL assign the correct role.
- Molecules to the LEFT of the reaction arrow are typically substrates.
- Molecules to the RIGHT of the reaction arrow are typically products.
- Molecules ABOVE/BELOW the arrow (or in conditions text) are catalysts, ligands, additives, or reagents.
- Labels like "1a-e" (a RANGE of variants) → use base "1a".
- But labels like "Med-1", "DF-2", "Cat-3" are NUMBERED compounds — keep the FULL label including the number suffix.

Output ONLY a JSON array:
[{"box": 0, "compound_label": "1a", "role": "substrate"}, ...]
"""

# ── VLM prompt for crop-based single-molecule label extraction ────────────────
VLM_CROP_LABEL_PROMPT = """\
This is a cropped region from a chemistry paper showing a single molecule structure.
What is the compound label (author-assigned code like "1a", "L1", "Pd-1", "S1") for this molecule?
Also identify the role: substrate/product/catalyst/ligand/additive/reagent/solvent/other.

RULES:
- The compound label is usually printed BELOW the molecule structure, sometimes to the side.
- Do NOT confuse atom labels (C, N, O, Ph, Me, OTf ...) with compound labels.
- Compound labels are short codes like "1a", "2b", "S1", "L1", "Cat-1", etc.
- If no compound label is visible, set compound_label to null.
- Output ONLY JSON: {"compound_label": "1a", "role": "substrate"}
"""

# ── VLM prompt for full-page annotated image + table context ────────────────
VLM_PAGE_TABLE_PROMPT = """\
You are analyzing a chemistry paper page with a reaction condition screening table.
The image shows molecule structures with red numbered boxes (0, 1, 2, ...).

TASK:
For each red-numbered box, identify the compound label and role by reading the image.

There are TWO types of molecules:

TYPE 1: Reaction scheme molecules (usually above the table)
- Molecules LEFT of the reaction arrow → substrates
- Molecules RIGHT of the reaction arrow → products
- Molecules ABOVE/BELOW the arrow → catalyst / reagent

TYPE 2: Reference molecules (drawn elsewhere on the page, e.g. a panel of structures)
These are molecules with SHORT CODES printed near them. Common patterns:
- Ligands: L1, L2, L3, ... or CL1, SL1, etc.
- Catalysts: Cat-1, Cat-2, A1, A5, A7, ...
- Photocatalysts: PC1, PC2, ...
- Chiral ligands: CL1, CL3, ...
- Bases: B1, B2, ...
- Additives: Add-1, ...

HOW TO MATCH:
1. Read the label printed NEAR each red-numbered box (usually below or beside the molecule).
2. For reaction scheme molecules, use position (left/right of arrow) to determine the role.
3. For reference molecules, read the label text directly from the image.

For each box, output:
- compound_label: the label/code (e.g. "1a", "3a", "L1", "Cat-1", "A5")
- role: substrate / product / catalyst / ligand / additive / reagent / solvent / other

RULES:
- Do NOT confuse atom labels or functional group abbreviations \
(C, N, O, Ph, Me, OTf, Tf, Ds, Boc, TMS, TIPS, Bpin, ...) with compound labels.
- Compound labels are author-assigned codes: "1a", "2", "3a", "L1", "L2", "Cat-1", "A5", "CL3", etc.
- If a box clearly shows a substrate/product/ligand but you cannot determine its exact label, \
still assign the correct role and set compound_label to null.

Output ONLY JSON:
{{"molecules": [{{"box_id": 0, "compound_label": "1a", "role": "substrate"}}, ...]}}
"""

# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1: VLM visual label reading
# ══════════════════════════════════════════════════════════════════════════════

def annotate_page(source_img: np.ndarray, bboxes: list,
                  scale: float = 0.4) -> np.ndarray:
    """Draw numbered markers at bbox centers.

    White circle with bold red number at the center of each bbox.
    """
    img = source_img.copy()
    h, w = img.shape[:2]
    outline_thickness = max(1, int(h / 3000))
    font_scale = max(1.0, h / 3000)
    font_thickness = max(2, int(h / 1200))
    radius = max(18, int(h / 150))

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox

        # Thin red outline
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), outline_thickness)

        # Circled number at bbox center
        label = str(i)
        (tw, th_txt), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # White filled circle with red border
        cv2.circle(img, (cx, cy), radius, (255, 255, 255), -1)
        cv2.circle(img, (cx, cy), radius, (0, 0, 255), max(2, outline_thickness))
        # Bold red number centered in circle
        tx = cx - tw // 2
        ty = cy + th_txt // 2
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 255), font_thickness)

    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_mol_region(source_img: np.ndarray, bbox: list,
                    pad_lr: float = 0.3, pad_top: float = 0.3,
                    pad_bottom: float = 1.0) -> np.ndarray:
    """Crop a molecule region with extra padding (especially below for labels).

    Args:
        source_img: full page image (BGR)
        bbox: [x1, y1, x2, y2] of the molecule
        pad_lr: left/right padding as fraction of bbox width
        pad_top: top padding as fraction of bbox height
        pad_bottom: bottom padding as fraction of bbox height (larger to capture label)
    Returns:
        cropped image (BGR)
    """
    h, w = source_img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    cx1 = max(0, int(x1 - bw * pad_lr))
    cy1 = max(0, int(y1 - bh * pad_top))
    cx2 = min(w, int(x2 + bw * pad_lr))
    cy2 = min(h, int(y2 + bh * pad_bottom))

    return source_img[cy1:cy2, cx1:cx2].copy()


def _strip_thinking(text: str) -> str:
    """Remove thinking blocks from thinking-model responses.

    Handles both complete ``<think>...</think>`` blocks and responses that
    contain only a closing tag.
    """
    text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    return text.strip()


def parse_vlm_crop_response(response: str) -> dict | None:
    """Parse VLM response for a single-molecule crop label query.

    Expected format: {"compound_label": "1a", "role": "substrate"}
    """
    response = _strip_thinking(response)
    try:
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            label = clean_compound_label(data.get("compound_label"))
            role = data.get("role", "other")
            if role not in {"substrate", "product", "catalyst", "ligand",
                            "additive", "reagent", "solvent", "other"}:
                role = "other"
            return {"compound_label": label, "role": role}
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def clean_compound_label(raw: str | None) -> str | None:
    """Normalize compound labels."""
    if not raw:
        return None
    raw = raw.strip()
    # Unicode subscripts
    for old, new in [("₀", "0"), ("₁", "1"), ("₂", "2"), ("₃", "3"),
                     ("₄", "4"), ("₅", "5"), ("₆", "6"), ("₇", "7"),
                     ("₈", "8"), ("₉", "9")]:
        raw = raw.replace(old, new)
    # Range like "2b-2d", "1a-e" → first ID.
    # But NOT numbered compounds like "DF-1", "Med-2", "Cat-3".
    # A range has a letter suffix on both sides (e.g. "2b-2d", "1a-e").
    m = re.match(r'^(\d+[a-z])\s*[-–]\s*(?:\d+)?[a-z]$', raw)
    if m:
        raw = m.group(1)
    if len(raw) > 8:
        return None
    return raw


def parse_vlm_label_response(response: str, n_boxes: int) -> list[dict]:
    """Parse VLM JSON array response for label extraction."""
    response = _strip_thinking(response)
    results = []
    # Try JSON array
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if isinstance(data, list):
                results = data
    except json.JSONDecodeError:
        pass

    # Fallback regex
    if not results:
        pattern = re.compile(
            r'\{[^}]*"box"\s*:\s*(\d+)[^}]*"compound_label"\s*:\s*'
            r'(?:"([^"]*?)"|null)[^}]*"role"\s*:\s*(?:"([^"]*?)"|null)[^}]*\}'
        )
        for m in pattern.finditer(response):
            results.append({
                "box": int(m.group(1)),
                "compound_label": m.group(2) if m.group(2) else None,
                "role": m.group(3) if m.group(3) else None,
            })

    # Post-process
    for r in results:
        r["compound_label"] = clean_compound_label(r.get("compound_label"))
        if r.get("role") and r["role"] not in {
            "substrate", "product", "catalyst", "ligand",
            "additive", "reagent", "solvent", "other"
        }:
            r["role"] = "other"

    return results



def run_vlm_label_api(api_url: str, image: np.ndarray,
                      model_name: str = None) -> str:
    """Send annotated image to vLLM OpenAI-compatible API."""
    import base64
    import requests

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model_name or "default",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": VLM_LABEL_PROMPT},
            ],
        }],
        "max_tokens": 45500,
        "temperature": 1,
    }
    t0 = time.time()
    resp = requests.post(api_url, json=payload, headers=_API_HEADERS, timeout=600)
    resp.raise_for_status()
    _resp_json = resp.json()
    _msg = _resp_json["choices"][0]["message"]
    if _USAGE_LOG:
        log_usage(_USAGE_LOG, stage="align_labels:full_page",
                  doi=_USAGE_DOI, page=_USAGE_PAGE,
                  model=payload.get("model"),
                  usage=_resp_json.get("usage"),
                  message=_msg,
                  elapsed=time.time() - t0)
    return _msg.get("content") or ""


_cached_api_model_name = None

def _resolve_api_model_name(api_url: str, model_name: str = None) -> str:
    """Auto-detect model name from /v1/models if not provided or wrong."""
    global _cached_api_model_name
    if _cached_api_model_name:
        return _cached_api_model_name
    if model_name:
        _cached_api_model_name = model_name
        return model_name
    import requests as _req
    base = api_url.rsplit("/v1/", 1)[0] + "/v1"
    resp = _req.get(f"{base}/models", timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    if models:
        _cached_api_model_name = models[0]["id"]
        print(f"  Auto-detected API model: {_cached_api_model_name}")
    else:
        raise RuntimeError(
            "The API returned no models; pass an explicit --api-model."
        )
    return _cached_api_model_name


def run_vlm_crop_label_api(api_url: str, crop: np.ndarray,
                           model_name: str = None) -> str:
    """Send a single molecule crop to vLLM API for label identification."""
    import base64
    import requests

    resolved_model = _resolve_api_model_name(api_url, model_name)

    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": resolved_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": VLM_CROP_LABEL_PROMPT},
            ],
        }],
        "max_tokens": 45500,
        "temperature": 1,
    }
    t0 = time.time()
    resp = requests.post(api_url, json=payload, headers=_API_HEADERS, timeout=600)
    resp.raise_for_status()
    _resp_json = resp.json()
    _msg = _resp_json["choices"][0]["message"]
    if _USAGE_LOG:
        log_usage(_USAGE_LOG, stage="align_labels:crop",
                  doi=_USAGE_DOI, page=_USAGE_PAGE,
                  model=resolved_model,
                  usage=_resp_json.get("usage"),
                  message=_msg,
                  elapsed=time.time() - t0)
    return _msg.get("content") or ""


def sort_bboxes_row_col(bboxes: list) -> tuple[list, list]:
    """Sort bboxes by row (y-band) then column (x). Returns (sorted_indices, sorted_bboxes)."""
    if not bboxes:
        return [], []
    y_centers = [(b[1] + b[3]) / 2 for b in bboxes]
    heights = [b[3] - b[1] for b in bboxes]
    median_h = float(np.median(heights)) if heights else 50.0
    row_tol = median_h * 0.5

    y_order = sorted(range(len(bboxes)), key=lambda i: y_centers[i])
    rows = []
    current_row = [y_order[0]]
    current_y = y_centers[y_order[0]]
    for idx in y_order[1:]:
        if abs(y_centers[idx] - current_y) <= row_tol:
            current_row.append(idx)
        else:
            rows.append(current_row)
            current_row = [idx]
            current_y = y_centers[idx]
    rows.append(current_row)

    sorted_indices = []
    for row in rows:
        sorted_indices.extend(sorted(row, key=lambda i: bboxes[i][0]))
    sorted_bboxes = [bboxes[i] for i in sorted_indices]
    return sorted_indices, sorted_bboxes


def run_vlm_page_match_api(api_url: str, annotated_img: np.ndarray,
                           model_name: str = None,
                           num_boxes: int = 0) -> str:
    """Send annotated page image to VLM API for label reading.

    Returns raw response text.
    """
    import base64
    import requests

    resolved_model = _resolve_api_model_name(api_url, model_name)

    # Encode annotated image as PNG
    _, buf = cv2.imencode(".png", annotated_img)
    b64 = base64.b64encode(buf.tobytes()).decode()

    prompt_text = VLM_PAGE_TABLE_PROMPT

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system",
             "content": "You are a chemistry image analysis assistant. "
                        "You MUST respond with valid JSON only — no reasoning, "
                        "no explanation, no markdown fences."},
            {"role": "user",
             "content": [
                 {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{b64}"}},
                 {"type": "text", "text": prompt_text},
             ]},
        ],
        "max_tokens": 45500,
        "temperature": 1,
        "response_format": {"type": "json_object"},
    }
    t0 = time.time()
    resp = requests.post(api_url, json=payload, headers=_API_HEADERS, timeout=600)
    resp.raise_for_status()
    _resp_json = resp.json()
    _msg = _resp_json["choices"][0]["message"]
    if _USAGE_LOG:
        log_usage(_USAGE_LOG, stage="align_labels:page_match",
                  doi=_USAGE_DOI, page=_USAGE_PAGE,
                  model=resolved_model,
                  usage=_resp_json.get("usage"),
                  message=_msg,
                  elapsed=time.time() - t0)
    return _msg.get("content") or ""


def parse_vlm_page_match_response(response: str,
                                  sorted_indices: list) -> dict:
    """Parse VLM page-match response, map box_id back to original mol index.

    Returns {original_mol_idx: {"compound_label": ..., "role": ...}}.
    """
    vlm_results = {}
    try:
        response = _strip_thinking(response)
        match = re.search(r'\{[\s\S]*\}', response)
        if not match:
            return vlm_results
        data = json.loads(match.group())
        molecules = data.get("molecules", [])
        for item in molecules:
            box_id = item.get("box_id")
            if box_id is None or not isinstance(box_id, int):
                continue
            if box_id < 0 or box_id >= len(sorted_indices):
                continue
            original_idx = sorted_indices[box_id]
            label = clean_compound_label(item.get("compound_label"))
            role = item.get("role", "other")
            if role not in {"substrate", "product", "catalyst", "ligand",
                            "additive", "reagent", "solvent", "other"}:
                role = "other"
            vlm_results[original_idx] = {
                "compound_label": label,
                "role": role,
            }
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass
    return vlm_results




# ══════════════════════════════════════════════════════════════════════════════
# Task collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_tasks(boxes_index: dict, image_dirs: list, args) -> list[dict]:
    """Build flat list of per-page tasks from boxes_index."""
    tasks = []
    for doi_key, pages in boxes_index.items():
        if args.journal:
            journal = doi_key.split("/")[0]
            if journal not in args.journal.split(","):
                continue
        for page_key, page_data in pages.items():
            mols = page_data.get("molecules", [])
            if not mols:
                continue
            src_rel = page_data["source_image"]
            # Find image in multiple dirs
            img_path = None
            for d in image_dirs:
                p = Path(d) / src_rel
                if p.exists():
                    img_path = str(p)
                    break
            if img_path is None:
                continue
            tasks.append({
                "doi_key": doi_key,
                "page_key": page_key,
                "source_image": img_path,
                "source_rel": src_rel,
                "molecules": mols,
            })

    tasks.sort(key=lambda t: (t["doi_key"], t["page_key"]))
    if args.test > 0:
        tasks = tasks[:args.test]
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# Per-page processing
# ══════════════════════════════════════════════════════════════════════════════

def process_page(task: dict, args, precomputed_vlm: dict = None) -> dict:
    """Align one page and attach externally precomputed structure fields.

    Args:
        precomputed_vlm: if provided, dict of {box_id: {compound_label, role, ...}}
            from batch VLM inference. Skips per-page VLM call.
    """
    doi_key = task["doi_key"]
    page_key = task["page_key"]
    mols = task["molecules"]

    # Update per-task usage context so run_vlm_*_api records the correct DOI/page
    global _USAGE_DOI, _USAGE_PAGE
    _USAGE_DOI = doi_key
    _USAGE_PAGE = str(page_key)

    # Load source image
    src_path = task["source_image"]
    source_img = cv2.imread(src_path)
    if source_img is None:
        return {"error": f"Cannot read {src_path}", "molecules": []}

    h, w = source_img.shape[:2]
    bboxes = [m["bbox"] for m in mols]

    # ── Strategy 1: VLM label reading ───────────────────────────────────────
    vlm_results = {}  # keyed by ORIGINAL mol index
    vlm_failed = False
    if precomputed_vlm is not None:
        vlm_results = precomputed_vlm
    elif args.api_url:
        # Full-page annotated image → single VLM call (with retry).
        # Retry on exception OR on empty parse result (the original silent
        # failure mode that produced all-None compound_label files).
        sorted_indices, sorted_bboxes = sort_bboxes_row_col(bboxes)
        annotated = annotate_page(source_img, sorted_bboxes, scale=args.scale)
        last_err = "initial"
        for attempt in range(3):
            try:
                response = run_vlm_page_match_api(
                    args.api_url, annotated,
                    args.api_model, num_boxes=len(bboxes))
                vlm_results = parse_vlm_page_match_response(response, sorted_indices)
                if vlm_results:
                    last_err = None
                    break
                last_err = "empty_parse_result"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:160]}"
            if attempt < 2:
                time.sleep(2 * (2 ** attempt))
        if last_err is not None:
            print(f"    VLM API failed after 3 retries for "
                  f"{doi_key}/{page_key}: {last_err}")
            vlm_failed = True

    # ── Load externally precomputed structure-recognition output ─────────────
    existing_smiles = {}
    journal = doi_key.split("/")[0]
    doi_dir = doi_key.split("/")[1] if "/" in doi_key else doi_key
    smiles_path = Path(args.smiles_dir) / journal / doi_dir / f"{page_key}.json"
    if not smiles_path.is_file():
        raise FileNotFoundError(
            f"Missing external structure-recognition JSON: {smiles_path}"
        )
    try:
        sdata = json.loads(smiles_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read external structure-recognition JSON: {smiles_path}"
        ) from exc
    molecules = sdata.get("molecules")
    if not isinstance(molecules, list):
        raise ValueError(
            f"Expected a 'molecules' list in external JSON: {smiles_path}"
        )
    for mol_entry in molecules:
        if not isinstance(mol_entry, dict):
            continue
        mid = mol_entry.get("mol_id")
        if mid is not None:
            existing_smiles[str(mid)] = mol_entry

    # ── Fuse & build results ─────────────────────────────────────────────────
    results = []
    warnings = []

    for i, mol in enumerate(mols):
        mol_id = mol.get("mol_id", str(i))
        bbox = mol["bbox"]

        vlm_info = vlm_results.get(i, {})
        vlm_label = vlm_info.get("compound_label")
        vlm_role = vlm_info.get("role", "other")
        label = vlm_label
        confidence = 0.80 if vlm_label else 0.0

        # Copy only the public structure fields from the external JSON. Values
        # are not normalized, expanded, repaired, or revalidated here.
        existing = existing_smiles.get(str(mol_id), {})
        if not existing:
            warnings.append(f"external_structure_missing_for_mol_id:{mol_id}")

        # Keep VLM-assigned role even when no label — unlabeled molecules
        # in reaction schemes still have roles (substrate, product, catalyst, etc.)
        entry = {
            "mol_id": mol_id if isinstance(mol_id, str) else f"mol_{i:03d}",
            "compound_label": label,
            "role": vlm_role,
            "confidence": confidence,
            "bbox": bbox,
            "smiles_source": "external_precomputed" if existing else "none",
        }
        for field in ("smiles", "smiles_expanded", "rdkit_valid"):
            if field in existing:
                entry[field] = existing[field]

        results.append(entry)

    out = {
        "doi_key": doi_key,
        "page_key": page_key,
        "molecules": results,
        "warnings": warnings,
    }
    if vlm_failed:
        out["vlm_failed"] = True
        out["warnings"] = (out.get("warnings") or []) + [
            "vlm_page_match_failed_all_retries"
        ]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "VLM label alignment using externally precomputed structure JSON"
        )
    )

    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec 'RANK/TOTAL' e.g. '0/2' or '1/2' for splitting work across processes")

    # API mode
    parser.add_argument("--api-url", type=str, default=None,
                        help="VLM API endpoint (e.g. http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (或 PROXY_API_KEY / OPENAI_API_KEY 环境变量)")
    parser.add_argument("--usage-log", type=str, default=None,
                        help="Token usage JSONL log path")
    parser.add_argument("--api-model", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of concurrent API requests (default 8)")

    # Data paths
    parser.add_argument("--images-dir", type=str, nargs="*", default=None,
                        help="Image directories to search (default: screening_tables_kimi, screening_tables_si)")
    parser.add_argument("--boxes-index", type=str, default=str(DEFAULT_BOXES_INDEX))
    parser.add_argument(
        "--smiles-dir",
        type=str,
        default=str(DEFAULT_SMILES_DIR),
        help="Directory containing externally precomputed per-page JSON",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    # Processing options
    parser.add_argument("--scale", type=float, default=0.4,
                        help="Image scale for VLM (default: 0.4)")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--journal", type=str, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip already-processed pages (default: True)")

    args = parser.parse_args()

    # Keep --help usable in a minimal environment; image dependencies are only
    # required when the alignment job actually starts.
    global cv2, np, Image
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        parser.error(
            "Image dependencies are required to run label alignment "
            f"({exc.name} is unavailable)."
        )

    # Resolve API key + usage log into module globals consumed by run_vlm_* helpers
    global _API_HEADERS, _USAGE_LOG
    _api_key = resolve_api_key(args.api_key)
    _API_HEADERS = build_headers(_api_key)
    _USAGE_LOG = args.usage_log or DEFAULT_USAGE_LOG
    print(f"Auth: {'Bearer ***' if _api_key else '(none)'}")
    print(f"Usage log: {_USAGE_LOG}")

    image_dirs = [Path(d) for d in args.images_dir] if args.images_dir else DEFAULT_IMAGES_DIRS
    smiles_dir = Path(args.smiles_dir)
    if not smiles_dir.is_dir():
        parser.error(
            "External structure-recognition outputs are required before label "
            f"alignment; --smiles-dir is not a directory: {smiles_dir}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load boxes index
    print(f"Loading boxes index: {args.boxes_index}")
    with open(args.boxes_index) as f:
        boxes_index = json.load(f)

    tasks = collect_tasks(boxes_index, image_dirs, args)

    # Shard: split tasks across parallel processes
    if args.shard:
        rank, total = map(int, args.shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % total == rank]
        print(f"Shard {rank}/{total}: {len(tasks)} tasks")

    # Resume: skip already processed
    if args.resume:
        before = len(tasks)
        new_tasks = []
        for t in tasks:
            journal = t["doi_key"].split("/")[0]
            doi_dir = t["doi_key"].split("/")[1] if "/" in t["doi_key"] else t["doi_key"]
            out_file = output_dir / journal / doi_dir / f"{t['page_key']}.json"
            if out_file.exists():
                continue
            new_tasks.append(t)
        skipped = before - len(new_tasks)
        tasks = new_tasks
        if skipped:
            print(f"Resume: skipping {skipped} already-processed pages")

    total_pages = len(tasks)
    total_mols = sum(len(t["molecules"]) for t in tasks)
    print(f"Label alignment pipeline")
    print(f"  Images:  {[str(d) for d in image_dirs]}")
    print(f"  Output:  {output_dir}")
    print(f"  Pages:   {total_pages}")
    print(f"  Molecules: {total_mols}")
    print(f"  External structures: {smiles_dir}")
    print(f"  VLM API: {args.api_url or 'not set (VLM disabled)'}")
    print("=" * 60)

    if not tasks:
        print("Nothing to process!")
        return

    t0 = time.time()
    total_labeled = 0
    total_processed = 0
    total_valid = 0

    def _process_and_save(task_with_idx):
        idx, task = task_with_idx
        result = process_page(task, args)
        # On VLM failure after retries: do NOT persist. --resume checks file
        # existence, so next run will automatically retry this page.
        if result.get("vlm_failed"):
            fail_log = output_dir / "_stage8_failures.log"
            try:
                with open(fail_log, "a") as f:
                    f.write(f"{task['doi_key']}/{task['page_key']}\n")
            except Exception:
                pass
            return idx, task, result
        # Save result
        journal = task["doi_key"].split("/")[0]
        doi_dir = task["doi_key"].split("/")[1] if "/" in task["doi_key"] else task["doi_key"]
        out_sub = output_dir / journal / doi_dir
        out_sub.mkdir(parents=True, exist_ok=True)
        out_file = out_sub / f"{task['page_key']}.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return idx, task, result

    # Use concurrent workers when running API mode for higher GPU utilization
    n_workers = args.concurrency if args.api_url else 1
    completed = 0

    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for idx, task in enumerate(tasks):
                fut = pool.submit(_process_and_save, (idx, task))
                futures[fut] = idx

            for fut in as_completed(futures):
                completed += 1
                try:
                    idx, task, result = fut.result()
                except Exception as e:
                    print(f"  ERROR: task {futures[fut]}: {e}")
                    continue

                n_mols = len(result.get("molecules", []))
                n_labeled = sum(1 for m in result.get("molecules", []) if m.get("compound_label"))
                n_valid = sum(1 for m in result.get("molecules", []) if m.get("rdkit_valid"))
                total_processed += n_mols
                total_labeled += n_labeled
                total_valid += n_valid

                elapsed = time.time() - t0
                rate = elapsed / completed
                label_pct = (total_labeled / total_processed * 100
                             if total_processed > 0 else 0)
                valid_pct = (total_valid / total_processed * 100
                             if total_processed > 0 else 0)

                if completed % 5 == 0 or completed == total_pages or completed <= 3:
                    print(f"  [{completed}/{total_pages}] {task['source_rel']}: "
                          f"{n_mols} mols, {n_labeled} labeled, {n_valid} valid SMILES | "
                          f"cum: {label_pct:.0f}% labeled, {valid_pct:.0f}% valid, "
                          f"{rate:.1f}s/page")
    else:
        for i, task in enumerate(tasks, 1):
            _, _, result = _process_and_save((i - 1, task))
            completed = i

            n_mols = len(result.get("molecules", []))
            n_labeled = sum(1 for m in result.get("molecules", []) if m.get("compound_label"))
            n_valid = sum(1 for m in result.get("molecules", []) if m.get("rdkit_valid"))
            total_processed += n_mols
            total_labeled += n_labeled
            total_valid += n_valid

            elapsed = time.time() - t0
            rate = elapsed / i if i > 0 else 0
            label_pct = (total_labeled / total_processed * 100
                         if total_processed > 0 else 0)
            valid_pct = (total_valid / total_processed * 100
                         if total_processed > 0 else 0)

            if i % 5 == 0 or i == total_pages or i <= 3:
                print(f"  [{i}/{total_pages}] {task['source_rel']}: "
                      f"{n_mols} mols, {n_labeled} labeled, {n_valid} valid SMILES | "
                      f"cum: {label_pct:.0f}% labeled, {valid_pct:.0f}% valid, "
                      f"{rate:.1f}s/page")

    elapsed = time.time() - t0
    label_pct = (total_labeled / total_processed * 100 if total_processed > 0 else 0)
    valid_pct = (total_valid / total_processed * 100 if total_processed > 0 else 0)

    print("=" * 60)
    print(f"Done! {total_processed} molecules from {total_pages} pages in {elapsed:.1f}s")
    print(f"  Labeled: {total_labeled}/{total_processed} ({label_pct:.1f}%)")
    print(f"  Valid SMILES: {total_valid}/{total_processed} ({valid_pct:.1f}%)")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
