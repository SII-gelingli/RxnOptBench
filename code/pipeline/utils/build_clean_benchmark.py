#!/usr/bin/env python3
"""Build the 'clean' benchmark from stage10_expanded/.

Criterion: a row-level reaction (expanded_rows[i].reaction) is kept iff
  - substrates is non-empty
  - products is non-empty
  - every substrate and product SMILES is non-empty and contains no '*'

Each kept row becomes one record in reactions_qualified.json.
SMILES are canonicalized with RDKit when possible (kept as-is on failure).
The same records are grouped into ``by_journal/<journal>/<table>.json`` for
the downstream VLM schema-conversion stage.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

BASE = Path(__file__).resolve().parents[2]
STAGE10 = BASE / "screening_tables_expanded"
OUT_DIR = BASE / "benchmark_release"
OUT_DATA = OUT_DIR / "reactions_qualified.json"
OUT_STATS = OUT_DIR / "stats.json"
OUT_README = OUT_DIR / "README.md"
OUT_TABLES = OUT_DIR / "by_journal"

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")
_COND_SMI_RE = re.compile(r"\[SMILES:\s*([^\]]+?)\s*\]")


def _canon(smi: str) -> str:
    if not smi:
        return smi
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return smi


def _slug(s: str, n: int = 20) -> str:
    return (s or "")[:n]


def _filename_slug(value: str, limit: int, fallback: str = "") -> str:
    return _SLUG_RE.sub("_", value or "").strip("_")[:limit] or fallback


def _has_star(s: str) -> bool:
    return "*" in (s or "")


def _row_clean(rxn: dict) -> bool:
    subs = rxn.get("substrates") or []
    prods = rxn.get("products") or []
    if not subs or not prods:
        return False
    for m in subs + prods:
        s = (m or {}).get("smiles", "")
        if not s or _has_star(s):
            return False
    # Also reject rows whose condition strings carry a [SMILES: ...] with '*'
    for c in rxn.get("conditions") or []:
        if not isinstance(c, str):
            continue
        for m in _COND_SMI_RE.finditer(c):
            if _has_star(m.group(1)):
                return False
    return True


def _mol_record(m: dict, include_yield: bool = False) -> dict:
    rec = {
        "label": m.get("label", "") or "",
        "smiles": _canon(m.get("smiles", "")),
        "description": m.get("description", "") or "",
    }
    if include_yield:
        rec["yield"] = m.get("yield")
        rec["ee"] = m.get("ee")
        rec["dr"] = m.get("dr")
    return rec


def build():
    records: list[dict] = []
    seen_ids: set = set()
    by_journal = Counter()
    by_doi = set()
    by_table = set()
    with_ee = 0
    with_dr = 0
    subs_dist = Counter()
    prods_dist = Counter()
    total_rows = 0
    total_tables_seen = 0
    tables_kept = 0
    grouped_tables: dict[str, dict] = {}
    table_rel_by_key: dict[tuple, str] = {}
    used_table_paths: dict[str, tuple] = {}

    t0 = time.time()
    for f in sorted(STAGE10.rglob("p*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for e in data:
            if not isinstance(e, dict):
                continue
            journal = e.get("journal", "")
            doi = e.get("doi", "")
            page = e.get("page", "")
            table_title = e.get("table_title", "") or ""
            top_rxn = e.get("reaction") or {}
            conditions_on_arrow = top_rxn.get("conditions_on_arrow") or []
            columns = e.get("columns") or []
            column_footnotes = e.get("column_footnotes") or {}
            raw_rows = e.get("rows") or []

            ers = e.get("expanded_rows", []) or []
            total_rows += len(ers)
            # Whole-table gate: every row must be clean; empty tables are skipped.
            if not ers:
                total_tables_seen += 1
                continue
            total_tables_seen += 1
            all_clean = True
            for er in ers:
                rxn = er.get("reaction") or {}
                if not _row_clean(rxn):
                    all_clean = False
                    break
            if not all_clean:
                continue
            tables_kept += 1

            for er in ers:
                rxn = er.get("reaction") or {}
                subs = [_mol_record(m) for m in rxn.get("substrates") or []]
                prods = [_mol_record(m, include_yield=True)
                         for m in rxn.get("products") or []]

                row_idx = er.get("row_index")
                rid = f"{journal}/{doi}/p{page}/t{_slug(table_title)}/r{row_idx}"

                if rid in seen_ids:
                    # Duplicate stage10 table entry (same page/title repeated);
                    # keep only the first occurrence.
                    continue
                seen_ids.add(rid)

                raw_row = (
                    raw_rows[row_idx]
                    if isinstance(row_idx, int) and 0 <= row_idx < len(raw_rows)
                    else None
                )

                rec = {
                    "id": rid,
                    "row_index": row_idx,
                    "doi": doi,
                    "journal": journal,
                    "page": page,
                    "table_title": table_title,
                    "columns": columns,
                    "column_footnotes": column_footnotes,
                    "raw_row": raw_row,
                    "substrates": subs,
                    "products": prods,
                    "conditions": rxn.get("conditions") or [],
                    "conditions_on_arrow": conditions_on_arrow,
                    "notes": rxn.get("notes", "") or "",
                }
                records.append(rec)

                table_key = (journal, doi, page, table_title)
                rel_path = table_rel_by_key.get(table_key)
                if rel_path is None:
                    doi_dir = _filename_slug(
                        f.parent.name, 100, fallback="unknown_doi"
                    )
                    title_part = _filename_slug(table_title, 40)
                    stem = f"{doi_dir}__p{page}__{title_part}"
                    rel_path = f"{journal}/{stem}.json"
                    suffix = 2
                    while (
                        rel_path in used_table_paths
                        and used_table_paths[rel_path] != table_key
                    ):
                        rel_path = f"{journal}/{stem}__{suffix}.json"
                        suffix += 1
                    used_table_paths[rel_path] = table_key
                    table_rel_by_key[table_key] = rel_path
                    grouped_tables[rel_path] = {
                        "doi": doi,
                        "journal": journal,
                        "page": page,
                        "table_title": table_title,
                        "reaction": top_rxn,
                        "columns": columns,
                        "column_footnotes": column_footnotes,
                        "footnotes": e.get("footnotes") or {},
                        "caption": e.get("caption", "") or "",
                        "rows": [],
                    }
                grouped_tables[rel_path]["rows"].append(rec)

                by_journal[journal] += 1
                if doi:
                    by_doi.add(doi)
                by_table.add(f"{doi}/p{page}/{table_title}")
                if any(p.get("ee") for p in prods):
                    with_ee += 1
                if any(p.get("dr") for p in prods):
                    with_dr += 1
                subs_dist[min(len(subs), 4)] += 1
                prods_dist[min(len(prods), 4)] += 1

    print(f"Scanned in {time.time()-t0:.1f}s.  Records: {len(records)} / {total_rows}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    for rel_path, table in sorted(grouped_tables.items()):
        output = OUT_TABLES / rel_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(table, ensure_ascii=False, indent=2) + "\n"
        )
    stats = {
        "total_rows": total_rows,
        "qualified_rows": len(records),
        "criteria": "whole_table_clean_no_star_in_subs_prods_or_cond_smiles",
        "tables_seen": total_tables_seen,
        "tables_kept": tables_kept,
        "schema_input_tables": len(grouped_tables),
        "with_ee": with_ee,
        "with_dr": with_dr,
        "unique_doi": len(by_doi),
        "unique_tables": len(by_table),
        "by_journal": dict(by_journal.most_common()),
        "subs_dist_1_2_3_4+": {str(k): subs_dist.get(k, 0) for k in (1, 2, 3, 4)},
        "prods_dist_1_2_3_4+": {str(k): prods_dist.get(k, 0) for k in (1, 2, 3, 4)},
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    # README
    jrows = "\n".join(f"| {j} | {c} |" for j, c in by_journal.most_common())
    readme = f"""# RxnOptBench — Clean Subset

## 概述

从 stage10_expanded/ 中筛选的"干净"反应：**整张表的所有行**都必须满足 —
反应物、产物，以及条件中的 `[SMILES: …]` 均非空且不含 `*` 通配符。
只要该表有任何一行不合格，整张表都不会被收录。
SMILES 已用 RDKit canonical 化。

## 数据规模

| 指标 | 数量 |
|------|------|
| 合格反应条目 | {len(records):,} |
| 原始行级反应总数 | {total_rows:,} |
| 唯一 DOI | {len(by_doi):,} |
| 唯一筛选表 | {len(by_table):,} |
| 覆盖期刊 | {len(by_journal)} |
| 含 ee 值 | {with_ee:,} ({100*with_ee/max(len(records),1):.1f}%) |
| 含 dr 值 | {with_dr:,} ({100*with_dr/max(len(records),1):.1f}%) |

## 期刊分布

| 期刊 | 反应数 |
|------|--------|
{jrows}

## 数据格式 (`reactions_qualified.json`)

```json
{{
  "id": "journal/doi/p{{page}}/t{{title_slug}}/r{{row_index}}",
  "row_index": 0,
  "doi": "10.xxxx/...",
  "journal": "JACS",
  "page": 13,
  "table_title": "Table S1. ...",
  "columns": ["Entry", "Substrate", "Yield (%)", ...],
  "column_footnotes": {{"Yield (%)": "b: Isolated yields.", ...}},
  "raw_row": {{"Entry": "1", "Substrate": "1a", "Yield (%)": "85", ...}},
  "substrates": [{{"label": "1a", "smiles": "...", "description": "..."}}],
  "products":   [{{"label": "3",  "smiles": "...", "description": "...",
                  "yield": "85%", "ee": "92%", "dr": "10:1"}}],
  "conditions": ["CuCl (10 mol%)", "DCE", ...],
  "conditions_on_arrow": [...],
  "notes": "..."
}}
```

## 筛选标准

对 `stage10_expanded/**/*.json` 中每个 `expanded_rows[i].reaction` 判断：

1. `substrates` 非空，`products` 非空
2. 每个 substrate/product 的 `smiles` 非空且**不含 `*`**
3. `conditions` 中任一 `[SMILES: …]` 内嵌 SMILES 也**不含 `*`**
4. **整张表的每一行都必须同时满足上述 1–3**，否则整张表被丢弃
   （避免同一张表里掺入半解析失败的行）

注：本次未额外要求 yield 或 conditions 非空。

## 底物 / 产物分布

| 底物数 | 反应数 | 产物数 | 反应数 |
|--------|--------|--------|--------|
""" + "\n".join(
        f"| {k if k<4 else '4+'} | {subs_dist.get(k,0):,} | {k if k<4 else '4+'} | {prods_dist.get(k,0):,} |"
        for k in (1, 2, 3, 4)
    ) + f"""

## SMILES 来源

SMILES 源自 `mol_smiles_decimer[_si]` 的 `smiles_rexp` 字段（Stage 7 OCSR
输出 → re-expand 带立体信息 + 超原子展开），并已通过
`pipeline/utils/propagate_rexp_to_stage10.py` 传播到 stage10_expanded。

## 文件说明

- `reactions_qualified.json` — {len(records):,} 条合格反应
- `stats.json` — 统计元信息
- `by_journal/` — 按表组织的干净记录，作为 VLM schema 转换输入
- `README.md` — 本文件
- `visualize_benchmark.py` — 可视化工具（保留自旧版本，如需适配新 schema 可手动调整）
"""
    OUT_README.write_text(readme)
    print(f"Wrote {OUT_DATA.name} / {OUT_STATS.name} / {OUT_README.name}")


def main():
    global STAGE10, OUT_DIR, OUT_DATA, OUT_STATS, OUT_README, OUT_TABLES
    parser = argparse.ArgumentParser(
        description="Build a clean row-level reaction benchmark"
    )
    parser.add_argument("--input-dir", default=str(STAGE10))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    STAGE10 = Path(args.input_dir)
    OUT_DIR = Path(args.output_dir)
    OUT_DATA = OUT_DIR / "reactions_qualified.json"
    OUT_STATS = OUT_DIR / "stats.json"
    OUT_README = OUT_DIR / "README.md"
    OUT_TABLES = OUT_DIR / "by_journal"
    build()


if __name__ == "__main__":
    main()
