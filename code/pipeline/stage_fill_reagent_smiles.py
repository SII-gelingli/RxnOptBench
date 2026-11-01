#!/usr/bin/env python3
"""
Post-process stage10 output: fill missing SMILES for text-only reagents
(CO, NH4OAc, H2, Cs2CO3, acetic acid, ...) that OCSR never captures.

Strategy (in order):
  1. Built-in high-confidence dict for ~80 very common reagents/solvents/bases.
  2. Local disk cache (pubchem_cache.json at repo root).
  3. PubChem REST name→CanonicalSMILES lookup (network needed).
     Results get written back to the cache.

Targets: every mol in a table's
  reaction.{substrates|products|catalysts|ligands}
  AND every expanded_row's reaction.{...} (same roles)
whose `smiles` is empty/None AND has a non-empty `description` (or a
`label` that itself is a plain chemical name rather than a short code).

The rewritten mol gets:
  smiles        = "<resolved>"
  smiles_source = "name_lookup"
  name_lookup_query = "<query used>"
  name_lookup_via   = "builtin" | "cache" | "pubchem"

Usage:
  python pipeline/stage_fill_reagent_smiles.py
  python pipeline/stage_fill_reagent_smiles.py --dry-run
  python pipeline/stage_fill_reagent_smiles.py --offline           # no network
  python pipeline/stage_fill_reagent_smiles.py --journal NatCommun \
      --doi 10.1038__s41467-025-66742-4_si
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "screening_tables_expanded"
DEFAULT_CACHE = BASE_DIR / "pubchem_cache.json"

# Built-in, hand-curated SMILES for very common reagents. These are trusted
# more than PubChem's first hit for ambiguous short names.
BUILTIN = {
    # Gases / small molecules
    "co": "[C-]#[O+]",
    "carbon monoxide": "[C-]#[O+]",
    "co2": "O=C=O",
    "carbon dioxide": "O=C=O",
    "h2": "[H][H]",
    "hydrogen": "[H][H]",
    "o2": "O=O",
    "oxygen": "O=O",
    "n2": "N#N",
    "nitrogen": "N#N",
    "nh3": "N",
    "ammonia": "N",
    "h2o": "O",
    "water": "O",
    "h2o2": "OO",
    "hcl": "Cl",
    "hydrochloric acid": "Cl",
    "hbr": "Br",
    "hf": "F",
    "hi": "I",
    # Salts / bases / acids
    "nh4oac": "[NH4+].CC(=O)[O-]",
    "nh4cl": "[NH4+].[Cl-]",
    "ammonium acetate": "[NH4+].CC(=O)[O-]",
    "ammonium chloride": "[NH4+].[Cl-]",
    "acoh": "CC(=O)O",
    "aceticacid": "CC(=O)O",
    "acetic acid": "CC(=O)O",
    "tfa": "OC(=O)C(F)(F)F",
    "trifluoroacetic acid": "OC(=O)C(F)(F)F",
    "tsoh": "Cc1ccc(S(=O)(=O)O)cc1",
    "ptsa": "Cc1ccc(S(=O)(=O)O)cc1",
    "p-toluenesulfonic acid": "Cc1ccc(S(=O)(=O)O)cc1",
    "msoh": "CS(=O)(=O)O",
    "methanesulfonic acid": "CS(=O)(=O)O",
    "phoh": "Oc1ccccc1",
    "phenol": "Oc1ccccc1",
    "naoh": "[Na+].[OH-]",
    "koh": "[K+].[OH-]",
    "liooh": "[Li+].[OH-]",
    "na2co3": "[Na+].[Na+].[O-]C([O-])=O",
    "k2co3": "[K+].[K+].[O-]C([O-])=O",
    "cs2co3": "[Cs+].[Cs+].[O-]C([O-])=O",
    "nahco3": "[Na+].OC([O-])=O",
    "khco3": "[K+].OC([O-])=O",
    "k3po4": "[K+].[K+].[K+].[O-]P([O-])([O-])=O",
    "na2so4": "[Na+].[Na+].[O-]S([O-])(=O)=O",
    "mgso4": "[Mg+2].[O-]S([O-])(=O)=O",
    "naotbu": "[Na+].CC(C)(C)[O-]",
    "kotbu": "[K+].CC(C)(C)[O-]",
    "lda": "CC(C)[N-]C(C)C.[Li+]",
    "nhmds": "[Li]N([Si](C)(C)C)[Si](C)(C)C",  # note: NaHMDS, KHMDS use Na/K
    "nahmds": "[Na]N([Si](C)(C)C)[Si](C)(C)C",
    "khmds": "[K]N([Si](C)(C)C)[Si](C)(C)C",
    "lihmds": "[Li]N([Si](C)(C)C)[Si](C)(C)C",
    # Amines
    "et3n": "CCN(CC)CC",
    "triethylamine": "CCN(CC)CC",
    "dipea": "CC(C)N(C(C)C)CC",
    "i-pr2net": "CC(C)N(C(C)C)CC",
    "hunig's base": "CC(C)N(C(C)C)CC",
    "dbu": "N=1CCCCCC=2N1CCC2",
    "dabco": "C1CN2CCN1CC2",
    "dmap": "CN(C)c1ccncc1",
    "pyridine": "c1ccncc1",
    "n-methylmorpholine": "CN1CCOCC1",
    "nmm": "CN1CCOCC1",
    # Solvents
    "meoh": "CO",
    "methanol": "CO",
    "etoh": "CCO",
    "ethanol": "CCO",
    "iproh": "CC(C)O",
    "i-proh": "CC(C)O",
    "isopropanol": "CC(C)O",
    "n-buoh": "CCCCO",
    "t-buoh": "CC(C)(C)O",
    "thf": "C1CCOC1",
    "tetrahydrofuran": "C1CCOC1",
    "2-methf": "CC1CCCO1",
    "2-methyltetrahydrofuran": "CC1CCCO1",
    "dioxane": "C1COCCO1",
    "1,4-dioxane": "C1COCCO1",
    "dme": "COCCOC",
    "dmf": "CN(C)C=O",
    "dmac": "CC(=O)N(C)C",
    "dmso": "CS(C)=O",
    "dcm": "ClCCl",
    "mc": "ClCCl",
    "chcl3": "ClC(Cl)Cl",
    "chloroform": "ClC(Cl)Cl",
    "ccl4": "ClC(Cl)(Cl)Cl",
    "dce": "ClCCCl",
    "1,2-dce": "ClCCCl",
    "mecn": "CC#N",
    "acetonitrile": "CC#N",
    "meno2": "C[N+](=O)[O-]",
    "nitromethane": "C[N+](=O)[O-]",
    "hexanes": "CCCCCC",
    "hexane": "CCCCCC",
    "cyclohexane": "C1CCCCC1",
    "pentane": "CCCCC",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "xylene": "Cc1ccccc1C",
    "mesitylene": "Cc1cc(C)cc(C)c1",
    "pfh": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # perfluorohexane
    "hfip": "OC(C(F)(F)F)C(F)(F)F",
    "tfe": "OCC(F)(F)F",
    "et2o": "CCOCC",
    "ether": "CCOCC",
    "diethyl ether": "CCOCC",
    "mtbe": "COC(C)(C)C",
    "etoac": "CCOC(C)=O",
    "ethyl acetate": "CCOC(C)=O",
    "acetone": "CC(C)=O",
    # Oxidants / reductants
    "nabh4": "[BH4-].[Na+]",
    "naboh": "[BH4-].[Na+]",
    "libh4": "[BH4-].[Li+]",
    "nabh3cn": "[Na+].[BH3-]C#N",
    "nabh(oac)3": "CC(=O)O[BH-](OC(C)=O)OC(C)=O.[Na+]",
    "lialh4": "[AlH4-].[Li+]",
    "dibal-h": "CC(C)CC(C)C[AlH]CC(C)C",
    "dibal": "CC(C)CC(C)C[AlH]CC(C)C",
    "9-bbn": "B1CCCCCCCC1",
    "tbhp": "CC(C)(C)OO",
    "mcpba": "O=C(OO)c1cccc(Cl)c1",
    "dmp": "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21",
    "dess-martin": "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21",
    # Ligands / catalysts (very common short codes)
    "pph3": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    "dppe": "c1ccc(P(CCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1",
    "dppp": "c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1",
    "dppb": "c1ccc(P(CCCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1",
    "dppf": "c1ccc(P(c2ccccc2)c2cccc2[Fe]3456789c1cccc1)cc1",  # ferrocene linkage hard to encode
    # Simple organics
    "ch2i2": "ICI",
    "tms-cl": "C[Si](C)(C)Cl",
    "tmscl": "C[Si](C)(C)Cl",
    "tms-cn": "C[Si](C)(C)C#N",
    "tmscn": "C[Si](C)(C)C#N",
    "tbaf": "CCCC[N+](CCCC)(CCCC)CCCC.[F-]",
    "tbai": "CCCC[N+](CCCC)(CCCC)CCCC.[I-]",
}


def _norm_query(s: str) -> str:
    """Normalize a query key for dict lookup. Lowercase, strip, collapse
    whitespace, drop surrounding punctuation/parens."""
    if not s:
        return ""
    t = s.strip().lower()
    # strip leading/trailing quotes and parens
    t = re.sub(r"^[\s\"'\(\[]+|[\s\"'\)\]]+$", "", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def resolve_builtin(query: str) -> str | None:
    k = _norm_query(query)
    if not k:
        return None
    if k in BUILTIN:
        return BUILTIN[k]
    # try without spaces/hyphens (e.g. "p-tsoh" → "ptsoh")
    k2 = re.sub(r"[-\s]", "", k)
    return BUILTIN.get(k2)


# ── PubChem REST ─────────────────────────────────────────────────────────────

_PUBCHEM_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                "{q}/property/CanonicalSMILES/TXT")
_last_call_ts = [0.0]
_pubchem_lock = Lock()


def pubchem_lookup(query: str, timeout: int = 10,
                   min_interval: float = 0.25) -> str | None:
    """Return canonical SMILES for the given name, or None on failure.

    Throttles to `min_interval` seconds between requests to be polite.
    """
    if not query:
        return None
    q = urllib.parse.quote(query.strip())
    url = _PUBCHEM_URL.format(q=q)
    with _pubchem_lock:
        dt = time.time() - _last_call_ts[0]
        if dt < min_interval:
            time.sleep(min_interval - dt)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "RxnOptBench/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    return None
                text = r.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            text = ""
        finally:
            _last_call_ts[0] = time.time()
    if not text:
        return None
    # Take first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def validate_smiles(smi: str) -> bool:
    try:
        from rdkit import Chem
    except ImportError:
        # Without rdkit, accept anything non-empty without obvious breakage
        return bool(smi) and "*" not in smi
    if not smi:
        return False
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


# ── Query selection ──────────────────────────────────────────────────────────

# Labels that are pure placeholder codes — we should NOT feed these to PubChem.
_LABEL_CODE_RX = re.compile(r"^([0-9]+[a-z]?|[Ll][0-9]+[a-z]?|[Cc]at[-\s]?[0-9]+|"
                            r"[LBC][0-9]+|[A-Z][0-9]{1,2}|S[-\s]?[0-9]+)$")


def is_placeholder_label(label: str) -> bool:
    if not label:
        return True
    return bool(_LABEL_CODE_RX.match(label.strip()))


def pick_query(mol: dict) -> str:
    """Prefer description; fall back to label if the label is a real name."""
    desc = (mol.get("description") or "").strip()
    if desc:
        return desc
    lbl = (mol.get("label") or "").strip()
    if lbl and not is_placeholder_label(lbl):
        return lbl
    return ""


# ── Cache ────────────────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_cache(cache: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True))
    tmp.replace(path)


# ── Core ─────────────────────────────────────────────────────────────────────

ROLE_KEYS = ("substrates", "products", "catalysts", "ligands")


def resolve(query: str, cache: dict, offline: bool,
            stats: dict) -> tuple[str | None, str | None]:
    """Return (smiles, via) or (None, None). Updates cache in place.

    via ∈ {"builtin", "cache", "pubchem"}.
    """
    if not query:
        return None, None
    # 1. builtin
    b = resolve_builtin(query)
    if b:
        stats["builtin"] += 1
        return b, "builtin"
    # 2. cache
    key = _norm_query(query)
    if key in cache:
        v = cache[key]
        if v:
            stats["cache_hit"] += 1
            return v, "cache"
        stats["cache_neg"] += 1
        return None, None  # cached negative
    # 3. pubchem
    if offline:
        return None, None
    smi = pubchem_lookup(query)
    if smi and validate_smiles(smi):
        cache[key] = smi
        stats["pubchem_hit"] += 1
        return smi, "pubchem"
    # record negative result so we don't keep hitting the API
    cache[key] = ""
    stats["pubchem_miss"] += 1
    return None, None


def process_mol_list(mols, cache, offline, stats):
    """Mutate in place. Return count of SMILES filled in this list."""
    if not isinstance(mols, list):
        return 0
    filled = 0
    for mol in mols:
        if not isinstance(mol, dict):
            continue
        cur = (mol.get("smiles") or "").strip()
        if cur:  # already has SMILES
            continue
        q = pick_query(mol)
        if not q:
            stats["no_query"] += 1
            continue
        smi, via = resolve(q, cache, offline, stats)
        if not smi:
            continue
        mol["smiles"] = smi
        mol["smiles_source"] = "name_lookup"
        mol["name_lookup_query"] = q
        mol["name_lookup_via"] = via
        mol["rdkit_valid"] = validate_smiles(smi)
        filled += 1
    return filled


def process_table(table, cache, offline, stats):
    filled = 0
    rxn = table.get("reaction") or {}
    for role in ROLE_KEYS:
        filled += process_mol_list(rxn.get(role) or [], cache, offline, stats)
    for er in table.get("expanded_rows", []) or []:
        er_rxn = er.get("reaction") or {}
        for role in ROLE_KEYS:
            filled += process_mol_list(er_rxn.get(role) or [], cache, offline, stats)
    return filled


def process_file(path: Path, cache, offline, stats, dry_run: bool) -> int:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0
    tables = data if isinstance(data, list) else [data]
    total = 0
    for tbl in tables:
        if isinstance(tbl, dict):
            total += process_table(tbl, cache, offline, stats)
    if total and not dry_run:
        path.write_text(json.dumps(data if isinstance(data, list) else data,
                                   ensure_ascii=False, indent=2))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--offline", action="store_true",
                    help="Skip PubChem; use only builtin + cache")
    ap.add_argument("--journal", default=None, help="filter to one journal")
    ap.add_argument("--doi", default=None, help="filter to one DOI dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--save-cache-every", type=int, default=200,
                    help="Flush cache to disk every N new queries")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    cache_path = Path(args.cache)

    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve symlinks so writes land on the real directory
    input_dir = input_dir.resolve()

    print(f"Input:  {input_dir}")
    print(f"Cache:  {cache_path}  (exists={cache_path.exists()})")
    print(f"Mode:   {'OFFLINE' if args.offline else 'online (PubChem enabled)'}")
    print(f"Dry run: {args.dry_run}")

    cache = load_cache(cache_path)
    cache_start_size = len(cache)

    stats = {"builtin": 0, "cache_hit": 0, "cache_neg": 0,
             "pubchem_hit": 0, "pubchem_miss": 0, "no_query": 0}

    files_touched = 0
    mols_filled = 0
    files_scanned = 0
    new_since_flush = 0

    for jdir in sorted(input_dir.iterdir()):
        if not jdir.is_dir():
            continue
        if args.journal and jdir.name != args.journal:
            continue
        for doidir in sorted(jdir.iterdir()):
            if not doidir.is_dir():
                continue
            if args.doi and doidir.name != args.doi:
                continue
            for pf in sorted(doidir.glob("p*.json")):
                files_scanned += 1
                before = len(cache)
                n = process_file(pf, cache, args.offline, stats, args.dry_run)
                if n:
                    files_touched += 1
                    mols_filled += n
                new_since_flush += (len(cache) - before)
                if new_since_flush >= args.save_cache_every and not args.dry_run:
                    save_cache(cache, cache_path)
                    new_since_flush = 0
                if files_scanned % 100 == 0:
                    print(f"  scanned={files_scanned} touched={files_touched} "
                          f"filled={mols_filled} cache={len(cache)}",
                          flush=True)

    if not args.dry_run and len(cache) != cache_start_size:
        save_cache(cache, cache_path)

    print()
    print("=" * 60)
    print(f"Files scanned:  {files_scanned}")
    print(f"Files touched:  {files_touched}")
    print(f"Mols filled:    {mols_filled}")
    print(f"Cache size:     {cache_start_size} → {len(cache)}  "
          f"(+{len(cache) - cache_start_size})")
    print("Resolution sources:")
    for k in ("builtin", "cache_hit", "pubchem_hit"):
        print(f"  {k:15s} {stats[k]}")
    print("Skipped / misses:")
    for k in ("no_query", "cache_neg", "pubchem_miss"):
        print(f"  {k:15s} {stats[k]}")
    if args.dry_run:
        print("(DRY RUN — no files or cache writes)")


if __name__ == "__main__":
    main()
