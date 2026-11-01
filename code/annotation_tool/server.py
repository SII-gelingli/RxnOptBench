#!/usr/bin/env python3
"""
RxnOptBench — Human Review Tool

Web-based annotation tool for the human review gate.
Supports:
  - Browse tables grouped by journal
  - Per-table actions: ACCEPT / SKIP / REJECT (one click each)
  - Per-field edits with an explicit Save action
  - Delete individual entries / nested items
  - View source images (entry_X_Y.png) alongside JSON
  - Filter by journal and status; resumable state via status.json

Storage:
  - Original data is read-only.
  - Edited copies live in <ANN_ROOT>/<journal>/<filename>.json
  - Per-table status in <ANN_ROOT>/status.json

Zero external deps — stdlib only.

Run:
  python3 server.py
  open http://localhost:5050
"""

import http.server
import json
import mimetypes
import os
import socketserver
import sys
import threading
import urllib.parse
from datetime import datetime
from io import BytesIO
from pathlib import Path

# RDKit for SMILES → PNG rendering
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")          # silence parse warnings
    _RDKIT_OK = True
except Exception as _e:
    _RDKIT_OK = False
    print(f"[warn] rdkit not available: {_e}", file=sys.stderr)

# ---------------- config ----------------
RELEASE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get(
    "OMB_ANNOTATION_DATA_ROOT",
    RELEASE_ROOT / "review_data" / "input",
))
ANN_ROOT = Path(os.environ.get(
    "OMB_ANNOTATION_OUTPUT_ROOT",
    RELEASE_ROOT / "review_data" / "annotations",
))
SRC_ROOT = Path(os.environ.get(
    "OMB_ANNOTATION_IMAGE_ROOT",
    RELEASE_ROOT / "review_data" / "images",
))
# fallback for per-entry crops since _rolefixed has no _imgs subdirs
CROPS_FALLBACK = Path(os.environ.get(
    "OMB_ANNOTATION_CROPS_ROOT",
    RELEASE_ROOT / "review_data" / "crops",
))
PORT      = int(os.environ.get("PORT", 5050))
HOST      = os.environ.get("HOST", "127.0.0.1")
STATUS_FILE = ANN_ROOT / "status.json"

ANN_ROOT.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()

# ---------------- caches ----------------
import time as _time
_TABLE_CACHE = {"ts": 0.0, "data": None}      # list_tables() result, 60s TTL
_STATUS_CACHE = {"ts": 0.0, "data": None}     # status.json, 2s TTL
_TABLE_TTL = 60.0
_STATUS_TTL = 2.0


# ---------------- state ----------------
def load_status() -> dict:
    now = _time.time()
    if _STATUS_CACHE["data"] is not None and now - _STATUS_CACHE["ts"] < _STATUS_TTL:
        return _STATUS_CACHE["data"]
    data = {}
    if STATUS_FILE.exists():
        try:
            data = json.loads(STATUS_FILE.read_text())
        except Exception:
            data = {}
    _STATUS_CACHE["data"] = data
    _STATUS_CACHE["ts"] = now
    return data


def save_status(status: dict) -> None:
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    tmp.replace(STATUS_FILE)
    _STATUS_CACHE["data"] = status        # keep cache hot
    _STATUS_CACHE["ts"] = _time.time()


def set_status_for(rel_path: str, value: str, annotator: str = "anonymous") -> None:
    with LOCK:
        s = load_status()
        s[rel_path] = {
            "status":     value,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "annotator":  annotator,
        }
        save_status(s)


def clear_status_for(rel_path: str) -> None:
    with LOCK:
        s = load_status()
        s.pop(rel_path, None)
        save_status(s)


# ---------------- table I/O ----------------
def list_journals() -> list:
    return sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir() and not d.name.startswith("."))


def list_tables() -> list:
    now = _time.time()
    if _TABLE_CACHE["data"] is not None and now - _TABLE_CACHE["ts"] < _TABLE_TTL:
        return _TABLE_CACHE["data"]
    out = []
    for j in list_journals():
        for f in sorted((DATA_ROOT / j).glob("*.json")):
            out.append({"journal": j, "filename": f.name, "rel_path": f"{j}/{f.name}"})
    _TABLE_CACHE["data"] = out
    _TABLE_CACHE["ts"] = now
    return out


def load_table(rel_path: str):
    edited = ANN_ROOT / rel_path
    if edited.exists():
        return json.loads(edited.read_text()), True
    return json.loads((DATA_ROOT / rel_path).read_text()), False


def save_table(rel_path: str, data) -> None:
    out = ANN_ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(out)


def _crops_dir(rel_path: str) -> Path | None:
    """Return the per-entry crops directory, falling back if DATA_ROOT lacks one."""
    for root in (DATA_ROOT, CROPS_FALLBACK):
        p = root / rel_path
        d = p.with_name(p.stem + "_imgs")
        if d.exists():
            return d
    return None


def list_images(rel_path: str) -> list:
    d = _crops_dir(rel_path)
    if d is None:
        return []
    return sorted(x.name for x in d.glob("*.png"))


def img_path(rel_path: str, fname: str) -> Path:
    d = _crops_dir(rel_path)
    if d is None:
        # return a non-existent path so _file returns 404
        return DATA_ROOT / rel_path
    return d / fname


# cache: journal -> {normalized_dirname -> actual_dirname}
_SRC_DIR_CACHE: dict = {}

def _normalize_doi_token(s: str) -> str:
    # "10.1021__acscatal.4c08055_si"        -> "10_1021_acscatal_4c08055_si"
    # "10.1038__s41929-025-01336-3_si"      -> "10_1038_s41929_025_01336_3_si"
    s = s.replace(".", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _src_dir_for_journal(journal: str) -> dict:
    if journal in _SRC_DIR_CACHE:
        return _SRC_DIR_CACHE[journal]
    j = SRC_ROOT / journal
    m = {}
    if j.exists():
        for d in j.iterdir():
            if d.is_dir():
                m[_normalize_doi_token(d.name)] = d.name
    _SRC_DIR_CACHE[journal] = m
    return m

_MOL_CACHE: dict = {}                   # (smi, w, h) -> PNG bytes
_MOL_CACHE_MAX = 4000

def render_mol_png(smi: str, w: int = 200, h: int = 150):
    """SMILES -> PNG bytes; None if RDKit cannot parse it."""
    if not _RDKIT_OK or not smi:
        return None
    key = (smi, w, h)
    if key in _MOL_CACHE:
        return _MOL_CACHE[key]
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        img = Draw.MolToImage(m, size=(w, h))
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
    except Exception:
        return None
    if len(_MOL_CACHE) > _MOL_CACHE_MAX:
        _MOL_CACHE.pop(next(iter(_MOL_CACHE)))
    _MOL_CACHE[key] = data
    return data


def source_page_path(rel_path: str) -> Path | None:
    """
    Resolve the original SI page PNG for a given table JSON.
    e.g. "ACSCatal/10_1021_acscatal_4c08055_si__p7__Figure_S8_xxx.json"
         -> all_si_pages_hires_2025/ACSCatal/10.1021__acscatal.4c08055_si/p7.png
    """
    try:
        journal, fname = rel_path.split("/", 1)
    except ValueError:
        return None
    stem = fname[:-5] if fname.endswith(".json") else fname
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    doi_token = parts[0]                # e.g. "10_1021_acscatal_4c08055_si"
    page_tok  = parts[1]                # e.g. "p7"
    dirs = _src_dir_for_journal(journal)
    actual = dirs.get(doi_token)
    if not actual:
        return None
    p = SRC_ROOT / journal / actual / f"{page_tok}.png"
    return p if p.exists() else None


# ---------------- HTTP handler ----------------
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "OMBReview/1.0"
    protocol_version = "HTTP/1.1"          # enable keep-alive (saves TLS handshakes)

    # ---- helpers ----
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str, code=200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, p: Path, immutable: bool = False):
        if not p.exists():
            self.send_error(404)
            return
        ctype, _ = mimetypes.guess_type(str(p))
        ctype = ctype or "application/octet-stream"
        data = p.read_bytes()
        # If client sent If-None-Match and it matches, return 304
        etag = f'"{p.stat().st_mtime_ns:x}-{len(data):x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", etag)
        if immutable:
            # source-page PNGs and per-entry crops never change in-place
            self.send_header("Cache-Control", "public, max-age=86400, immutable")
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, loc):
        self.send_response(303)
        self.send_header("Location", loc)
        self.send_header("Content-Length", "0")    # required for HTTP/1.1 keep-alive
        self.end_headers()

    def log_message(self, *a, **kw):
        pass  # quiet

    # ---- routes ----
    def _route_tail(self, path: str) -> str:
        """Match by last segment so any reverse-proxy prefix is tolerated."""
        p = path.rstrip("/")
        if not p:
            return "/"
        # last segment, e.g. "/proxy/5050/api/table" -> "api/table"
        # match common 2-segment routes first
        for tail in ("api/table", "api/status", "api/export", "api/save"):
            if p.endswith("/" + tail) or p == tail:
                return "/" + tail
        seg = p.rsplit("/", 1)[-1]
        if seg in ("annotate", "img", "srcimg", "mol"):
            return "/" + seg
        # treat anything else with empty path as index
        return "/"

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        path = self._route_tail(u.path)

        if path == "/":
            return self._html(render_index(q))

        if path == "/annotate":
            rel = q.get("path", [""])[0]
            if not rel:
                return self._redirect("./")
            return self._html(render_annotate(rel))

        if path == "/img":
            rel = q.get("path", [""])[0]
            img = q.get("img", [""])[0]
            if not rel or not img:
                self.send_error(400)
                return
            return self._file(img_path(rel, img), immutable=True)

        if path == "/srcimg":
            rel = q.get("path", [""])[0]
            if not rel:
                self.send_error(400); return
            sp = source_page_path(rel)
            if sp is None:
                self.send_error(404); return
            return self._file(sp, immutable=True)

        if path == "/mol":
            smi = q.get("smi", [""])[0]
            try:
                w = int(q.get("w", ["200"])[0]); h = int(q.get("h", ["150"])[0])
            except ValueError:
                w, h = 200, 150
            w = max(60, min(w, 600)); h = max(40, min(h, 500))
            if not smi:
                self.send_error(400); return
            data = render_mol_png(smi, w, h)
            if data is None:
                self.send_error(422); return       # unparsable SMILES
            etag = f'"mol-{abs(hash((smi, w, h))):x}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Content-Length", "0")
                self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/api/table":
            rel = q.get("path", [""])[0]
            data, edited = load_table(rel)
            sp = source_page_path(rel)
            return self._json({"rel_path": rel, "data": data, "edited": edited,
                               "images": list_images(rel),
                               "src_page": (sp.name if sp else None),
                               "status": load_status().get(rel, {"status": "pending"})})

        if path == "/api/status":
            return self._json(load_status())

        if path == "/api/export":
            return self._json({"status": load_status(),
                               "ann_root": str(ANN_ROOT)})

        self.send_error(404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        path = self._route_tail(u.path)
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(body_raw)
        except Exception:
            return self._json({"ok": False, "error": "bad JSON"}, 400)

        if path == "/api/save":
            rel = body.get("rel_path", "")
            action = body.get("action", "")
            if not rel:
                return self._json({"ok": False, "error": "missing rel_path"}, 400)

            if action == "edit":
                save_table(rel, body.get("data", {}))
                set_status_for(rel, "edited")
            elif action == "accept":
                set_status_for(rel, "accepted")
            elif action == "skip":
                set_status_for(rel, "skipped")
            elif action == "reject":
                set_status_for(rel, "rejected")
            elif action == "reset":
                clear_status_for(rel)
                edited = ANN_ROOT / rel
                if edited.exists():
                    edited.unlink()
            else:
                return self._json({"ok": False, "error": f"unknown action {action}"}, 400)

            return self._json({"ok": True, "rel_path": rel, "action": action})

        self.send_error(404)


# ============================================================
# HTML rendering
# ============================================================
INDEX_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Helvetica,sans-serif;margin:0;background:#f5f6f8;color:#222}
header{background:#1f2d3d;color:#fff;padding:12px 20px;display:flex;align-items:center;gap:24px}
header h1{margin:0;font-size:18px;font-weight:600}
.summary{display:flex;gap:18px;font-size:13px;color:#cfd8e3}
.summary span b{color:#fff}
.controls{padding:14px 20px;background:#fff;border-bottom:1px solid #e0e3e8;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.controls label{font-size:12px;color:#666;display:flex;flex-direction:column;gap:3px}
.controls select,.controls input{padding:5px 8px;font-size:13px;border:1px solid #cfd4dc;border-radius:4px;background:#fff;min-width:140px}
table.tbl{width:100%;border-collapse:collapse;background:#fff}
table.tbl th,table.tbl td{padding:8px 12px;border-bottom:1px solid #eef0f3;text-align:left;font-size:13px}
table.tbl th{background:#f8f9fb;font-weight:600;color:#555;position:sticky;top:0;z-index:1}
table.tbl tr:hover{background:#f9fbff}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;letter-spacing:.3px}
.b-pending {background:#e9ecf1;color:#5f6c7e}
.b-accepted{background:#dcf2e3;color:#1d6b3a}
.b-edited  {background:#fff4d6;color:#8a6300}
.b-skipped {background:#e7e5fb;color:#54448e}
.b-rejected{background:#fbe1de;color:#a02a1a}
a.go{color:#1f6feb;text-decoration:none;font-weight:500}
a.go:hover{text-decoration:underline}
.toolbar{padding:8px 20px;background:#fff;border-top:1px solid #eef0f3;font-size:12px;color:#777}
"""

ANNOTATE_CSS = """
body{font:13px/1.45 -apple-system,Segoe UI,Helvetica,sans-serif;margin:0;background:#f3f5f8;color:#222}
header{background:#1f2d3d;color:#fff;padding:10px 18px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
header h1{margin:0;font-size:16px;font-weight:600;flex:1;font-family:Menlo,monospace;font-size:13px}
.actions{display:flex;gap:8px}
.actions button{padding:7px 14px;border:0;border-radius:5px;font-size:13px;font-weight:600;cursor:pointer}
.btn-accept{background:#27ae60;color:#fff}
.btn-skip  {background:#7f8c8d;color:#fff}
.btn-reject{background:#c0392b;color:#fff}
.btn-save  {background:#2980b9;color:#fff}
.btn-reset {background:#e1e3e8;color:#333}
.actions button:hover{filter:brightness(1.08)}
.nav{display:flex;gap:8px;align-items:center;color:#cfd8e3;font-size:12px}
.nav a{color:#cfd8e3;text-decoration:none;border:1px solid #4a5a70;padding:4px 10px;border-radius:4px}
.nav a:hover{color:#fff;border-color:#fff}
.status-line{padding:6px 18px;background:#fff;border-bottom:1px solid #e6e8ec;font-size:12px;color:#555}
.status-line .badge{margin-right:8px}
main{display:grid;grid-template-columns:42% 58%;gap:0;height:calc(100vh - 90px)}
#imgs{overflow-y:auto;padding:14px;background:#fff;border-right:1px solid #e0e3e8}
#imgs h3{font-size:13px;color:#555;margin:0 0 10px 0;text-transform:uppercase;letter-spacing:.5px}
#imgs .img-card{margin-bottom:14px;background:#f9fafc;border:1px solid #e6e8ec;border-radius:5px;padding:8px}
#imgs .img-card .lbl{font-family:Menlo,monospace;font-size:11px;color:#666;margin-bottom:5px}
#imgs img{max-width:100%;display:block;border-radius:3px}
#editor{overflow-y:auto;padding:14px 18px}
.entry{background:#fff;border:1px solid #d8dde4;border-radius:5px;margin-bottom:14px;padding:10px 12px}
.entry.collapsed{padding-bottom:10px;background:#f7f8fb}
.entry.collapsed .entry-body{display:none}
.entry-head{display:flex;align-items:center;gap:10px;margin-bottom:8px;border-bottom:1px solid #eef0f3;padding-bottom:6px;cursor:pointer;user-select:none}
.entry.collapsed .entry-head{margin-bottom:0;border-bottom:0;padding-bottom:0}
.entry-head:hover{background:#fafbfd}
.entry-head .caret{font-size:10px;color:#7a8696;width:12px;display:inline-block;transition:transform .15s}
.entry.collapsed .entry-head .caret{transform:rotate(-90deg)}
.entry-head .eid{font-family:Menlo,monospace;font-size:13px;font-weight:600;color:#1f2d3d;background:#eef3fa;padding:2px 8px;border-radius:3px}
.entry-head .preview{color:#888;font-size:11px;font-family:Menlo,monospace;margin-left:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
.entry-head .del{margin-left:auto;background:#fbe1de;color:#a02a1a;border:0;padding:3px 9px;border-radius:3px;font-size:11px;cursor:pointer}
.entry-head .del:hover{background:#f1bdb6}
.bulk-toggle{background:transparent;border:1px solid #cfd4dc;color:#555;padding:3px 10px;border-radius:3px;font-size:11px;cursor:pointer;margin-right:6px}
.bulk-toggle:hover{background:#eef3fa;border-color:#2980b9;color:#2980b9}
.mol-strip{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;padding:10px;background:#fafbfd;border-radius:4px;margin-bottom:10px;border:1px solid #e6e8ec}
.mol-row{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
.mol-cell{background:#fff;border:1px solid #e0e3e8;border-radius:3px;padding:4px;text-align:center;font-size:10px;max-width:140px}
.mol-cell.no-smi{background:#fff5f5;border-color:#f0c8c2}
.mol-cell.bad-smi{background:#fff8e0;border-color:#e5c97a}
.mol-cell img{display:block;max-width:100%;max-height:90px;margin:0 auto}
.mol-lbl{color:#3a4a5a;font-family:Menlo,monospace;font-size:10px;margin-top:3px;font-weight:600}
.mol-arrow{font-size:22px;color:#888;padding:0 6px;font-weight:600}
.kv{display:grid;grid-template-columns:160px 1fr;gap:4px 8px;margin:3px 0}
.kv label{font-size:11px;color:#777;font-weight:500;align-self:center;font-family:Menlo,monospace}
.kv input,.kv textarea,.kv select{font-family:inherit;font-size:12px;padding:4px 6px;border:1px solid #d4d8df;border-radius:3px;width:100%;box-sizing:border-box;background:#fdfdfd}
.kv input:focus,.kv textarea:focus{outline:none;border-color:#2980b9;background:#fff}
.kv textarea{font-family:Menlo,monospace;resize:vertical;min-height:32px}
.section{margin-top:8px;padding-left:12px;border-left:3px solid #cfd6e0}
.section>.title{font-size:11px;font-weight:700;color:#5a6779;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;display:flex;gap:8px;align-items:center}
.list-item{background:#f9fafc;border:1px solid #e6e8ec;border-radius:4px;padding:7px 9px;margin:5px 0;position:relative}
.list-item .rm{position:absolute;top:4px;right:4px;background:transparent;border:0;color:#a02a1a;cursor:pointer;font-size:14px;line-height:1;padding:2px 6px}
.list-item .rm:hover{background:#fbe1de;border-radius:3px}
.add{margin-top:4px;background:transparent;border:1px dashed #b9c0cb;color:#777;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:11px}
.add:hover{border-color:#2980b9;color:#2980b9}
.smiles{font-family:Menlo,monospace}
.unsaved{color:#c0392b;font-weight:600}
"""

PAGE_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style></head><body>"""


def render_index(q: dict) -> str:
    tables = list_tables()
    status = load_status()
    for t in tables:
        s = status.get(t["rel_path"], {})
        t["status"] = s.get("status", "pending")
        t["updated_at"] = s.get("updated_at", "")

    counts = {k: 0 for k in ("pending", "accepted", "edited", "skipped", "rejected")}
    for t in tables:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    total = len(tables)

    journal_filter = (q.get("journal", [""]) or [""])[0]
    status_filter  = (q.get("status",  [""]) or [""])[0]
    search         = (q.get("q",       [""]) or [""])[0]

    filtered = tables
    if journal_filter:
        filtered = [t for t in filtered if t["journal"] == journal_filter]
    if status_filter:
        filtered = [t for t in filtered if t["status"] == status_filter]
    if search:
        filtered = [t for t in filtered if search.lower() in t["filename"].lower()]

    journals = list_journals()
    journal_opts = '<option value="">All</option>' + "".join(
        f'<option value="{j}"{" selected" if j==journal_filter else ""}>{j}</option>'
        for j in journals
    )
    status_opts = '<option value="">All</option>' + "".join(
        f'<option value="{s}"{" selected" if s==status_filter else ""}>{s}</option>'
        for s in ("pending", "accepted", "edited", "skipped", "rejected")
    )

    rows = []
    for t in filtered:
        rows.append(f"""<tr>
            <td>{t["journal"]}</td>
            <td><a class="go" href="annotate?path={urllib.parse.quote(t["rel_path"])}">{t["filename"]}</a></td>
            <td><span class="badge b-{t["status"]}">{t["status"]}</span></td>
            <td>{t.get("updated_at","")}</td>
        </tr>""")

    summary = (
        f'<span>Total <b>{total}</b></span>'
        f'<span>Pending <b>{counts["pending"]}</b></span>'
        f'<span>Accepted <b>{counts["accepted"]}</b></span>'
        f'<span>Edited <b>{counts["edited"]}</b></span>'
        f'<span>Skipped <b>{counts["skipped"]}</b></span>'
        f'<span>Rejected <b>{counts["rejected"]}</b></span>'
    )

    html = PAGE_HEAD.format(title="OMB Review", css=INDEX_CSS) + f"""
<header>
  <h1>RxnOptBench Human Review</h1>
  <div class="summary">{summary}</div>
</header>
<form class="controls" method="get" action="./">
  <label>Journal<select name="journal" onchange="this.form.submit()">{journal_opts}</select></label>
  <label>Status<select name="status" onchange="this.form.submit()">{status_opts}</select></label>
  <label>Filename contains<input type="text" name="q" value="{search}" placeholder="DOI fragment..." /></label>
  <button type="submit" style="padding:6px 14px;background:#2980b9;color:#fff;border:0;border-radius:4px;cursor:pointer">Apply</button>
  <a href="./" style="margin-left:auto;color:#777;text-decoration:none">reset filters</a>
</form>
<table class="tbl">
  <thead><tr><th>Journal</th><th>Table file</th><th>Status</th><th>Updated</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<div class="toolbar">Showing {len(filtered)} of {total} tables. Edits stored at <code>{ANN_ROOT}</code></div>
</body></html>"""
    return html


def render_annotate(rel_path: str) -> str:
    # navigation: find prev/next & next-pending
    all_t = list_tables()
    paths = [t["rel_path"] for t in all_t]
    try:
        idx = paths.index(rel_path)
    except ValueError:
        return "<h1>Table not found</h1>"
    prev_p = paths[idx - 1] if idx > 0 else None
    next_p = paths[idx + 1] if idx < len(paths) - 1 else None

    s = load_status()
    next_pending = None
    for t in all_t[idx + 1:]:
        if s.get(t["rel_path"], {}).get("status", "pending") == "pending":
            next_pending = t["rel_path"]
            break

    nav_html = ""
    if prev_p:
        nav_html += f'<a href="annotate?path={urllib.parse.quote(prev_p)}">← Prev</a>'
    if next_p:
        nav_html += f'<a href="annotate?path={urllib.parse.quote(next_p)}">Next →</a>'
    if next_pending:
        nav_html += f'<a href="annotate?path={urllib.parse.quote(next_pending)}" style="background:#27ae60;color:#fff;border-color:#27ae60">Next pending ⤳</a>'
    nav_html += '<a href="./" style="background:#34495e">List</a>'

    page = PAGE_HEAD.format(title=rel_path, css=ANNOTATE_CSS) + f"""
<header>
  <h1>{rel_path} <span id="dirty" class="unsaved" style="display:none">● unsaved</span></h1>
  <div class="actions">
    <button class="btn-accept" onclick="doAction('accept')">✓ Accept</button>
    <button class="btn-save"   onclick="doAction('edit')">💾 Save edits</button>
    <button class="btn-skip"   onclick="doAction('skip')">⏭ Skip</button>
    <button class="btn-reject" onclick="doAction('reject')">✗ Reject</button>
    <button class="btn-reset"  onclick="doAction('reset')">↺ Reset</button>
  </div>
  <div class="nav">{nav_html}</div>
</header>
<div class="status-line">
  Status: <span id="status-pill" class="badge b-pending">loading...</span>
  <span id="status-meta"></span>
</div>
<main>
  <section id="imgs">
    <h3>Source page <span id="src-name" style="font-weight:400;color:#888;font-family:Menlo,monospace;font-size:11px"></span></h3>
    <div id="src-page"></div>
  </section>
  <section id="editor">
    <div style="margin-bottom:10px;display:flex;gap:6px;align-items:center;font-size:11px;color:#666">
      <button class="bulk-toggle" onclick="bulkToggle(true)">▸ Collapse all</button>
      <button class="bulk-toggle" onclick="bulkToggle(false)">▾ Expand all</button>
      <span style="color:#999">— click any entry header to fold it</span>
    </div>
    <div id="form"></div>
  </section>
</main>

<script>
const REL = {json.dumps(rel_path)};
const NEXT_PENDING = {json.dumps(next_pending)};
let DATA = null;
let DIRTY = false;

function markDirty(){{ DIRTY = true; document.getElementById('dirty').style.display = 'inline'; }}
function clearDirty(){{ DIRTY = false; document.getElementById('dirty').style.display = 'none'; }}

window.addEventListener('beforeunload', e => {{
  if(DIRTY){{ e.preventDefault(); e.returnValue = ''; }}
}});

async function load(){{
  const r = await fetch('api/table?path=' + encodeURIComponent(REL));
  const j = await r.json();
  DATA = j.data;
  // source page (full SI page render)
  const sp = document.getElementById('src-page');
  if(j.src_page){{
    sp.innerHTML = `<img src="srcimg?path=${{encodeURIComponent(REL)}}" style="max-width:100%;display:block;border:1px solid #ddd;border-radius:4px">`;
    document.getElementById('src-name').textContent = '(' + j.src_page + ')';
  }} else {{
    sp.innerHTML = '<div style="color:#999;padding:20px;text-align:center;background:#f9f9fa;border-radius:4px">No source page image found.</div>';
  }}
  // status
  const stat = j.status.status || 'pending';
  const pill = document.getElementById('status-pill');
  pill.className = 'badge b-' + stat;
  pill.textContent = stat;
  const meta = document.getElementById('status-meta');
  meta.textContent = j.status.updated_at ? `(updated ${{j.status.updated_at}}${{j.edited?', edits saved':''}})` : '';
  // form
  render();
}}

// per-entry collapsed state, keyed by entry id, persisted across re-renders
const COLLAPSED = new Set();

// schema templates for "+ Add" buttons when a list / object is empty.
// keys are field names; values are blank templates (deep-cloned each insert).
const TEMPLATES = {{
  reactants: {{content:"", box_id:"", Idt:"", amount:"", conversion_rate:"", diastereomeric_ratio:"", other_info:"", SMILES:""}},
  products:  {{content:"", box_id:"", Idt:"", yield_ratio:"", isolated_yield:"", enantiomeric_excess:"", regioisomeric_ratio:"", enantiomeric_ratio:"", alkene_configuration:"", diastereomeric_ratio:"", other_info:"", SMILES:""}},
  reagents:  {{content:"", ratio:"", amount:"", other_info:""}},
  catalysts: {{content:"", ratio:"", amount:"", other_info:""}},
  solvents:  {{content:"", ratio:"", amount:"", other_info:""}},
  selected_conditions: "",
  stages: {{selected_conditions:[], reagents:[], catalysts:[], solvents:[], reaction_temperature:"", reaction_time:"", atmosphere:"", other_info:""}},
}};
const REACTION_TEMPLATE = {{
  is_multi_reactions:"False", "R-info":"",
  reactants:[], products:[], stages:[]
}};

function deepClone(x){{ return JSON.parse(JSON.stringify(x)); }}

function templateFor(key, existingList){{
  // 1) if list has items, mirror the first one with empty leaves
  if(existingList && existingList.length){{
    const first = existingList[0];
    if(typeof first === 'string') return '';
    if(first && typeof first === 'object') {{
      const t = {{}};
      for(const k of Object.keys(first)){{
        const v = first[k];
        t[k] = Array.isArray(v) ? [] : (v && typeof v === 'object' ? {{}} : '');
      }}
      return t;
    }}
  }}
  // 2) fall back to known schema
  if(key in TEMPLATES) return deepClone(TEMPLATES[key]);
  // 3) unknown — guess empty string
  return '';
}}

function entryPreview(reactions){{
  // tiny summary so collapsed rows still show useful info
  try {{
    const r = reactions[0] || {{}};
    const p = (r.products || [])[0] || {{}};
    const y = p.yield_ratio || p.isolated_yield || '';
    const ee = p.enantiomeric_excess || p.enantiomeric_ratio || '';
    const dr = p.diastereomeric_ratio || '';
    const id = p.Idt || '';
    const bits = [];
    if(id) bits.push(id);
    if(y)  bits.push('yield ' + y);
    if(ee) bits.push('ee ' + ee);
    if(dr) bits.push('dr ' + dr);
    return bits.join(' · ');
  }} catch(e) {{ return ''; }}
}}

function render(){{
  const root = document.getElementById('form');
  root.innerHTML = '';
  // top-level: dict[entry_id -> list[reaction]]
  Object.keys(DATA).sort((a,b)=>{{const x=parseInt(a), y=parseInt(b); return isNaN(x)||isNaN(y) ? a.localeCompare(b) : x-y;}}).forEach(eid => {{
    const reactions = DATA[eid];
    const card = document.createElement('div');
    card.className = 'entry' + (COLLAPSED.has(eid) ? ' collapsed' : '');
    card.dataset.eid = eid;

    const head = document.createElement('div');
    head.className = 'entry-head';
    head.innerHTML = `
      <span class="caret">▾</span>
      <span class="eid">Entry ${{eid}}</span>
      <span style="color:#999;font-size:11px">${{reactions.length}} rxn</span>
      <span class="preview">${{entryPreview(reactions)}}</span>
    `;
    head.onclick = (ev) => {{
      if(ev.target.tagName === 'BUTTON') return;
      if(COLLAPSED.has(eid)) COLLAPSED.delete(eid); else COLLAPSED.add(eid);
      card.classList.toggle('collapsed');
    }};

    const del = document.createElement('button');
    del.className = 'del';
    del.textContent = '🗑 delete entry';
    del.onclick = (ev) => {{
      ev.stopPropagation();
      if(confirm(`Delete entry ${{eid}} and all its reactions?`)){{
        delete DATA[eid]; COLLAPSED.delete(eid); markDirty(); render();
      }}
    }};
    head.appendChild(del);
    card.appendChild(head);

    const body = document.createElement('div');
    body.className = 'entry-body';
    reactions.forEach((rxn, ri) => {{
      const rxnDiv = document.createElement('div');
      rxnDiv.className = 'section';
      rxnDiv.innerHTML = `<div class="title">Reaction #${{ri}} <button class="rm" style="position:static;margin-left:auto" onclick="(${{deleteReaction.toString()}})('${{eid}}', ${{ri}})">×</button></div>`;
      // structure strip (reactants -> products) above this reaction's fields
      const strip = document.createElement('div');
      strip.className = 'mol-strip';
      strip.dataset.eid = eid; strip.dataset.ri = ri;
      strip.innerHTML = molStripHTML(eid, ri);
      rxnDiv.appendChild(strip);
      rxnDiv.appendChild(renderObject(rxn, [eid, ri]));
      body.appendChild(rxnDiv);
    }});
    // + Add reaction inside this entry
    const addRxn = document.createElement('button');
    addRxn.className = 'add';
    addRxn.style.marginTop = '6px';
    addRxn.textContent = '+ Add reaction to this entry';
    addRxn.onclick = () => {{
      DATA[eid].push(deepClone(REACTION_TEMPLATE));
      markDirty(); render();
    }};
    body.appendChild(addRxn);
    card.appendChild(body);

    root.appendChild(card);
  }});

  // global + Add new entry
  const addEntry = document.createElement('button');
  addEntry.className = 'add';
  addEntry.style.cssText = 'margin-top:8px;font-size:13px;padding:8px 16px;width:100%;border-style:dashed;border-color:#2980b9;color:#2980b9';
  addEntry.textContent = '+ Add new entry';
  addEntry.onclick = () => {{
    // pick next available numeric id
    const used = new Set(Object.keys(DATA));
    let nid = 0;
    while(used.has(String(nid))) nid++;
    DATA[String(nid)] = [deepClone(REACTION_TEMPLATE)];
    markDirty(); render();
  }};
  root.appendChild(addEntry);
}}

function bulkToggle(collapse){{
  COLLAPSED.clear();
  if(collapse){{ Object.keys(DATA).forEach(eid => COLLAPSED.add(eid)); }}
  render();
}}

function deleteReaction(eid, ri){{
  if(!confirm(`Delete reaction #${{ri}} of entry ${{eid}}?`)) return;
  DATA[eid].splice(ri, 1);
  if(DATA[eid].length === 0) delete DATA[eid];
  markDirty(); render();
}}

function renderObject(obj, path){{
  const div = document.createElement('div');
  Object.keys(obj).forEach(k => {{
    const v = obj[k];
    const child = renderField(k, v, [...path, k]);
    if(child) div.appendChild(child);
  }});
  return div;
}}

function renderField(key, val, path){{
  if(Array.isArray(val)){{
    return renderList(key, val, path);
  }}
  if(val !== null && typeof val === 'object'){{
    const div = document.createElement('div');
    div.className = 'section';
    div.innerHTML = `<div class="title">${{key}}</div>`;
    div.appendChild(renderObject(val, path));
    return div;
  }}
  // scalar
  const row = document.createElement('div');
  row.className = 'kv';
  const lbl = document.createElement('label');
  lbl.textContent = key;
  row.appendChild(lbl);
  const isLong = (typeof val === 'string') && (val.length > 60 || val.includes('\\n'));
  const isSmiles = key === 'SMILES';
  let inp;
  if(isLong){{
    inp = document.createElement('textarea');
    inp.rows = 2;
  }} else {{
    inp = document.createElement('input');
    inp.type = 'text';
  }}
  if(isSmiles) inp.className = 'smiles';
  inp.value = val == null ? '' : String(val);
  inp.addEventListener('input', () => {{
    setByPath(DATA, path, inp.value);
    markDirty();
  }});
  if(isSmiles){{
    inp.addEventListener('change', () => refreshStripFor(path));
    inp.addEventListener('blur',   () => refreshStripFor(path));
  }}
  row.appendChild(inp);
  return row;
}}

function molStripHTML(eid, ri){{
  const r = (DATA[eid] && DATA[eid][ri]) || {{}};
  const cell = (it, role) => {{
    const smi = (it && it.SMILES || '').trim();
    const idt = (it && it.Idt) || '';
    const lbl = idt ? `<div class="mol-lbl">${{escapeHTML(idt)}}</div>` : '';
    if(!smi){{
      return `<div class="mol-cell no-smi"><div style="color:#a02a1a;font-size:10px;padding:20px 4px">no SMILES</div>${{lbl}}</div>`;
    }}
    const url = `mol?smi=${{encodeURIComponent(smi)}}&w=180&h=130`;
    return `<div class="mol-cell ${{role}}">
      <img src="${{url}}" loading="lazy"
           onerror="this.outerHTML='<div style=\\'color:#8a6300;font-size:10px;padding:20px 4px\\'>bad SMILES</div>';this.parentNode&&this.parentNode.classList.add('bad-smi')">
      ${{lbl}}
    </div>`;
  }};
  const reactants = (r.reactants || []).map(x => cell(x, 'r'));
  const products  = (r.products  || []).map(x => cell(x, 'p'));
  return `
    <div class="mol-row">${{reactants.join('') || '<span style="color:#999;font-size:11px">no reactants</span>'}}</div>
    <div class="mol-arrow">→</div>
    <div class="mol-row">${{products.join('')  || '<span style="color:#999;font-size:11px">no products</span>'}}</div>
  `;
}}

function escapeHTML(s){{
  return String(s).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}})[c]);
}}

function refreshStripFor(path){{
  // path like [eid, ri, "reactants" or "products", index, "SMILES"]
  if(!path || path.length < 2) return;
  const eid = String(path[0]); const ri = path[1];
  const strip = document.querySelector(`.mol-strip[data-eid="${{eid}}"][data-ri="${{ri}}"]`);
  if(strip) strip.innerHTML = molStripHTML(eid, ri);
}}

function renderList(key, arr, path){{
  const div = document.createElement('div');
  div.className = 'section';
  const title = document.createElement('div');
  title.className = 'title';
  title.innerHTML = `${{key}} <span style="font-weight:400;color:#999">(${{arr.length}})</span>`;
  div.appendChild(title);
  arr.forEach((it, i) => {{
    const item = document.createElement('div');
    item.className = 'list-item';
    const rm = document.createElement('button');
    rm.className = 'rm';
    rm.title = 'remove';
    rm.textContent = '×';
    rm.onclick = () => {{
      if(confirm(`Remove item ${{i}} from ${{key}}?`)){{
        arr.splice(i, 1); markDirty(); render();
      }}
    }};
    item.appendChild(rm);
    if(it !== null && typeof it === 'object' && !Array.isArray(it)){{
      item.appendChild(renderObject(it, [...path, i]));
    }} else if(typeof it === 'string'){{
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.value = it;
      inp.addEventListener('input', () => {{ arr[i] = inp.value; markDirty(); }});
      item.appendChild(inp);
    }} else {{
      const span = document.createElement('span');
      span.textContent = String(it);
      item.appendChild(span);
    }}
    div.appendChild(item);
  }});
  // + Add item
  const add = document.createElement('button');
  add.className = 'add';
  add.textContent = `+ Add ${{key.replace(/s$/, '')}}`;
  add.onclick = () => {{
    arr.push(templateFor(key, arr));
    markDirty(); render();
  }};
  div.appendChild(add);
  return div;
}}

function setByPath(root, path, value){{
  let cur = root;
  for(let i=0; i<path.length-1; i++) cur = cur[path[i]];
  cur[path[path.length-1]] = value;
}}

async function doAction(action){{
  if(action === 'edit' && !DIRTY && !confirm('No unsaved edits detected. Save anyway?')) return;
  const body = {{rel_path: REL, action}};
  if(action === 'edit') body.data = DATA;
  if(action === 'reject' && !confirm('Reject this whole table?')) return;
  if(action === 'reset' && !confirm('Reset status and discard saved edits for this table?')) return;
  const r = await fetch('api/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(body),
  }});
  const j = await r.json();
  if(!j.ok){{ alert('Save failed: ' + (j.error||'?')); return; }}
  clearDirty();
  // navigate to next pending if accept/skip/reject
  if(['accept','skip','reject'].includes(action) && NEXT_PENDING){{
    location.href = 'annotate?path=' + encodeURIComponent(NEXT_PENDING);
  }} else {{
    load();  // refresh status pill
  }}
}}

// keyboard: a=accept s=skip r=reject e=save
document.addEventListener('keydown', e => {{
  if(e.target.matches('input, textarea, select')) return;
  if(e.key === 'a') doAction('accept');
  else if(e.key === 's') doAction('skip');
  else if(e.key === 'e') doAction('edit');
}});

load();
</script>
</body></html>"""
    return page


# ---------------- threaded server ----------------
class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    if not DATA_ROOT.exists():
        print(f"ERROR: data root {DATA_ROOT} not found", file=sys.stderr)
        sys.exit(1)
    n_tables = sum(1 for _ in DATA_ROOT.glob("*/*.json"))
    print(f"OMB Review Tool")
    print(f"  Data:    {DATA_ROOT}  ({n_tables} tables across {len(list_journals())} journals)")
    print(f"  Edits:   {ANN_ROOT}")
    print(f"  Listen:  http://{HOST}:{PORT}")
    print(f"  Hotkeys: a=accept  s=skip  r=reject  e=save  (when not focused on input)")
    srv = ThreadingServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
