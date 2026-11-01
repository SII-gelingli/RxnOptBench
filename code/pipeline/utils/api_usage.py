"""Shared helpers for calling the OpenAI-compatible proxy/API across stages.

Two things:
  1. Authorization header injection (Bearer token from --api-key or env vars).
  2. JSONL usage logging so we can compute token/cost per DOI afterwards.

Usage in a stage:

    from pipeline.utils.api_usage import (
        resolve_api_key, build_headers, log_usage, DEFAULT_USAGE_LOG,
    )

    headers = build_headers(resolve_api_key(args.api_key))
    t0 = time.time()
    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    log_usage(
        args.usage_log or DEFAULT_USAGE_LOG,
        stage="recognize", doi=doi, page=page, model=model,
        usage=data.get("usage"), elapsed=time.time() - t0,
    )

The log file is an append-only JSONL (one JSON object per HTTP call). It is
safe to write from multiple processes / threads concurrently because each
record is one line and we open/append/close per call.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional


# Default log path (inside the RxnOptBench workspace).
# Resolved relative to this file so it works regardless of CWD.
_THIS = Path(__file__).resolve()
# pipeline/utils/api_usage.py -> RxnOptBench/
_WORKSPACE = _THIS.parent.parent.parent
DEFAULT_USAGE_LOG = str(_WORKSPACE / "logs" / "api_usage.jsonl")

# Environment variables consulted in order for the API key.
_KEY_ENV_VARS = (
    "PROXY_API_KEY",
    "OPENAI_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
)

_write_lock = threading.Lock()


def resolve_api_key(arg_key: Optional[str] = None) -> str:
    """Return the API key to use, checking args first then known env vars.

    Returns an empty string if nothing is set (local vLLM deployments
    don't need a real key). Callers may use ``"EMPTY"`` if they prefer.
    """
    if arg_key:
        return arg_key
    for name in _KEY_ENV_VARS:
        val = os.environ.get(name)
        if val:
            return val
    return ""


def build_headers(api_key: Optional[str]) -> dict:
    """Build default headers including Authorization when a key is provided."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def log_usage(
    log_path: str,
    *,
    stage: str,
    doi: Optional[str] = None,
    page: Optional[str] = None,
    model: Optional[str] = None,
    usage: Optional[dict] = None,
    elapsed: Optional[float] = None,
    message: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append a single usage record to ``log_path`` (JSONL).

    ``usage`` should be the ``resp["usage"]`` block from an OpenAI-compatible
    response. If ``message`` is provided (resp["choices"][0]["message"]),
    we record char counts for ``reasoning_content`` vs ``content`` so the
    cost reporter can estimate the reasoning-token share.
    """
    usage = usage or {}
    reasoning_chars = None
    content_chars = None
    if message is not None:
        rc = message.get("reasoning_content") or ""
        c = message.get("content") or ""
        reasoning_chars = len(rc)
        content_chars = len(c)
    rec: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "doi": doi,
        "page": page,
        "model": model,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
    }
    if extra:
        rec.update(extra)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def extract_doi_from_path(p: str | Path) -> Optional[str]:
    """Best-effort DOI recovery from a file / relative path.

    The pipeline uses filenames like ``10.1021__jacs.4c13321_si.json`` or
    directory names like ``10.1021__jacs.4c13321_si/p18.png``; we strip the
    known suffixes and convert ``__`` back to ``/``.
    """
    if p is None:
        return None
    name = Path(str(p)).name
    if not name:
        return None
    # Strip file extension
    stem = Path(name).stem
    # Remove common suffixes
    for suf in ("_si_3", "_si_2", "_si"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    # Parent dir may be the DOI (e.g. pXX.png inside <doi>_si/)
    if stem.startswith("p") and stem[1:].isdigit():
        parent = Path(str(p)).parent.name
        stem = parent
        for suf in ("_si_3", "_si_2", "_si"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
    return stem.replace("__", "/") if stem else None
