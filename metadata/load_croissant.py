#!/usr/bin/env python3
"""Load complete nested RxnOptBench records through unmodified mlcroissant.

Unpublished drafts require --transport-dir for explicit local mapping.
For a separately verified published revision: accept the research agreement on HF,
then log in with the official HF CLI. A standard cached login or HF_TOKEN is used
by huggingface_hub; no token is embedded in metadata or written by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlcroissant
from huggingface_hub import hf_hub_download


def load_dataset(metadata_path: str | Path, transport_dir: str | Path | None = None):
    """Return standard Dataset with hash-checked local or authorized mapping."""
    metadata = json.loads(Path(metadata_path).read_text())
    if transport_dir is None and metadata["description"].startswith("DRAFT:"):
        raise ValueError("This candidate is unpublished. Pass transport_dir for local loading; remote main URLs describe an earlier release.")
    mapping = {}
    for resource in metadata["distribution"]:
        if not resource["@id"].startswith("transport/"):
            continue
        if transport_dir is not None:
            local_path = Path(transport_dir) / resource["name"]
            if hashlib.sha256(local_path.read_bytes()).hexdigest() != resource["sha256"]:
                raise ValueError(f"Local distribution SHA-256 mismatch: {resource['@id']}")
            mapping[resource["@id"]] = str(local_path.resolve())
            continue
        url = urlparse(resource["contentUrl"])
        parts = url.path.split("/")
        if url.scheme != "https" or url.netloc != "huggingface.co" or parts[1] != "datasets" or parts[4] != "resolve":
            raise ValueError("Expected pinned Hugging Face dataset distribution URL")
        repo_id = "/".join(parts[2:4])
        revision = unquote(parts[5])
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Formal remote loading requires an immutable 40-character data revision")
        filename = unquote("/".join(parts[6:]))
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", revision=revision, filename=filename)
        actual = hashlib.sha256(Path(local_path).read_bytes()).hexdigest()
        if actual != resource["sha256"]:
            raise ValueError(f"Distribution SHA-256 mismatch: {resource['@id']}")
        mapping[resource["@id"]] = local_path
    return mlcroissant.Dataset(metadata, mapping=mapping)


def records(dataset, subset: str):
    """Yield original nested dicts; bare ids are only unique within each subset."""
    for loaded in dataset.records(subset):
        yield json.loads(loaded[f"{subset}/json_record"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--subset", default="multiple_choice_all_varying")
    parser.add_argument("--transport-dir", type=Path, help="Optional explicit local TSV.gz directory; otherwise use authorized immutable remote files.")
    args = parser.parse_args()
    dataset = load_dataset(args.metadata, args.transport_dir)
    count = sum(1 for _ in records(dataset, args.subset))
    # Do not print gated question content or authentication values.
    print(json.dumps({"subset": args.subset, "records_loaded": count}))


if __name__ == "__main__":
    main()
