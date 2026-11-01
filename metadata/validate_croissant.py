#!/usr/bin/env python3
"""Validate metadata and load every question without logging credentials/data.

Local mode uses the generated transport files and candidate JSON/JSONL. Authenticated
mode downloads the exact revision URLs using a standard cached HF login or HF_TOKEN,
checks SHA-256, and gives those files to the unmodified reference mlcroissant loader.
The loader's generic HTTP downloader does not natively consume HF_TOKEN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import jsonschema
import mlcroissant
import requests

HERE = Path(__file__).resolve().parent
REQUIRED_RAI = ("rai:dataLimitations", "rai:dataBiases", "rai:personalSensitiveInformation", "rai:dataUseCases", "rai:dataSocialImpact", "rai:hasSyntheticData", "prov:wasDerivedFrom", "prov:wasGeneratedBy")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, default=HERE / "croissant.draft.json")
    p.add_argument("--stage", type=Path, required=True, help="Downloaded HF artifact root containing data/.")
    p.add_argument("--transport-dir", type=Path, help="Default: --stage/data/croissant.")
    p.add_argument("--mode", choices=["local", "authenticated"], default="local")
    p.add_argument("--report", type=Path)
    p.add_argument("--all-distributions", action="store_true", help="Also remotely hash-check native JSON and JSONL resources.")
    p.add_argument("--skip-json-schema", action="store_true")
    args = p.parse_args()
    metadata = json.loads(args.metadata.read_text())
    if args.mode == "authenticated" and metadata["description"].startswith("DRAFT:"):
        p.error("Candidate is unpublished: remote main URLs belong to an earlier version. Use --mode local; remote validation must follow an independently verified publication handoff.")
    report = {
        "status": "running", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "metadata_sha256": sha256(args.metadata),
        "mlcroissant_version": version("mlcroissant"),
        "jsonschema_version": version("jsonschema"),
        "data_revision": sorted({d["contentUrl"].split("/resolve/")[1].split("/")[0]
                                 for d in metadata["distribution"]}),
        "required_rai_present": {key: key in metadata and metadata[key] not in (None, "", [])
                                 for key in REQUIRED_RAI},
        "files": [], "record_sets": [],
    }
    output = args.report or (args.metadata.parent / f"validation_{args.mode}.json")
    try:
        assert all(report["required_rai_present"].values())
        assert isinstance(metadata["rai:hasSyntheticData"], bool)
        schema = json.loads((args.metadata.parent / "record_schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        resources = {d["@id"]: d for d in metadata["distribution"]}
        mapping = {}
        session = requests.Session()
        if args.mode == "authenticated":
            from huggingface_hub import get_token
            token = get_token()
            if not token:
                raise ValueError("No standard Hugging Face cached login or HF_TOKEN is available")
            # Token remains only in memory/session headers; never serialise session.
            session.headers["Authorization"] = "Bearer " + token
            report["authentication"] = "standard cached HF login or HF_TOKEN; secret not recorded"
            remote_dir = HERE / "private_download_cache"
            remote_dir.mkdir(exist_ok=True)
            resources_to_download = [d for d in metadata["distribution"]
                                     if args.all_distributions or d["@id"].startswith("transport/")]
            for resource in resources_to_download:
                file_id = resource["@id"]
                path = remote_dir / (file_id.replace("/", "__") + "__" + resource["name"])
                # Force a fresh network request; an existing cache cannot masquerade as remote access.
                response = session.get(resource["contentUrl"], timeout=(20, 120))
                if response.status_code != 200:
                    raise RuntimeError(f"Authenticated distribution download failed: {file_id}, HTTP {response.status_code}")
                path.write_bytes(response.content)
                actual = sha256(path)
                assert actual == resource["sha256"], f"Remote checksum differs: {file_id}"
                report["files"].append({"id": file_id, "sha256": actual, "bytes": len(response.content), "http_status": 200})
                if file_id.startswith("transport/"):
                    mapping[file_id] = str(path.resolve())
            session.headers.pop("Authorization", None)
            del token
        else:
            report["authentication"] = "none; local files mapped explicitly"
            for file_id, resource in resources.items():
                if file_id.startswith("transport/"):
                    path = (args.transport_dir or args.stage / "data/croissant") / resource["name"]
                    mapping[file_id] = str(path.resolve())
                elif file_id.startswith("native-json/"):
                    path = args.stage / "data/original_json" / resource["name"]
                else:
                    path = args.stage / "data/jsonl" / resource["name"]
                actual = sha256(path)
                assert actual == resource["sha256"], f"Local checksum differs: {file_id}"
                report["files"].append({"id": file_id, "sha256": actual, "bytes": path.stat().st_size})
        dataset = mlcroissant.Dataset(metadata, mapping=mapping)
        report["static_errors"] = sorted(dataset.metadata.ctx.issues.errors)
        report["static_warnings"] = sorted(dataset.metadata.ctx.issues.warnings)
        assert not report["static_errors"]
        total = 0
        for record_set in metadata["recordSet"]:
            subset = record_set["@id"]
            original = json.loads((args.stage / "data/original_json" / f"{subset}.json").read_text())
            jsonl = [json.loads(line) for line in (args.stage / "data/jsonl" / f"{subset}.jsonl").read_text().splitlines()]
            assert original == jsonl, f"Native JSON/JSONL mismatch: {subset}"
            restored = []
            validator = jsonschema.Draft202012Validator(schema["$defs"][subset])
            for index, loaded in enumerate(dataset.records(subset)):
                row = json.loads(as_text(loaded[f"{subset}/json_record"]))
                assert row == original[index], f"Lossless record mismatch: {subset} row {index}"
                assert as_text(loaded[f"{subset}/id"]) == row["id"]
                assert as_text(loaded[f"{subset}/question_type"]) == row["question_type"]
                assert as_text(loaded[f"{subset}/parent_table_id"]) == row["meta"]["parent_table_id"]
                if not args.skip_json_schema:
                    validator.validate(row)
                restored.append(row)
            assert len(restored) == len(original), f"Record count mismatch: {subset}"
            assert len({r["id"] for r in restored}) == len(restored)
            report["record_sets"].append({"id": subset, "records": len(restored), "lossless_matches": len(restored),
                                          "json_schema_validated": not args.skip_json_schema,
                                          "native_json_equals_jsonl": True})
            total += len(restored)
        assert total == 4773 and len(report["record_sets"]) == 8
        report["records_loaded"] = total
        report["status"] = "pass"
        report["limits"] = (
            "This checks the official reference loader, complete local record equality, declared RAI presence, and nested "
            "structural schema. It does not establish scientific label correctness, source-document licensing, gated-access "
            "acceptance by reviewers, or approval by the NeurIPS online validator. Authenticated mode verifies authorized "
            "network access; anonymous downloads remain gated. This draft has not undergone remote loading or online checker validation."
        )
    except Exception as error:
        report["status"] = "fail"
        # Controlled error text only; no request/session/header objects.
        report["error_type"] = type(error).__name__
        report["error"] = str(error)[:2000]
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"report": str(output), "status": report["status"], "records_loaded": report.get("records_loaded", 0)}))


if __name__ == "__main__":
    main()
