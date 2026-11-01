#!/usr/bin/env python3
"""Restore saved FIB predictions to the submitted candidate-index space.

Release 0.2.3 of the reviewer package stored FIB predictions after a
deterministic candidate-order audit permutation.  The authoritative dataset
files in this repository are the source-order files from submitted public
revision 7b3c1b4c2e3839a40ba27d07309c0dd9a1f22bdb.  This packaging utility
inverts that permutation for parsed predictions and the final structured
answer embedded in raw response strings.

The selected chemical value is invariant: a shuffled index is replaced by the
index of the same candidate in the submitted source order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = ROOT / "results" / "predictions"
DEFAULT_DATA_ROOT = ROOT / "data" / "original_json"
DEFAULT_AUDIT_PATH = ROOT / "metadata" / "result_index_restoration.json"
SEED = 20260504
TASKS = {
    "fib_all": "fill_in_blank_all_blank.json",
    "fib_all_reference": "fill_in_blank_all_blank_reference.json",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submitted_indices_in_shuffled_order(
    sample_id: str, key: str, size: int
) -> list[int]:
    payload = f"{SEED}::{sample_id}::{key}".encode()
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    order = list(range(size))
    random.Random(value).shuffle(order)
    return order


def restore_prediction(prediction: Any, sample: dict[str, Any]) -> Any:
    if not isinstance(prediction, dict):
        return prediction
    restored = dict(prediction)
    for key in sample["query_keys"]:
        index = restored.get(key)
        candidates = sample["input"]["candidate_values"][key]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(candidates)
        ):
            continue
        order = submitted_indices_in_shuffled_order(
            sample["id"], key, len(candidates)
        )
        restored[key] = order[index]
    return restored


def replace_raw_indices(
    raw: Any, shuffled_prediction: Any, submitted_prediction: Any
) -> Any:
    if (
        not isinstance(raw, str)
        or not isinstance(shuffled_prediction, dict)
        or not isinstance(submitted_prediction, dict)
    ):
        return raw

    output = raw
    for key, old_value in shuffled_prediction.items():
        new_value = submitted_prediction.get(key)
        if (
            not isinstance(old_value, int)
            or isinstance(old_value, bool)
            or not isinstance(new_value, int)
            or old_value == new_value
        ):
            continue
        quoted_key = rf"([\"']{re.escape(key)}[\"']\s*:\s*)"
        output, count = re.subn(
            quoted_key + rf"{old_value}(?!\d)",
            rf"\g<1>{new_value}",
            output,
        )
        if count:
            continue
        bare_key = rf"(\b{re.escape(key)}\b\s*[:=]\s*)"
        output = re.sub(
            bare_key + rf"{old_value}(?!\d)",
            rf"\g<1>{new_value}",
            output,
        )
    return output


def model_directories(results_root: Path) -> list[Path]:
    return sorted(
        model
        for tier in results_root.iterdir()
        if tier.is_dir()
        for model in tier.iterdir()
        if model.is_dir()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write restored prediction/raw files and the audit manifest.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    data_root = args.data_root.resolve()
    audit_path = args.audit_path.resolve()
    if args.apply and audit_path.exists():
        raise SystemExit(
            f"refusing to apply twice; audit manifest already exists: {audit_path}"
        )

    datasets = {
        task: {
            sample["id"]: sample
            for sample in load_json(data_root / filename)
        }
        for task, filename in TASKS.items()
    }
    audit: dict[str, Any] = {
        "operation": "candidate_shuffle_v3_to_submitted_source_order",
        "seed": SEED,
        "submitted_revision": "7b3c1b4c2e3839a40ba27d07309c0dd9a1f22bdb",
        "semantic_invariant": (
            "Each parsed index continues to select the same chemical candidate."
        ),
        "models": {},
    }

    for model_dir in model_directories(results_root):
        model_key = str(model_dir.relative_to(results_root))
        model_audit: dict[str, Any] = {}
        for task, samples in datasets.items():
            prediction_path = model_dir / f"{task}.json"
            raw_path = model_dir / f"{task}_raw.json"
            predictions = load_json(prediction_path)
            raw_responses = load_json(raw_path)
            restored_predictions: dict[str, Any] = {}
            restored_raw = dict(raw_responses)
            changed = 0

            for sample_id, prediction in predictions.items():
                if sample_id not in samples:
                    raise ValueError(
                        f"{model_key}/{task}: unknown sample ID {sample_id}"
                    )
                restored = restore_prediction(prediction, samples[sample_id])
                restored_predictions[sample_id] = restored
                changed += int(restored != prediction)
                if sample_id in restored_raw:
                    restored_raw[sample_id] = replace_raw_indices(
                        restored_raw[sample_id], prediction, restored
                    )

            task_audit = {
                "records": len(predictions),
                "predictions_with_changed_indices": changed,
                "prediction_sha256_before": sha256(prediction_path),
                "raw_sha256_before": sha256(raw_path),
            }
            if args.apply:
                write_json(prediction_path, restored_predictions)
                write_json(raw_path, restored_raw)
                task_audit.update(
                    {
                        "prediction_sha256_after": sha256(prediction_path),
                        "raw_sha256_after": sha256(raw_path),
                    }
                )
            model_audit[task] = task_audit
        audit["models"][model_key] = model_audit

    if args.apply:
        write_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not args.apply:
        print("\nDry run only. Re-run with --apply to write restored files.")


if __name__ == "__main__":
    main()
