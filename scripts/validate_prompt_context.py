#!/usr/bin/env python3
"""Verify that every stored constant condition is rendered in model prompts."""

import argparse
import json
from pathlib import Path


DATASETS = (
    ("mc_single", "classic", "multiple_choice_single_varying.json"),
    ("mc_single", "reference", "multiple_choice_single_varying_reference.json"),
    (
        "mc_single",
        "reference_control",
        "multiple_choice_single_varying_reference_control.json",
    ),
    ("mc_all", "classic", "multiple_choice_all_varying.json"),
    ("mc_all", "reference", "multiple_choice_all_varying_reference.json"),
    (
        "mc_all",
        "reference_control",
        "multiple_choice_all_varying_reference_control.json",
    ),
    ("fib_all", "classic", "fill_in_blank_all_blank.json"),
    ("fib_all", "reference", "fill_in_blank_all_blank_reference.json"),
)

TEMPLATE_FILES = {
    ("mc_single", "classic"): "prompt_single_varying.txt",
    ("mc_single", "reference"): "prompt_single_varying_reference.txt",
    ("mc_single", "reference_control"): "prompt_single_varying.txt",
    ("mc_all", "classic"): "prompt_all_varying.txt",
    ("mc_all", "reference"): "prompt_all_varying_reference.txt",
    ("mc_all", "reference_control"): "prompt_all_varying.txt",
    ("fib_all", "classic"): "prompt_all_blank.txt",
    ("fib_all", "reference"): "prompt_all_blank_reference.txt",
}


def format_species_list(species):
    parts = []
    for item in species:
        name = item.get("content", "").strip()
        smiles = item.get("SMILES", "").strip()
        if name and smiles:
            parts.append(f"{name} (SMILES: {smiles})")
        elif smiles:
            parts.append(f"SMILES: {smiles}")
        elif name:
            parts.append(name)
    return "\n".join(f"  - {part}" for part in parts) if parts else "  (none)"


def format_condition_dict(conditions):
    lines = []
    for key, items in conditions.items():
        parts = []
        for item in items or []:
            name = item.get("content", "").strip()
            smiles = item.get("SMILES", "").strip()
            if name and smiles:
                parts.append(f"{name} (SMILES: {smiles})")
            elif name:
                parts.append(name)
            elif smiles:
                parts.append(f"SMILES: {smiles}")
        value = ", ".join(parts) if parts else "(none)"
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines) if lines else "  (none)"


def format_mc_options(options):
    blocks = []
    for index, option in enumerate(options):
        blocks.append(
            "\n".join(
                [f"**Option {index}:**", format_condition_dict(option)]
            )
        )
    return "\n\n".join(blocks)


def format_fib_candidates(candidate_values):
    blocks = []
    for key, candidates in candidate_values.items():
        lines = [f"**{key}:**"]
        for candidate in candidates:
            values = candidate.get("value", [])
            parts = []
            for value in values:
                name = value.get("content", "").strip()
                smiles = value.get("SMILES", "").strip()
                if name and smiles:
                    parts.append(f"{name} (SMILES: {smiles})")
                elif name:
                    parts.append(name)
                elif smiles:
                    parts.append(f"SMILES: {smiles}")
            rendered = ", ".join(parts) if parts else "(none)"
            lines.append(f"  {candidate['index']}: {rendered}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_references(references):
    blocks = []
    for index, reference in enumerate(references, start=1):
        outcome = reference.get("outcome", {})
        score_parts = []
        for key in ("yield", "ee", "dr", "er", "rr"):
            if outcome.get(key) is not None:
                suffix = "%" if key == "yield" else ""
                score_parts.append(f"{key}={outcome[key]}{suffix}")
        scores = f" ({', '.join(score_parts)})" if score_parts else ""
        blocks.append(
            "\n".join(
                [
                    f"*Reference {index}{scores}:*",
                    format_condition_dict(reference.get("conditions", {})),
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "  (none)"


def render_prompt(task_type, setting, sample, prompt_dir):
    inp = sample["input"]
    template_path = prompt_dir / TEMPLATE_FILES[(task_type, setting)]
    template = template_path.read_text(encoding="utf-8")
    values = {
        "reactants": format_species_list(inp.get("reactants", [])),
        "products": format_species_list(inp.get("products", [])),
        "constant_conditions": format_condition_dict(
            inp.get("constant_conditions", {})
        ),
        "fixed_conditions": format_condition_dict(
            inp.get("fixed_conditions", {})
        ),
        "references": format_references(inp.get("reference_conditions", [])),
        "options": format_mc_options(sample.get("options", [])),
        "candidate_values": format_fib_candidates(
            inp.get("candidate_values", {})
        ),
        "target_key": sample.get("target_key", ""),
        "query_keys_json": ",\n".join(
            f'  "{key}": <candidate_index>'
            for key in sample.get("query_keys", [])
        ),
    }
    return template.format(**values)


def condition_tokens(conditions):
    for key, values in conditions.items():
        yield key
        for value in values:
            content = value.get("content", "").strip()
            smiles = value.get("SMILES", "").strip()
            if content:
                yield content
            if smiles:
                yield smiles


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "original_json",
        help="Directory containing the eight submitted JSON datasets.",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=root / "prompts",
        help="Directory containing the six submitted prompt templates.",
    )
    args = parser.parse_args()

    checked_samples = 0
    checked_tokens = 0
    failures = []
    for task_type, setting, filename in DATASETS:
        dataset_path = args.data_dir / filename
        samples = json.loads(dataset_path.read_text(encoding="utf-8"))
        for sample in samples:
            prompt = render_prompt(task_type, setting, sample, args.prompt_dir)
            tokens = list(
                condition_tokens(
                    sample["input"].get("constant_conditions", {})
                )
            )
            missing = [token for token in tokens if token not in prompt]
            if missing:
                failures.append((sample["id"], missing))
            checked_samples += 1
            checked_tokens += len(tokens)

    if checked_samples != 4773:
        raise SystemExit(f"expected 4773 prompts, audited {checked_samples}")
    if failures:
        for sample_id, missing in failures[:20]:
            print(f"FAIL {sample_id}: {missing}")
        raise SystemExit(
            f"{len(failures)} prompts omitted one or more constant-condition fields"
        )

    print(
        f"OK: {checked_samples} prompts include all "
        f"{checked_tokens} constant-condition keys and values"
    )


if __name__ == "__main__":
    main()
