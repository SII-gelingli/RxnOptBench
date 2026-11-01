#!/usr/bin/env python3
"""
Print a fully rendered prompt for a given task/setting/sample.
"""

import argparse
import json
from pathlib import Path

import inference_open_source as inf


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "original_json"
DATASET_FILENAMES = {
    ("mc_single", "classic"): "multiple_choice_single_varying.json",
    ("mc_single", "reference"): "multiple_choice_single_varying_reference.json",
    ("mc_single", "reference_control"): "multiple_choice_single_varying_reference_control.json",
    ("mc_all", "classic"): "multiple_choice_all_varying.json",
    ("mc_all", "reference"): "multiple_choice_all_varying_reference.json",
    ("mc_all", "reference_control"): "multiple_choice_all_varying_reference_control.json",
    ("fib_all", "classic"): "fill_in_blank_all_blank.json",
    ("fib_all", "reference"): "fill_in_blank_all_blank_reference.json",
}


def default_dataset_path(task_type, setting):
    filename = DATASET_FILENAMES.get((task_type, setting))
    if filename is None:
        raise ValueError(f"Unsupported task/setting combination: {task_type} + {setting}")
    return DATASET_DIR / filename


def main():
    parser = argparse.ArgumentParser(description="Inspect rendered prompts for V6 tasks.")
    parser.add_argument("--task-type", choices=["mc_single", "mc_all", "fib_all"], required=True)
    parser.add_argument("--setting", choices=["classic", "reference", "reference_control"], default="classic")
    parser.add_argument("--reasoning-mode", choices=["none", "required"], default="none")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    if args.dataset is not None:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = default_dataset_path(args.task_type, args.setting)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.sample_id is not None:
        sample = next((item for item in dataset if item["id"] == args.sample_id), None)
        if sample is None:
            raise ValueError(f"Sample id not found: {args.sample_id}")
    else:
        sample = dataset[args.index]

    if args.task_type == "mc_single":
        prompt = inf.build_mc_single_prompt(
            sample, args.reasoning_mode, args.setting, str(dataset_path)
        )
    elif args.task_type == "mc_all":
        prompt = inf.build_mc_all_prompt(
            sample, args.reasoning_mode, args.setting, str(dataset_path)
        )
    else:
        prompt = inf.build_fib_all_prompt(
            sample, args.reasoning_mode, args.setting, str(dataset_path)
        )

    print(f"task_type={args.task_type}")
    print(f"setting={args.setting}")
    print(f"reasoning_mode={args.reasoning_mode}")
    print(f"dataset={dataset_path}")
    print(f"sample_id={sample['id']}")
    print("")
    print(prompt)


if __name__ == "__main__":
    main()
