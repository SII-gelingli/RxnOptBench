#!/usr/bin/env python3
"""Compute the exact declared uniform-random baseline for a native gold version.

FIB Score|Hit is the finite-batch conditional expectation given at least one
Hit, not a macro mean of legal-row scores. No model/API calls are made.
Requires NumPy as listed in requirements-scoring.txt.
"""
import argparse
import json
from pathlib import Path
from recompute_alignment import uniform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    args = p.parse_args()
    result = {}
    for task, name in [('mc_single', 'multiple_choice_single_varying'), ('mc_all', 'multiple_choice_all_varying'), ('fib_all', 'fill_in_blank_all_blank')]:
        values = json.loads((args.data_dir / (name + '.json')).read_text(encoding='utf-8'))
        result[task] = uniform(values, fib=task.startswith('fib'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
