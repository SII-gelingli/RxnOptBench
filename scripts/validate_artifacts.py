#!/usr/bin/env python3
"""Validate downloaded gated artifacts and rescore archived predictions, offline.

The code repository contains no real benchmark records. --artifact-root points
at an authorized download. Score differences are reported, never hidden by a
model-specific gold-label exception. This script makes no model/API requests.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'mc_single': ('multiple_choice_single_varying', 1039, 'mc_single', 'classic'),
    'mc_all': ('multiple_choice_all_varying', 552, 'mc_all', 'classic'),
    'fib_all': ('fill_in_blank_all_blank', 552, 'fib_all', 'classic'),
    'mc_single_reference': ('multiple_choice_single_varying_reference', 850, 'mc_single', 'reference'),
    'mc_all_reference': ('multiple_choice_all_varying_reference', 310, 'mc_all', 'reference'),
    'fib_all_reference': ('fill_in_blank_all_blank_reference', 310, 'fib_all', 'reference'),
    'mc_single_reference_control': ('multiple_choice_single_varying_reference_control', 850, 'mc_single', 'reference_control'),
    'mc_all_reference_control': ('multiple_choice_all_varying_reference_control', 310, 'mc_all', 'reference_control'),
}
MODELS = {'ChemLLM', 'ChemDFM', 'ChemVLM', 'Intern-S1-Pro', 'Qwen3.5-397B-A17B', 'Kimi-K2.6', 'GLM-5.1', 'MiniMax-M2.7', 'DeepSeek-V4-Pro', 'GPT-5.5', 'Gemini-3.1-Pro', 'Claude-Opus-4.7'}
# The archived run manifest identifies these terminal API statuses as not
# submitted. Retain that declared historical aggregation, and expose all counts.
RECORDED_EXCLUSIONS = {'DeepSeek-V4-Pro': {'mc_all': {'api_timeout_skip'}, 'fib_all': {'api_timeout_skip'}}}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--snapshot', choices=['final', 'historical'], default='final')
    parser.add_argument('--output', type=Path, required=True, help='Write an audit JSON outside immutable downloaded inputs.')
    parser.add_argument('--include-all-submitted', action='store_true', help='Also score terminal API skips as non-Hits/incorrect; never changes stored files.')
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    data_root = artifact / ('historical/submitted_v0.2.4/data' if args.snapshot == 'historical' else 'data')
    errors, datasets = [], {}
    report = {'snapshot': args.snapshot, 'aggregation': 'all archived IDs' if args.include_all_submitted else 'recorded submitted predictions; terminal api_timeout_skip exclusions explicit', 'code_sha256': {}, 'dataset_sha256': {}, 'counts': {}, 'results': [], 'historical_display_differences': []}
    for path in CODE_ROOT.rglob('*.py'):
        if '__pycache__' not in path.parts:
            try:
                ast.parse(path.read_text(encoding='utf-8'))
            except SyntaxError:
                errors.append('Python syntax error: ' + str(path.relative_to(CODE_ROOT)))
    mc = module('release_mc', CODE_ROOT / 'evaluation/evaluate_multiple_choice.py')
    fib = module('release_fib', CODE_ROOT / 'evaluation/evaluate_fill_in_blank.py')
    renderer = module('release_prompts', CODE_ROOT / 'scripts/validate_prompt_context.py')
    for path in [CODE_ROOT / 'evaluation/evaluate_multiple_choice.py', CODE_ROOT / 'evaluation/evaluate_fill_in_blank.py', *sorted((CODE_ROOT / 'prompts').glob('*.txt'))]:
        report['code_sha256'][str(path.relative_to(CODE_ROOT))] = sha(path)
    total, constants = 0, 0
    for task, (stem, expected, task_type, setting) in TASKS.items():
        path = data_root / 'original_json' / (stem + '.json')
        values = read(path)
        datasets[task] = values
        report['dataset_sha256'][str(path.relative_to(artifact))] = sha(path)
        if len(values) != expected or len({x['id'] for x in values}) != len(values):
            errors.append(task + ': unexpected count or duplicate within-subset ID')
        jsonl = data_root / 'jsonl' / (stem + '.jsonl')
        if not jsonl.is_file():
            errors.append(task + ': required JSONL file is missing')
        elif values != [json.loads(x) for x in jsonl.read_text(encoding='utf-8').splitlines() if x.strip()]:
            errors.append(task + ': JSON/JSONL content or order mismatch')
        for value in values:
            prompt = renderer.render_prompt(task_type, setting, value, CODE_ROOT / 'prompts')
            tokens = list(renderer.condition_tokens(value['input'].get('constant_conditions', {})))
            constants += len(tokens)
            if any(token not in prompt for token in tokens):
                errors.append(task + ': missing constant-condition prompt field')
        total += len(values)
    report['counts'].update(questions=total, prompt_templates=len(list((CODE_ROOT / 'prompts').glob('*.txt'))), rendered_prompts=total, constant_condition_tokens=constants)
    reviewed = artifact / 'reviewed_data'
    manifest = read(reviewed / 'manifest.json')
    actual = {str(p.relative_to(reviewed)): sha(p) for p in sorted((reviewed / 'tables').rglob('*.json'))}
    declared_count = manifest.get('file_count')
    if (not isinstance(declared_count, int) or isinstance(declared_count, bool)
            or declared_count <= 0 or len(actual) != declared_count
            or len(manifest.get('files', {})) != declared_count
            or manifest.get('files') != actual):
        errors.append('reviewed table count/hash mismatch')
    report['counts']['reviewed_table_files'] = len(actual)
    directories = sorted(p for tier in (artifact / 'results/predictions').iterdir() if tier.is_dir() for p in tier.iterdir() if p.is_dir())
    if {p.name for p in directories} != MODELS:
        errors.append('unexpected saved model set')
    expected_names = {task + suffix + '.json' for task in TASKS for suffix in ['', '_raw', '_parse_methods']}
    output_files = 0
    for directory in directories:
        if {p.name for p in directory.iterdir() if p.is_file()} != expected_names:
            errors.append(directory.name + ': saved file set mismatch')
        for task in TASKS:
            expected_ids = {x['id'] for x in datasets[task]}
            parsed = read(directory / (task + '.json'))
            raw = read(directory / (task + '_raw.json'))
            methods = read(directory / (task + '_parse_methods.json'))
            for label, payload in [('parsed', parsed), ('derived_response_text', raw), ('parse_methods', methods)]:
                if not isinstance(payload, dict) or set(payload) != expected_ids:
                    errors.append(directory.name + '/' + task + '/' + label + ': ID coverage mismatch')
            timeout_ids = sorted(k for k in parsed if methods.get(k) == "api_timeout_skip")
            # A stored numeric zero can be a transport placeholder, not a model
            # answer selecting option 0. All terminal skips become no-answer.
            cleaned = {k: (None if k in timeout_ids else v) for k, v in parsed.items()}
            excluded = set() if args.include_all_submitted else RECORDED_EXCLUSIONS.get(directory.name, {}).get(task, set())
            excluded_ids = sorted(k for k in parsed if methods.get(k) in excluded)
            predictions = {k: v for k, v in cleaned.items() if k not in excluded_ids}
            # Passing explicit sample filtering makes the denominator observable;
            # excluded API calls are not submitted predictions in archive mode.
            used_data = [x for x in datasets[task] if x['id'] in predictions]
            evaluator = fib if task.startswith('fib') else mc
            metrics, _ = evaluator.evaluate(used_data, predictions)
            report['results'].append({'model': directory.name, 'task': task, 'archived_ids': len(parsed), 'excluded_terminal_status_count': len(excluded_ids), 'excluded_terminal_ids': excluded_ids, 'api_timeout_no_answer_count': len(timeout_ids), 'api_timeout_no_answer_ids': timeout_ids, 'metrics': metrics})
            output_files += 3
    report['counts'].update(models=len(directories), saved_output_files=output_files)
    repeat_root = artifact / 'results/deepseek_repeatability/outputs'
    repeat_files, repeat_predictions = 0, 0
    for run in ['run_01', 'run_02', 'run_03']:
        run_dir = repeat_root / run
        if not run_dir.is_dir():
            errors.append(run + ': repeat-run directory missing')
            continue
        if {p.name for p in run_dir.iterdir() if p.is_file()} != expected_names:
            errors.append(run + ': repeat artifact file set mismatch')
        for task in TASKS:
            expected_ids = {x['id'] for x in datasets[task]}
            for suffix in ['', '_raw', '_parse_methods']:
                path = run_dir / (task + suffix + '.json')
                if not path.is_file():
                    errors.append(run + '/' + task + suffix + ': missing required repeat artifact')
                    continue
                payload = read(path)
                if not isinstance(payload, dict) or set(payload) != expected_ids:
                    errors.append(run + '/' + task + suffix + ': repeat ID coverage mismatch')
                repeat_files += 1
                if suffix == '':
                    repeat_predictions += len(payload)
    report['counts'].update(repeat_runs=3, saved_repeat_files=repeat_files, archived_repeat_predictions=repeat_predictions)
    historical_metrics = artifact / 'historical/submitted_v0.2.4/results/table2_metrics.tsv'
    if args.snapshot == 'historical' and historical_metrics.exists():
        shown = {r['Model']: r for r in csv.DictReader(historical_metrics.open(encoding='utf-8'), delimiter='\t')}
        for model in sorted(MODELS):
            scores = {r['task']: r['metrics'] for r in report['results'] if r['model'] == model}
            m1, ma, f = scores['mc_single'], scores['mc_all'], scores['fib_all']
            fields = {'MC_single_Acc_pct': f"{m1['exact_match_accuracy']*100:.1f}", 'MC_single_Score': f"{m1['avg_relative_score']:.3f}", 'MC_all_Acc_pct': f"{ma['exact_match_accuracy']*100:.1f}", 'MC_all_Score': f"{ma['avg_relative_score']:.3f}", 'FIB_Hit_pct': f"{f['hit_rate']*100:.1f}", 'FIB_Strict_Acc_pct': f"{f['best_hit_count']/f['num_samples']*100:.1f}", 'FIB_Score': f"{f['avg_relative_score']:.3f}"}
            for field, computed in fields.items():
                if shown[model][field] != computed:
                    report['historical_display_differences'].append({'model': model, 'field': field, 'archived_display': shown[model][field], 'recomputed': computed, 'explanation': 'Stored display and prediction-level recomputation differ. No unsupported parser-maintenance cause is asserted.'})
    report['errors'] = sorted(set(errors))
    report['status'] = 'PASS' if not errors else 'FAIL'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': report['status'], 'counts': report['counts'], 'errors': report['errors'], 'historical_display_differences': report['historical_display_differences']}, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
