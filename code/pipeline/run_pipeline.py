#!/usr/bin/env python3
"""
Unified pipeline orchestrator for screening table structured parsing.

Coordinates the complete extraction and benchmark-build workflow:

  Stage 0: render        → stage0_render.py          (PDF→600DPI PNG)
  Stage 1: screen_text   → stage1_screen_text.py     (keyword pre-screening)
  Stage 2: screen_vlm    → stage2_screen_vlm.py      (Kimi VLM screening + cross-page detection)
  Stage 3: link          → stage3_link.py            (symlink creation)
  Stage 4: recognize     → stage4_recognize.py       (table structure recognition)
  Stage 4b: complete     → stage4b_complete_tables.py (cross-page completion)
  Stage 5: postprocess   → stage5_postprocess.py     (Kimi footnote separation)
  Stage 6: detect_mols   → stage6_detect_mols.py     (molecule detection)
  External prerequisite  → precomputed OCSR JSON is required before align_labels
  Stage 8: align_labels  → stage8_align_labels.py    (VLM label-SMILES alignment)
  Stage 8b: table_align  → stage8b_table_align.py    (table-level slot alignment)
  Cross-validation       → stage_cross_validate_labels.py
  Stage 9: assemble      → stage9_assemble.py        (final assembly)
  Stage 10: expand_rows  → stage10_expand_rows.py    (row-level reactions)
  Post-processing        → fill reagent SMILES, repair SMILES, build benchmark
  Schema conversion      → stage11_convert_schema.py (review-schema VLM pass)

Usage:
    python pipeline/run_pipeline.py --stages all
    python pipeline/run_pipeline.py --start-from detect_mols --stop-after assemble
    python pipeline/run_pipeline.py --stages align_labels,assemble \
        --smiles-dir /path/to/external/results --journal JACS --test 5
    python pipeline/run_pipeline.py --resume --source si --detector decimer
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BASE_DIR / "pipeline"
PROGRESS_FILE = BASE_DIR / "pipeline_progress.json"

# ── Stage definitions ────────────────────────────────────────────────────────
STAGE_DEFS = [
    {
        "name": "render",
        "script": "pipeline/stage0_render.py",
        "default_args": [],
        "gpu": False,
        "description": "Render PDFs to 600 DPI PNGs",
        "output_dir": "stage0_rendered_pages",
    },
    {
        "name": "screen_text",
        "script": "pipeline/stage1_screen_text.py",
        "default_args": [],
        "gpu": False,
        "description": "Keyword-based pre-screening",
        "output_dir": "stage1_screened_text",
    },
    {
        "name": "screen_vlm",
        "script": "pipeline/stage2_screen_vlm.py",
        "default_args": [],
        "gpu": True,
        "description": "Kimi VLM screening confirmation + cross-page detection",
        "output_dir": "stage2_screened_vlm",
    },
    {
        "name": "link",
        "script": "pipeline/stage3_link.py",
        "default_args": [],
        "gpu": False,
        "description": "Create symlinks for screening pages",
    },
    {
        "name": "recognize",
        "script": "pipeline/stage4_recognize.py",
        "default_args": [],
        "gpu": True,
        "description": "Table structure recognition (Kimi VLM)",
        "output_dir": "stage4_tables_recognized",
    },
    {
        "name": "complete",
        "script": "pipeline/stage4b_complete_tables.py",
        "default_args": [],
        "gpu": True,
        "description": "Complete cross-page tables using adjacent pages",
        "output_dir": "screening_tables_recognized",
    },
    {
        "name": "postprocess",
        "script": "pipeline/stage5_postprocess.py",
        "default_args": [],
        "gpu": True,
        "description": "Footnote separation post-processing (Kimi)",
        "output_dir": "stage5_tables_processed",
    },
    {
        "name": "detect_mols",
        "script": "pipeline/stage6_detect_mols.py",
        "alt_script": "pipeline/utils/detect_molecules_screening.py",
        "default_args": [],
        "gpu": True,
        "description": "Molecule bounding box detection",
        "output_dir": "stage6_mol_boxes",
    },
    {
        "name": "align_labels",
        "script": "pipeline/stage8_align_labels.py",
        "default_args": [],
        "gpu": True,
        "description": "VLM label alignment with external structure JSON",
        "output_dir": "stage8_mol_labels",
    },
    {
        "name": "table_align",
        "script": "pipeline/stage8b_table_align.py",
        "default_args": [],
        "gpu": True,
        "description": "Per-table VLM alignment of reaction slots to mol_ids (Kimi VLM)",
        "output_dir": "mol_table_alignments_si",
    },
    {
        "name": "cross_validate",
        "script": "pipeline/stage_cross_validate_labels.py",
        "default_args": [],
        "gpu": False,
        "description": "Cross-validate page- and table-level label assignments",
        "output_dir": "mol_labels_screening_si",
    },
    {
        "name": "assemble",
        "script": "pipeline/stage9_assemble.py",
        "default_args": [],
        "gpu": False,
        "description": "Final structured data assembly",
        "output_dir": "stage9_assembled",
    },
    {
        "name": "expand_rows",
        "script": "pipeline/stage10_expand_rows.py",
        "default_args": [],
        "gpu": False,
        "description": "Per-row reaction expansion & correction (Kimi VLM)",
        "output_dir": "stage10_expanded",
    },
    {
        "name": "fill_reagent_smiles",
        "script": "pipeline/stage_fill_reagent_smiles.py",
        "default_args": ["--offline"],
        "gpu": False,
        "description": "Fill common text-only reagent SMILES without network access",
        "output_dir": "screening_tables_expanded",
    },
    {
        "name": "fix_smiles",
        "script": "pipeline/fix_smiles.py",
        "default_args": ["--apply"],
        "gpu": False,
        "description": "Repair common invalid SMILES patterns",
        "output_dir": "screening_tables_expanded",
    },
    {
        "name": "build_benchmark",
        "script": "pipeline/utils/build_clean_benchmark.py",
        "default_args": [],
        "gpu": False,
        "description": "Build the clean benchmark and statistics",
        "output_dir": "benchmark_release",
    },
    {
        "name": "schema_convert",
        "script": "pipeline/stage11_convert_schema.py",
        "default_args": [],
        "gpu": False,
        "description": "Convert clean rows to the human-review schema",
        "output_dir": "schema_output",
    },
]

STAGE_NAMES = [s["name"] for s in STAGE_DEFS]
STAGE_MAP = {s["name"]: s for s in STAGE_DEFS}


def format_command(cmd: list[str]) -> str:
    """Shell-format a command while suppressing credential values."""
    safe = list(cmd)
    for index, token in enumerate(safe[:-1]):
        if token == "--api-key":
            safe[index + 1] = "<redacted>"
    return shlex.join(safe)


def load_progress() -> dict:
    """Load pipeline progress from file."""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {"completed": [], "failed": [], "running": None, "last_updated": None}


def save_progress(progress: dict):
    """Save pipeline progress to file."""
    progress["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def resolve_script(stage: dict) -> Path:
    """Find the script file, checking primary and alt paths."""
    script = stage["script"]
    # Support absolute paths directly
    primary = Path(script) if Path(script).is_absolute() else BASE_DIR / script
    if primary.exists():
        return primary
    alt = stage.get("alt_script")
    if alt:
        alt_path = Path(alt) if Path(alt).is_absolute() else BASE_DIR / alt
        if alt_path.exists():
            return alt_path
    return primary  # Will fail later with clear error


def build_stage_args(stage_name: str, args) -> list:
    """Build extra CLI arguments for a specific stage based on global args."""
    extra = []

    # Only pass options to scripts that explicitly support them.
    supports_test = {
        "render", "screen_text", "screen_vlm", "detect_mols",
        "align_labels", "table_align", "assemble", "expand_rows",
        "schema_convert",
    }
    supports_journal = (
        supports_test - {"schema_convert"}
    ) | {"fill_reagent_smiles"}
    supports_resume = {
        "render", "screen_text", "screen_vlm", "complete", "postprocess",
        "detect_mols", "align_labels", "table_align", "assemble",
        "expand_rows", "schema_convert",
    }
    if args.test > 0 and stage_name in supports_test:
        extra.extend(["--test", str(args.test)])
    if args.journal and stage_name in supports_journal:
        extra.extend(["--journal", args.journal])
    if args.resume and stage_name in supports_resume:
        extra.append("--resume")

    # Derive API chat completions URL from --api-url
    api_chat_url = None
    api_base_url = None
    if args.api_url:
        api_chat_url = args.api_url
        # stage2 (screen_vlm) uses --api-base which expects .../v1
        if "/v1/chat/completions" in args.api_url:
            api_base_url = args.api_url.replace("/chat/completions", "")
        elif args.api_url.endswith("/v1"):
            api_base_url = args.api_url
            api_chat_url = args.api_url + "/chat/completions"
        else:
            api_base_url = args.api_url.rstrip("/") + "/v1"
            api_chat_url = api_base_url + "/chat/completions"

    # Stages that call an LLM — they all accept --api-key / --usage-log.
    llm_stages = {"screen_vlm", "recognize", "complete", "postprocess",
                  "align_labels", "table_align", "expand_rows",
                  "schema_convert"}

    # Stage-specific args
    if stage_name == "render":
        if args.source == "si":
            extra.extend([
                "--papers-dir", "supporting_info_2025",
                "--output-dir", "all_si_pages_hires_2025",
            ])
        else:
            extra.extend([
                "--papers-dir", "papers_2025",
                "--output-dir", "all_pages_hires_2025",
            ])

    elif stage_name == "screen_vlm":
        if api_base_url:
            extra.extend(["--api-base", api_base_url])
        extra.extend([
            "--source", "si" if args.source == "si" else "main"
        ])
        if args.concurrency:
            extra.extend(["--concurrency", str(args.concurrency)])

    elif stage_name == "recognize":
        extra.extend([
            "--input-dir",
            "screening_tables_si"
            if args.source == "si"
            else "screening_tables_main",
            "--output-dir",
            "screening_tables_recognized",
        ])
        if api_chat_url:
            extra.extend(["--api-url", api_chat_url])
            if args.concurrency:
                extra.extend(["--concurrency", str(args.concurrency)])

    elif stage_name == "complete":
        if args.source == "si":
            extra.extend([
                "--recognized-dir", "screening_tables_recognized",
                "--input-dirs", "screening_tables_si",
                "--all-pages-dirs", "all_si_pages_hires_2025",
            ])
        else:
            extra.extend([
                "--recognized-dir", "screening_tables_recognized",
                "--input-dirs", "screening_tables_main",
                "--all-pages-dirs", "all_pages_hires_2025",
            ])
        if api_chat_url:
            extra.extend(["--api-url", api_chat_url])
        if args.concurrency:
            extra.extend(["--concurrency", str(args.concurrency)])

    elif stage_name == "link":
        if args.source == "si":
            extra.extend([
                "--source", "si",
                "--results-dir", "scope_screen_results_2025_si",
                "--images-dir", "all_si_pages_hires_2025",
                "--output-dir", "screening_tables_si",
                "--no-exclude",
            ])
        else:
            extra.extend([
                "--source", "main",
                "--results-dir", "scope_screen_results_2025",
                "--images-dir", "all_pages_hires_2025",
                "--output-dir", "screening_tables_main",
                "--no-exclude",
            ])

    elif stage_name == "postprocess":
        if api_chat_url:
            extra.extend(["--api-url", api_chat_url])
        if args.concurrency:
            extra.extend(["--concurrency", str(args.concurrency)])

    elif stage_name == "detect_mols":
        if args.source == "si":
            extra.extend([
                "--images-dir", "screening_tables_si",
                "--output-dir", "mol_boxes_decimer_si",
            ])
        else:
            extra.extend([
                "--images-dir", "screening_tables_main",
                "--output-dir", "mol_boxes_decimer",
            ])
        if args.detector == "opencv":
            pass  # detect_molecules_screening.py has its own defaults
        # decimer is the default script

    elif stage_name == "align_labels":
        if api_chat_url:
            extra.extend(["--api-url", api_chat_url])
        if args.api_model:
            extra.extend(["--api-model", args.api_model])
        if args.concurrency:
            extra.extend(["--concurrency", str(args.concurrency)])
        if args.source == "si":
            extra.extend([
                "--images-dir", "screening_tables_si",
                "--boxes-index", "mol_boxes_decimer_si/boxes_index.json",
                "--smiles-dir",
                args.smiles_dir or "mol_smiles_decimer_si",
                "--output-dir", "mol_labels_screening_si",
            ])
        else:
            extra.extend([
                "--images-dir", "screening_tables_main",
                "--boxes-index", "mol_boxes_decimer/boxes_index.json",
                "--smiles-dir",
                args.smiles_dir or "mol_smiles_decimer",
                "--output-dir", "mol_labels_screening",
            ])

    elif stage_name == "table_align":
        if api_chat_url:
            extra.extend(["--api-url", api_chat_url])
        if args.api_model:
            extra.extend(["--api-model", args.api_model])
        if args.concurrency:
            extra.extend(["--concurrency", str(args.concurrency)])
        extra.extend([
            "--labels-dir",
            "mol_labels_screening_si"
            if args.source == "si"
            else "mol_labels_screening",
            "--boxes-dir",
            "mol_boxes_decimer_si"
            if args.source == "si"
            else "mol_boxes_decimer",
            "--output-dir",
            "mol_table_alignments_si"
            if args.source == "si"
            else "mol_table_alignments_main",
        ])

    elif stage_name == "assemble":
        extra.extend([
            "--labels-dir",
            "mol_labels_screening_si"
            if args.source == "si"
            else "mol_labels_screening",
            "--alignments-dir",
            "mol_table_alignments_si"
            if args.source == "si"
            else "mol_table_alignments_main",
            "--smiles-dir",
            args.smiles_dir
            or (
                "mol_smiles_decimer_si"
                if args.source == "si"
                else "mol_smiles_decimer"
            ),
            "--output-dir",
            "screening_tables_final",
        ])
        if args.detector == "opencv":
            extra.extend(["--detection-method", "OpenCV"])

    elif stage_name == "cross_validate":
        extra.extend([
            "--labels-dir",
            "mol_labels_screening_si"
            if args.source == "si"
            else "mol_labels_screening",
            "--alignments-dir",
            "mol_table_alignments_si"
            if args.source == "si"
            else "mol_table_alignments_main",
        ])

    elif stage_name == "expand_rows":
        extra.extend([
            "--input-dir", "screening_tables_final",
            "--output-dir", "screening_tables_expanded",
            "--image-dir",
            "all_si_pages_hires_2025"
            if args.source == "si"
            else "all_pages_hires_2025",
            "--mol-labels-dir",
            "mol_labels_screening_si"
            if args.source == "si"
            else "mol_labels_screening",
        ])
        if api_chat_url:
            extra.extend(["--api-url", api_chat_url])
        if args.concurrency:
            extra.extend(["--concurrency", str(args.concurrency)])

    elif stage_name in {"fill_reagent_smiles", "fix_smiles"}:
        extra.extend(["--input-dir", "screening_tables_expanded"])

    elif stage_name == "build_benchmark":
        extra.extend([
            "--input-dir", "screening_tables_expanded",
            "--output-dir", "benchmark_release",
        ])

    elif stage_name == "schema_convert":
        extra.extend([
            "--input-dir", "benchmark_release/by_journal",
            "--output-dir", "schema_output",
            "--schema", "pipeline/schema/review_schema.json",
            "--images-dir",
            "all_si_pages_hires_2025"
            if args.source == "si"
            else "all_pages_hires_2025",
        ])
        if api_chat_url:
            extra.extend(["--api-url", api_chat_url])
        if args.concurrency:
            extra.extend(["--concurrency", str(args.concurrency)])

    # Common auth + token-log propagation for every LLM-facing stage.
    if stage_name in llm_stages:
        if args.api_key:
            extra.extend(["--api-key", args.api_key])
        if args.usage_log:
            extra.extend(["--usage-log", args.usage_log])
        if args.api_model and "--api-model" not in extra and "--model" not in extra:
            if stage_name == "recognize":
                # stage4_recognize.py uses --model (not --api-model)
                extra.extend(["--model", args.api_model])
            elif stage_name in {
                "screen_vlm", "complete", "postprocess", "expand_rows",
                "schema_convert",
            }:
                extra.extend(["--api-model", args.api_model])

    return extra


def run_stage(stage_name: str, args, progress: dict) -> bool:
    """Run a single pipeline stage. Returns True on success."""
    stage = STAGE_MAP[stage_name]
    script_path = resolve_script(stage)

    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False

    extra_args = build_stage_args(stage_name, args)
    cmd = [sys.executable, str(script_path)] + stage["default_args"] + extra_args

    print(f"\n{'='*70}")
    print(f"  Stage: {stage_name}")
    print(f"  Description: {stage['description']}")
    print(f"  Script: {script_path}")
    print(f"  GPU: {'yes' if stage['gpu'] else 'no'}")
    print(f"  Command: {format_command(cmd)}")
    print(f"{'='*70}\n")

    progress["running"] = stage_name
    save_progress(progress)

    t0 = time.time()
    try:
        # Use subprocess.run for GPU stages (clean GPU memory release)
        # For CPU-only stages, also use subprocess for isolation
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env={**os.environ, "PYTHONPATH": str(BASE_DIR)},
            check=False,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0

        if success:
            print(f"\n  Stage '{stage_name}' completed in {elapsed:.1f}s")
            if stage_name not in progress["completed"]:
                progress["completed"].append(stage_name)
            if stage_name in progress["failed"]:
                progress["failed"].remove(stage_name)
        else:
            print(f"\n  Stage '{stage_name}' FAILED (exit code {result.returncode}) "
                  f"after {elapsed:.1f}s")
            if stage_name not in progress["failed"]:
                progress["failed"].append(stage_name)

        progress["running"] = None
        save_progress(progress)
        return success

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  Stage '{stage_name}' EXCEPTION after {elapsed:.1f}s: {e}")
        if stage_name not in progress["failed"]:
            progress["failed"].append(stage_name)
        progress["running"] = None
        save_progress(progress)
        return False


def resolve_stages(args) -> list[str]:
    """Determine which stages to run based on args."""
    if args.stages:
        if args.stages == "all":
            stages = list(STAGE_NAMES)
        else:
            stages = [s.strip() for s in args.stages.split(",")]
            for s in stages:
                if s not in STAGE_MAP:
                    print(f"ERROR: Unknown stage '{s}'. "
                          f"Available: {', '.join(STAGE_NAMES)}")
                    sys.exit(1)
        return stages

    if args.start_from or args.stop_after:
        start_idx = 0
        stop_idx = len(STAGE_NAMES) - 1
        if args.start_from:
            if args.start_from not in STAGE_MAP:
                print(f"ERROR: Unknown stage '{args.start_from}'")
                sys.exit(1)
            start_idx = STAGE_NAMES.index(args.start_from)
        if args.stop_after:
            if args.stop_after not in STAGE_MAP:
                print(f"ERROR: Unknown stage '{args.stop_after}'")
                sys.exit(1)
            stop_idx = STAGE_NAMES.index(args.stop_after)
        return STAGE_NAMES[start_idx:stop_idx + 1]

    # Default: all stages
    return list(STAGE_NAMES)


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline orchestrator for screening table parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available stages: {', '.join(STAGE_NAMES)}")

    # Stage selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stages", type=str, default=None,
                       help="Comma-separated stage list, or 'all'")
    group.add_argument("--start-from", type=str, default=None,
                       help="Start from this stage (inclusive)")
    parser.add_argument("--stop-after", type=str, default=None,
                        help="Stop after this stage (inclusive)")

    # Common args
    parser.add_argument("--test", type=int, default=0,
                        help="Process only N items (passed to each stage)")
    parser.add_argument("--journal", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages already completed (from pipeline_progress.json)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if stage is marked completed")

    # Source / detector selection
    parser.add_argument("--source", type=str, default="si",
                        choices=["2025", "si"],
                        help="Image source: '2025' (main text) or 'si' (SI)")
    parser.add_argument("--detector", type=str, default="decimer",
                        choices=["decimer", "opencv"],
                        help="Molecule detector to use")
    parser.add_argument(
        "--smiles-dir",
        type=str,
        default=None,
        help=(
            "Directory of externally precomputed OCSR JSON; required before "
            "align_labels (source-specific legacy directory is the default)"
        ),
    )

    parser.add_argument("--api-url", type=str, default=None,
                        help="vLLM API URL，提供后所有 LLM 阶段自动使用 API 模式 "
                             "(例如 http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API 密钥。若中转站需要 Authorization，会自动下发到每个 LLM stage。"
                             "也可走 PROXY_API_KEY / OPENAI_API_KEY 环境变量")
    parser.add_argument("--usage-log", type=str, default=None,
                        help="Token 用量 JSONL 日志路径 (默认 logs/api_usage.jsonl)")
    parser.add_argument("--api-model", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=None,
                        help="API 并发数 (传递给支持并发的阶段)")

    # Control
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stages without executing")
    parser.add_argument("--stop-on-error", action="store_true", default=True,
                        help="Stop pipeline on first error (default: True)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue to next stage even if current fails")
    parser.add_argument("--status", action="store_true",
                        help="Print pipeline progress and exit")

    args = parser.parse_args()

    if args.continue_on_error:
        args.stop_on_error = False

    # Status mode
    if args.status:
        progress = load_progress()
        print("Pipeline progress:")
        print(f"  Last updated: {progress.get('last_updated', 'never')}")
        print(f"  Running: {progress.get('running', 'none')}")
        print(f"  Completed: {progress.get('completed', [])}")
        print(f"  Failed: {progress.get('failed', [])}")
        remaining = [s for s in STAGE_NAMES
                     if s not in progress.get("completed", [])]
        print(f"  Remaining: {remaining}")
        return

    # Resolve stages
    stages = resolve_stages(args)
    progress = load_progress()

    api_required = {
        "complete", "postprocess", "table_align", "schema_convert"
    }
    missing_api = sorted(api_required.intersection(stages))
    if missing_api and not args.api_url and not args.dry_run:
        parser.error(
            "--api-url is required for these selected stages: "
            + ", ".join(missing_api)
        )

    # Resume: skip already completed stages
    if args.resume and not args.force:
        original = list(stages)
        stages = [s for s in stages if s not in progress.get("completed", [])]
        skipped = len(original) - len(stages)
        if skipped:
            print(f"Resume: skipping {skipped} completed stages: "
                  f"{[s for s in original if s not in stages]}")

    needs_external_structures = {"align_labels", "assemble"}.intersection(stages)
    if needs_external_structures and not args.dry_run:
        smiles_dir = Path(
            args.smiles_dir
            or (
                "mol_smiles_decimer_si"
                if args.source == "si"
                else "mol_smiles_decimer"
            )
        )
        if not smiles_dir.is_absolute():
            smiles_dir = BASE_DIR / smiles_dir
        if not smiles_dir.is_dir():
            parser.error(
                "External OCSR outputs are required before align_labels; "
                f"provide --smiles-dir (not found: {smiles_dir})"
            )

    # Swap detect_mols script based on detector choice
    if args.detector == "opencv":
        STAGE_MAP["detect_mols"]["script"] = "pipeline/utils/detect_molecules_screening.py"

    print("=" * 70)
    print("  Screening Table Structured Parsing Pipeline")
    print("=" * 70)
    print(f"  Stages to run: {stages}")
    print(f"  Source: {args.source}")
    print(f"  Detector: {args.detector}")
    if args.test:
        print(f"  Test mode: {args.test} items")
    if args.journal:
        print(f"  Journal filter: {args.journal}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would execute the following stages:\n")
        for i, stage_name in enumerate(stages, 1):
            stage = STAGE_MAP[stage_name]
            script = resolve_script(stage)
            extra = build_stage_args(stage_name, args)
            stage_args = stage["default_args"] + extra
            cmd = format_command(["python", str(script), *stage_args])
            status = "COMPLETED" if stage_name in progress.get("completed", []) else "pending"
            print(f"  {i}. [{status}] {stage_name}: {stage['description']}")
            print(f"     {cmd}")
        return

    # Execute stages
    total = len(stages)
    succeeded = 0
    failed = 0
    t0 = time.time()

    for i, stage_name in enumerate(stages, 1):
        print(f"\n>>> Running stage {i}/{total}: {stage_name} <<<")
        ok = run_stage(stage_name, args, progress)
        if ok:
            succeeded += 1
        else:
            failed += 1
            if args.stop_on_error:
                print(f"\nPipeline stopped at stage '{stage_name}' due to error.")
                print("Use --continue-on-error to skip failed stages.")
                break

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  Pipeline finished in {elapsed:.1f}s")
    print(f"  Succeeded: {succeeded}/{total}")
    print(f"  Failed: {failed}/{total}")
    if failed:
        print(f"  Failed stages: {progress.get('failed', [])}")
    remaining = [s for s in STAGE_NAMES if s not in progress.get("completed", [])]
    if remaining:
        print(f"  Remaining stages: {remaining}")
    else:
        print("  All stages completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
