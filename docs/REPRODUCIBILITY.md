# Reproducibility and known limitations

The code and gated artifact have separate roots. Download real records through
the HF agreement and pass that directory explicitly to portable scripts.
Standalone evaluation and prompt audits need only Python 3; construction also
needs RDKit. Extraction/analysis requirements are listed separately and are
not a universal hardware/software lockfile.

## What is reproducible offline

- Deterministic MC/FIB scoring from frozen native questions and archived parsed
  predictions, for all 12 models and eight subsets.
- Native/JSONL content and order checks; per-subset ID coverage and reviewed
  record hashes; six-template rendering across all 4,773 questions.
- Candidate-index restoration is documented by its operation manifest. Never
  apply it twice to already restored outputs.
- The current-policy labels can be regenerated from the historical questions
  and explicit target-policy ledger. No completed fixed-target human ledger
  was supplied for this candidate, so every unverified dr/rr axis is excluded
  from scoring and retained in raw metadata. This is not source adjudication.

## Construction gap

The available construction scripts expect list-format table inputs carrying
original table/row metadata. The reviewed records use a later schema keyed by
row index and omit that original identity metadata. The local publication
candidate retains 777 final reviewed files from final-benchmark source articles;
107 files from 42 other source articles are omitted. This source-level selection
does not establish that every retained table contributed to a final question.
Automatic extraction drafts, caches and editing history are not published.

A portable adapter reads the current reviewed-data manifest and constructs
questions without API requests. An earlier archived-semantics experiment used
the full 884-file snapshot and produced MC-single 962,
MC-all/FIB 553, single-reference/control 769 and all/FIB-reference 289,
compared with frozen counts 1,039, 552, 850 and 310. Missing identity metadata
alone does not explain all of these count differences. Do not claim these
records and scripts exactly regenerate the frozen benchmark. Those counts
describe the earlier full-snapshot experiment, not a new build from the current
777-file publication subset. The archived construction reports retain their
original scope and values.

The saved benchmark remains the authoritative evaluation input for its named
version. Do not replace it with adapter output or copy frozen answers into
rebuilt questions and call that independent reconstruction. The construction
comparison records unresolved gaps; no model rerun repairs a gold-provenance gap.

## External stages

The extraction pipeline includes local PDF rendering/joins/checks and optional
VLM/API/DECIMER stages. Source PDFs, SI, page images, model weights, the
unreleased OCSR module and its model-specific postprocessing are excluded.
The documented interface accepts external structured OCSR JSON; it does not
provide the missing recognizer. API-dependent stages were not rerun in this
release preparation, and no paid model calls are required by validation.

Human review occurs after extraction/schema conversion and before question
construction. The interface can accept/edit/reject locally supplied records;
the artifact retains final reviewed JSON, not identities, statuses or a full
edit-event history. The snapshot is not proof of target-specific ratio
verification.

## Hosted model and hardware limits

The artifact retains recoverable run settings in `results/run_manifest.tsv`,
not immutable server implementations or a complete per-model GPU/memory/time
inventory. Temperature zero does not freeze provider aliases or service
behavior. Saved outputs support score reproduction, not byte-for-byte replay
of nondeterministic or retired hosted services. FIB response text is a derived
index-restored artifact where identified by the restoration manifest.

## Portable construction and versioned-policy commands

```bash
python3 scripts/rebuild_from_reviewed_records.py \
  --artifact-root ../RxnOptBench-data --output-dir ../new-reviewed-rebuild
python3 scripts/audit_reviewed_reconstruction.py \
  --artifact-root ../RxnOptBench-data --build-dir ../new-reviewed-rebuild \
  --reference-data-dir ../RxnOptBench-data/historical/submitted_v0.2.4/data/original_json \
  --output-dir ../new-construction-audit
python3 scripts/relabel_target_policy.py \
  --source ../RxnOptBench-data/historical/submitted_v0.2.4 \
  --output ../new-target-policy
```

The policy command above starts from frozen historical questions, excludes all
unverified dr/rr, and makes no model calls. A supplied human target ledger can
include an axis only when it satisfies the script's explicit component, target
and per-row evidence schema. Relabeling gold does not regenerate question
chemistry or establish missing historical construction lineage. The construction
adapter keeps archived V6 semantics and is a separate experiment.

The earlier full-884-snapshot content-association comparison found 416 one-to-one parent candidates; these
are evidence-based associations rather than original extraction identity logs.
Among 3,624 paired questions, 3,049 matched all compared semantic fields after
provenance removal and correct candidate/answer reindexing. Literal differences
include typography, reference examples and candidate/answer representations;
they are not all established chemical errors. Default reconstruction does not
apply the optional association map or overwrite frozen inputs. These comparison
counts are not validation results for a new build from the current publication
subset.

## Reproduce the updated numerical tables

```bash
python3 -m pip install -r scripts/requirements-scoring.txt
python3 scripts/recompute_alignment.py --artifact-root ../RxnOptBench-data \
  --dataset-root ../RxnOptBench-data/data/original_json --output ../new-scores
python3 scripts/rescore_repeatability.py --artifact-root ../RxnOptBench-data \
  --output-dir ../new-repeat-scores
```

The first command independently computes all 12 models/eight tasks, old/new
gold differences, per-key and paired-context summaries, uniform-random baselines
and full-N timeout diagnostics. Its main-table convention preserves six
historical DeepSeek skips; all 11 timeout placeholders across tasks are first
treated as no-answer, so a stored zero never receives option-0 credit. The
second command rescores the three archived full repeat runs through exact
semantic candidate mappings, with no new API calls. Its source manifest has
been sanitized to remove service endpoints and rewrite paths; the original
manifest hash and transformation are recorded separately. Contemporaneous
HTTP request bytes and original input hashes were not recorded.

Precomputed new outputs reside under the data artifact's `metadata/scoring/`
and `metadata/repeatability/`, while the 72 repeat response/parse artifacts
remain in `results/deepseek_repeatability/outputs/`. These retain all eight
subsets and allow independent verification of the seven reported mean/range
metrics. Aggregate agreement validates arithmetic, not missing source rights
or human adjudication.
