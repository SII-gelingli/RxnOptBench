# RxnOptBench

**Camera-ready code preview — `v0.2.6-code-preview`.** The new data package
and matching final Croissant have not been released. This preview contains code,
prompts, synthetic examples and explicitly draft metadata. The existing
[`v0.2.5-eval-only`](https://github.com/SII-gelingli/RxnOptBench/tree/v0.2.5-eval-only)
release remains available with its matching data and instructions.

**RxnOptBench: Benchmarking LLMs for Reaction-Condition Optimization in Organic Methodology**

Lingli Ge, Yubin Wang, Junyuan Gao, Jiahe Song, Jiaxing Sun, Boyu Zhu,
Haote Yang, Jingchao Wang, Lixin Ma, Jiang Wu, Yuqiang Li, and Conghui He.

RxnOptBench evaluates reaction-condition selection over alternatives reported
in methodology screening tables. The frozen benchmark has eight test subsets,
4,773 questions and 552 parent tables.

This repository contains the authors' available extraction stages and input
interfaces, benchmark-construction code, deterministic evaluators, six prompt
templates, parsers, validators and human-review interface. Real benchmark
records, 777 retained final reviewed tables from sources represented in the
final benchmark, model outputs and data-derived
examples are prepared for separate distribution through the
[Hugging Face dataset](https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final)
under its noncommercial research terms and access agreement. The enlarged data
package has not yet been published; the dataset page currently serves the older
evaluation-only release. Source-rights scope for the new package remains under review.

Original software, prompt templates, documentation and invented examples are
MIT-licensed. Installed dependencies retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSE_SCOPE.md](LICENSE_SCOPE.md).
[UPSTREAM_LICENSE_NOTICES.md](UPSTREAM_LICENSE_NOTICES.md) explains the separate
rights in literature-derived material. The data-host candidate includes public
source-category evidence; it does not claim project-wide clearance or settle
unknown historical acquisition/API hosting and remaining SI/intermediate-note scope.

## Files

| Path | Contents |
|---|---|
| `evaluation/` | Standalone MC/FIB evaluators and archived construction helper |
| `code/` | Available extraction stages, question builders, inference/parser and review tool |
| `prompts/` | Six prompt templates; controls reuse the corresponding no-reference template |
| `scripts/` | Offline integrity/prompt/output validator, policy relabeler and construction adapter |
| `examples/synthetic/` | Invented interface examples containing no benchmark records |
| `analysis/` | Analysis programs; their data and historical results are in the gated artifact |
| `docs/` | Extraction and reproducibility boundaries |

## Evaluation without data download

Python 3 alone is sufficient for the standalone evaluators:

```bash
python3 evaluation/evaluate_multiple_choice.py \
  --dataset examples/synthetic/multiple_choice.json \
  --prediction examples/synthetic/prediction_multiple_choice.json --detail
python3 evaluation/evaluate_fill_in_blank.py \
  --dataset examples/synthetic/fill_in_blank.json \
  --prediction examples/synthetic/prediction_fill_in_blank.json --detail
```

The toy MC accuracy/score are 0.5/0.7; toy FIB Hit, Score | Hit and Strict are
2/3, 0.75 and 1/3. These are invented examples, not experimental results.

## Download and validate the matching data package (pending)

**Do not use the current HF `main` snapshot with this preview validator.** It
serves the older evaluation-only release and lacks the enlarged artifact files.
The following commands are for the matching `v0.2.6-target-policy` data release
after it is published; that data revision is not currently available. To use the
existing release now, follow the linked `v0.2.5-eval-only` instructions instead.

Accept the agreement on the dataset page using your own Hugging Face account,
then authenticate the official CLI locally. Keep credentials out of source
files, logs and commits. Download to a separate directory:

```bash
hf download SII-gelingli/RxnOptBench-final --repo-type dataset \
  --revision v0.2.6-target-policy --local-dir ../RxnOptBench-data
python3 scripts/validate_artifacts.py \
  --artifact-root ../RxnOptBench-data --snapshot final \
  --output ../validation/final.json
python3 scripts/validate_artifacts.py \
  --artifact-root ../RxnOptBench-data --snapshot historical \
  --output ../validation/historical.json
```

The validator runs offline. It checks eight native/JSONL pairs, scoped IDs,
all 4,773 rendered prompts and stored constant conditions, the reviewed
file set and hashes declared by its manifest, 12 models × 8 tasks × 3 saved files, and three archived full
DeepSeek repeat runs (72 additional files). It scores all eight tasks
with the selected gold snapshot and reports the exact denominators and recorded
terminal API exclusions. All 11 API-timeout placeholders are evaluated as no
answer, including reference tasks, so a placeholder integer zero cannot receive
credit. Only the historical headline MC-all/FIB exclusions (5 and 1) are
dropped; reference tasks use full N. `--include-all-submitted` makes a separate sensitivity
calculation including API-skip entries; it does not modify the archived run.

FIB Hit is the fraction of submitted complete tuples matching an observed
row; Strict is the fraction matching a tied-best observed row. Score | Hit
averages normalized utility over observed-row hits. Well-formed unobserved
tuples count as non-Hits and remain in both submitted-prediction denominators.
They have no measured outcome and are not labeled chemically invalid.

## Dataset versions and response-text provenance

Candidate indices are version-specific. The historical snapshot is under
`historical/submitted_v0.2.4/` in the gated artifact; the active `data/` snapshot
is identified by its `VERSION` and policy manifests. Pair outputs with the
corresponding candidate space and scoring policy. Candidate values, inputs,
IDs and observed condition support are unchanged by the target-policy update.

The artifact preserves `metadata/result_index_restoration.json`, including
before/after SHA-256 hashes, for the deterministic FIB conversion from audit
permutation to source order. Both parsed predictions and structured answers
inside some `*_raw.json` strings were changed during that conversion. Those
files are **derived response-text artifacts**, not byte-for-byte unaltered
provider responses. The conversion preserves selected chemical values. Its
existence does not establish which ordering a model actually received.

`historical/.../results/table2_metrics.tsv` preserves the historical printed
table. The validator independently exposes its ChemLLM Strict discrepancy
(printed 6.6%, retained predictions 29/552 = 5.3%) without inventing a parser
maintenance explanation. Current-policy scores are computed separately.

## Construction and extraction boundary

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The original builders
and a portable adapter are included. The candidate retains 777 final reviewed
files from final-benchmark source articles; 107 files from 42 other source
articles are omitted. This source-level selection does not establish that every
retained table contributed to a final question. Automatic extraction drafts,
caches and editing history are also omitted.

An earlier offline reconstruction and comparison used the full 884-file
reviewed snapshot and did not exactly regenerate the frozen eight subsets.
Its archived counts describe that earlier experiment, not a new build from
the 777-file publication subset. Exact construction lineage remains unresolved.
The frozen questions and all saved-output metrics can be inspected and
rescored directly without rerunning hosted models.

The extraction release deliberately omits the unreleased OCSR model, its
inference/model-specific postprocessing, weights, third-party source PDFs,
SI and source-page images. Available stages consume lawfully supplied inputs
and structured external OCSR JSON. API-dependent stages require user-provided
services; running `--stages all` is not a self-contained raw-PDF reconstruction.
