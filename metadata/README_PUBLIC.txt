RxnOptBench 0.2.6-target-policy: unpublished local Croissant candidate

Status
This candidate has not been uploaded or published. croissant.draft.json contains
complete Croissant 1.1 core and Responsible AI metadata for 8 test RecordSets,
4,773 questions. The 24 distribution SHA-256 values describe local candidate
files, not the earlier data currently available from the repository's main URLs.
There is no claimed immutable remote revision, remote-loading result, or online
checker approval for this candidate. The loader refuses draft remote loading.

Scope and policy
The Croissant scope is the eight evaluation subsets only. The broader candidate
contains 777 reviewed records conservatively limited to the final 232 source
groups. Those records, prompts, archived outputs, and available pipeline/review/
construction code are tracked by a separate manifest, outside these RecordSets.
The current bridge covers 416 final tables. Twelve final source groups have no
matching reviewed record; mappings are not invented to fill those gaps.
An earlier construction experiment produced 4,473 questions from a historical
full 884-record review snapshot. That experiment does not describe the current
777-record scope. Construction has not been rerun for the reduced snapshot, and
exact regeneration of the final 4,773 questions is not claimed. Exact lineage
from the reviewed snapshot to the final questions remains incomplete.
Publisher PDFs/images, weights, the unreleased molecular-recognition model, and
its model-specific postprocessing are excluded. Full reconstruction from raw
publisher PDFs is not asserted.

Unverified dr/rr axes do not enter utility or gold labels. The 65 observed
table/axis combinations have no independent human-verified fixed-target mapping
ledger in the workspace and are all excluded. Their parsed observations remain
in raw metadata; score-bearing dr/rr fields are null. Existing ee/er treatment is
unchanged. Input prompts, candidate indices, and reference/control contexts stay
unchanged; labels, ties, effective scores, and relative scores are recomputed.
The per-question change and historical-scoring manifests document those changes.
An old non-best reference row can become best under the corrected utility;
unchanged contexts must not be called guaranteed best-withheld contexts.
References are source-row-disjoint from candidates; repeated source rows can
still carry identical full conditions. Some variants contain all-option ties.
Reference/control pairs match candidate sets while many released option orders
differ. Full provider prompt-order logs have not been independently verified.
These comparisons do not isolate a causal reference effect.

The question union represents 4,768 observed rows from 552 parent tables. Of
these, 298 tables have every stated source row represented and 254 do not.
meta.global_best_scope is all_released_observations_for_parent_table, not all
unreleased original source rows and not chemical space. Individual question-best
labels use current_question_observed_candidates. The 236 raw source strings
reduce to 232 case-insensitive source groups. A separate public bibliographic
audit matched their 232 DOI identifiers; identity matching is not permission.
Sources are recent organic-methodology literature and include publisher-confirmed
2026 first publications; an exclusively 2025 first-publication claim is not made.

Synthetic-data disclosure refers to constructed questions, masking, contexts,
and computed annotations. The experimental measurements are literature-derived.
IDs are unique within a subset; use (subset, id) when combining RecordSets.

Source-category evidence and remaining scope
Public bibliographic/license evidence has been organized for 232 final-source
DOIs across six publisher families. The HF candidate's metadata/source_rights/
contains evidence inventories, including 102 ACS SI-license entries. These are
not a blanket clearance finding. Historical acquisition dates, institutional
subscriptions and API hosting remain unknown. Some actual SI correspondence,
third-party notices and expression in the 777 source-relevant reviewed records
still have unresolved scope. Limiting records to final source groups does not
establish item-level license clearance. No claim is made that the broader
package contains only facts or no source-derived expression. No new author
maintenance guarantee is introduced.

UPSTREAM_LICENSE_NOTICES.md at the intended hosts explains that author-owned
QA contributions use the existing data terms while upstream licensed objects
retain their original permissions. The custom agreement does not reduce rights
independently granted under applicable Creative Commons terms.

Intended host: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final
Code: https://github.com/SII-gelingli/RxnOptBench
Selected terms: code MIT; data RxnOptBench Noncommercial Research License v1.0.
The host's current public gated version is an earlier snapshot.

Transport
Each TSV.gz exposes id, question_type, parent_table_id, and json_record. Decoding
json_record restores the complete candidate record, including nested objects.
Native JSON and JSONL are equivalent canonical data; record_schema.json documents
all nested structure. transport/ contains real dataset records and belongs only
in the data-hosting candidate, never a public code-only GitHub release.

Load local candidate files
Place the public code checkout in RxnOptBench and the corresponding data
candidate in a sibling RxnOptBench-data directory. From the code checkout:
  python -m pip install -r metadata/requirements.txt
  python metadata/load_croissant.py metadata/croissant.draft.json --transport-dir ../RxnOptBench-data/data/croissant --subset multiple_choice_all_varying
Use the local candidate transport files supplied with the matching descriptor;
the earlier remote main data will not match this descriptor's hashes. When
working inside the data candidate instead, equivalent metadata/ and
data/croissant/ paths may be supplied explicitly. Draft remote loading is
disabled. Internal packaging/build commands are documented in README_LOCAL.txt;
the public metadata file set does not include those internal scripts.

Validation verifies hashes, standard reference-loader mapping, 8/4,773 record
counts, exact native/JSONL/loaded-record equality, and nested JSON Schema. It does
not prove chemical/scientific correctness, upstream rights, remote availability,
or acceptance of the gate. The validation report records this limited scope.

Official specification: https://docs.mlcommons.org/croissant/docs/croissant-spec.html
Conference hosting: https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting
