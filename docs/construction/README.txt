Offline question-construction adapter and reconstruction audit

Prepared for the camera-ready reproducibility release. No remote files or
submitted question sets were modified. The model outputs were not touched.

Current publication scope:
The local candidate retains 777 final reviewed files from source articles
represented in the final benchmark. It omits 107 files from 42 other source
articles, along with automatic extraction drafts, caches and editing history.
The source-level selection does not prove a final-question association for
every retained table or complete coverage of every final table. The adapter
uses the current reviewed_data/manifest.json file set and hashes.

Historical experiment scope:
The reconstruction and comparison results below used the earlier full
884-file reviewed snapshot. Their original numbers and JSON reports are
retained as historical evidence, not as a new build or validation of the
777-file publication subset. No new construction result is asserted here.

Publishable GitHub code:
  rebuild_from_reviewed_records.py -> scripts/rebuild_from_reviewed_records.py
  audit_reviewed_reconstruction.py -> scripts/audit_reviewed_reconstruction.py
  audit/construction_reconstruction_summary.json -> metadata/construction/

The optional audit/reviewed_table_id_map.json contains real dataset IDs and
is included only in the controlled-access data package. All its associated
reviewed files remain in the current subset. The full reconstruction_comparison
working report stays local and is not included in this minimal package.
Historical audit summaries retain their earlier full-snapshot scope; they are
not the current reviewed-file manifest. Do not include trial_reviewed_stable_ids/
in the public code repository: it contains real reviewed records and questions.

Prerequisite: Python 3.10+ and RDKit. The adapter itself uses the standard
library, but the unmodified archived common_v6.py imports RDKit.

From the final code repository, with a downloaded authorized HF snapshot:
  python scripts/rebuild_from_reviewed_records.py \
    --artifact-root /path/to/HF-snapshot \
    --output-dir /path/to/new-reviewed-rebuild

The code directory defaults to code/ adjacent to scripts/. A sibling
evaluation/common_v6.py is required. --code-root permits another explicit
code/evaluation checkout. The output directory must not exist. No API/network
calls are performed. No publisher documents, table images, OCSR models, or
credentials are required. Extraction/API stages and human review are not rerun.

Integrity and conversion:
- Require the exact reviewed_data manifest file set and SHA-256 bytes.
- Retain the JSON mapping's insertion order and original row labels.
- Require exactly one reaction per row; reject ambiguous flattening.
- Deep-copy every existing reaction field unchanged and add only meta_info.
- Reviewed snapshots omit extraction meta_info. Default root_img_name IDs are
  explicitly new stable reviewed filename stems, not original submitted IDs.
- An optional --table-id-map can use an explicit hash-bound identity bridge.
  The earlier full-snapshot audit derives 416 one-to-one candidate legacy parent-ID associations from
  unique unchanged MC input/condition/option content within the same source DOI.
  These are content-matching evidence, not an authoritative historical lineage
  record. They have NOT been applied to frozen questions or the default rebuild.

Scoring boundary:
The archived V6 builders retain historical major-fraction/parsed ratio utility.
This adapter does not implement or claim fixed-target dr/rr relabeling. A
target-aware policy must be separately source-adjudicated and use a versioned
builder/evaluator tree; choosing --code-root alone does not establish that
chemical verification. The frozen benchmark remains the reference for
reproducing the paper's historical model results.

Historical completed test (full 884-file snapshot):
The earlier full offline build from all 884 reviewed table files completed using the
archived 0.2.4 builders. It generated 4,473 questions versus 4,773 frozen ones:
  MC single:                    962 vs 1,039
  MC all:                       553 vs   552
  MC single reference/control:  769 vs   850 each
  MC all reference/control:     289 vs   310 each
  FIB all:                      553 vs   552
  FIB reference:                289 vs   310

After comparison-only normalization of option/candidate ordering and
provenance fields, the 416 content-associated parents give 3,624 paired
questions, of which 3,049 match literally in all semantic fields. Differences
include displayed condition text, reference examples, candidate-value
representations and some answer-value representations. These literal
differences must not all be classified as chemical or scoring errors.
Literal comparison counts typography differences too;
e.g. [Rh(COD)Cl]2 vs [Rh(COD)Cl]₂. No chemistry or labels were changed to force
agreement. Refer to the JSON reports rather than claiming exact regeneration.

An exact rebuild of the frozen submitted questions was not established by
that earlier full reviewed snapshot plus archived released code, and is not
claimed for the current publication subset.
The missing historical inputs/ID metadata and observed content differences
must remain an explicit reproducibility limitation. Do not replace canonical
questions or re-score existing model outputs against an adapter-generated build.
