# Pipeline specification

## Stage graph

1. `render`: render main-text or SI PDFs to high-resolution page images.
2. `screen_text`: produce a diagnostic list of likely pages from PDF text.
3. `screen_vlm`: independently screen the selected rendered-page tree and
   detect cross-page tables.
4. `link`: gather confirmed pages into the working image directory.
5. `recognize`: transcribe tables into structured JSON.
6. `complete`: merge incomplete tables with content from adjacent pages.
7. `postprocess`: separate footnote markers from table values.
8. `detect_mols`: detect molecular-structure bounding boxes with DECIMER.
9. External OCSR boundary: supply molecule-to-SMILES JSON produced outside
   this release.
10. `align_labels`: align page-level structure boxes to author labels and copy
    the externally supplied SMILES fields without modifying them.
11. `table_align`: align table reaction slots to structure boxes.
12. `cross_validate`: backfill and cross-check page/table label assignments.
13. `assemble`: join table, external OCSR, label, and alignment streams.
14. `expand_rows`: turn each optimization row into a complete reaction record.
15. `fill_reagent_smiles`: fill high-confidence text-only reagent SMILES.
16. `fix_smiles`: apply generic, rule-based validation and repair to retained
    SMILES fields.
17. `build_benchmark`: retain fully specified clean tables and emit statistics.
18. `schema_convert`: use the source page and a VLM to map each clean row into
    the structured schema consumed by the human-review tool.

## Model boundaries

- `render`, `screen_text`, `link`, `cross_validate`, `assemble`,
  `fill_reagent_smiles` in offline mode, `fix_smiles`, and `build_benchmark`
  are local transformations for fixed software versions and inputs.
- `screen_vlm`, `recognize`, `complete`, `postprocess`, `align_labels`,
  `table_align`, `expand_rows`, and `schema_convert` call a VLM/API.
- `detect_mols` uses DECIMER-Segmentation by default.
- The study used an unreleased OCSR model accessed through an API. Its model,
  inference script, raw-output parsing, graph-to-SMILES conversion, and
  superatom expansion are not distributed in this reviewer snapshot.
- SMILES returned by alignment VLMs are not trusted. Those stages resolve
  structures from externally supplied OCSR outputs by box identifier and do
  not alter those SMILES.

The numbered list above is presentation order, not the historical filenames'
stage-number notation (which includes `4b` and `8b`).

The release orchestrator does not apply an impact-factor-based journal
exclusion when linking confirmed pages.

## Cross-page tables

`recognize` marks incomplete tables. `complete` reads the target and adjacent
pages, returns only the named incomplete table, and merges headers, rows, and
footnotes. The stage is explicit in the reviewer orchestrator and is included
by `--stages all`.

## Human review

The human-review gate follows schema conversion and input curation. Annotators
can accept, reject, skip, or edit a table while the selected source JSON
remains read-only. Saved edited copies override the corresponding source table
when building the final reviewed snapshot. The release
contains public field exports of 777 final reviewed tables from source articles represented in the final
benchmark under `reviewed_data/tables/`, with public per-file hashes in
`reviewed_data/manifest.json` and original/public hashes in the export manifest. It omits 107 files from 42 other source articles.
This source-level subset does not establish a final-question association for
every retained table or exact reconstruction of the benchmark. Automatic
extraction drafts, caches, pre-review candidates, status records, saved edited
copies, rejected/pending records, and intermediate histories are intentionally
omitted.

## Clean benchmark rule

A whole table is retained only when every expanded reaction row has:

- at least one substrate and product;
- non-empty substrate/product SMILES;
- no wildcard `*` in substrate/product SMILES; and
- no wildcard inside condition strings encoded as `[SMILES: ...]`.

The whole-table rule avoids mixing fully resolved and partially failed rows
from the same optimization experiment.

`build_benchmark` also groups retained clean records under
`benchmark_release/by_journal/`. `schema_convert` mirrors those table paths
under `schema_output/`, with each row index mapped to a list of schema-format
reaction objects. The conversion prompt preserves input SMILES and separates
conversion, selectivity, and yield semantics. Since the VLM stages are
model-dependent, rerunning them creates candidate outputs rather than a claim
of byte-identical regeneration of the bundled pre-review snapshot.

## Public support-record field export

The 777 support records omit freeform explanations and structural descriptions;
all retained values and entry order are preserved. `reviewed_data/README.md`
and `public_export_manifest.json` list the exact boundary and original/public
hashes. Some omitted notes carried additional context; this export does not
reconstruct that context or claim exact final-QA regeneration. Final questions,
labels, archived predictions and evaluation rules are unchanged.
