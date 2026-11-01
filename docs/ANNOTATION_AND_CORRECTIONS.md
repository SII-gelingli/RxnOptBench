# Human Annotation and Final Corrections

## Released evidence

The release exposes two parts of the human-review boundary:

1. `code/annotation_tool/server.py`: the path-configurable review interface.
2. `reviewed_data/tables/`: public field exports of 777 final reviewed structured tables from
   source articles represented in the final benchmark.

`reviewed_data/manifest.json` records the exact relative path and SHA-256
digest of every retained table. These final records are the inputs needed for
scientific inspection of the reviewed extraction boundary. The release omits
107 files from 42 source articles outside the final benchmark. Selection is by
source article; it does not establish a final-question association for every
retained table or complete coverage of all final tables.

## Intentionally omitted material

The release does not include automatic extraction drafts, caches,
pre-review input tables,
per-table status or annotator records, saved edited copies, rejected or
pending records, operational timestamps, or a revision-by-revision edit
history. Those files are workflow metadata rather than benchmark questions or
final reviewer evidence.

The absence of that intermediate material means this release is a final-state
snapshot, not an immutable annotation event log. It supports the claim that
the retained tables are the final reviewed records; by itself it does not
establish the number or identity of annotators, inter-annotator agreement, or
a fully versioned correction history.

## Source-material boundary

The annotation interface accepts local image paths when used by a curator, but
this repository contains no publisher PDFs, supporting-information files,
source-page images, or table crops. The released JSON records retain
normalized DOI/table provenance without redistributing the formatted source
documents.

## Public support-record field export

The 777 support records omit freeform explanations and structural descriptions;
all retained values and entry order are preserved. `reviewed_data/README.md`
and `public_export_manifest.json` list the exact boundary and original/public
hashes. Some omitted notes carried additional context; this export does not
reconstruct that context or claim exact final-QA regeneration. Final questions,
labels, archived predictions and evaluation rules are unchanged.
