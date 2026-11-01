# Models and runtime configuration

## Models used in the study

The extraction pipeline used the following model boundary:

- Molecular-structure detection (`detect_mols`): DECIMER-Segmentation.
- Molecular-structure recognition: an unreleased OCSR model accessed through
  an API, represented in this release only by its externally supplied
  structured outputs.
- All other VLM stages (`screen_vlm`, `recognize`, `complete`,
  `postprocess`, `align_labels`, `table_align`, `expand_rows`, and
  `schema_convert`): Kimi-K2.6 through an OpenAI-compatible API.
- Rendering, linking, local cross-validation, assembly,
  benchmark filtering, and metric calculation: deterministic local code for
  fixed inputs and software versions.

The unreleased OCSR model and its model-specific inference/post-processing
implementation are not redistributed. This includes raw-output parsing,
graph-to-SMILES conversion, and superatom expansion. Generic downstream SMILES
validation and repair remain included. The released reviewed records,
question-generation code, evaluation code, and saved benchmark outputs define
the auditable release boundary.

## Runtime configuration

The stage scripts expose the sampling and systems settings used by each call,
including temperature, token limit, timeout, concurrency, resume behavior,
and local device selection. `pipeline/run_pipeline.py` forwards a single
explicit `--api-model` value to every API-backed VLM stage. The released
defaults use `kimi-k2.6`; an explicit immutable provider revision should be
recorded when the provider exposes one.

The inference runner in `code/inference_open_source.py` renders the evaluated
templates in `prompts/` against the eight datasets in `data/original_json/`.
All six templates include table-wide constant conditions; the two
single-varying MC templates additionally include task-specific fixed
conditions. The corresponding question builders are in
`code/build_fill_in_blank.py` and `code/build_multiple_choice.py`.

For the released benchmark runs, `results/run_manifest.tsv` records the exact
requested model identifier available in the frozen run metadata. The common
generation configuration was temperature `0`, maximum output length `2048`,
reasoning mode `none`, and no explicitly supplied sampling seed. The manifest
also records the single equivalent MiniMax requested-alias variation. It
deliberately omits provider endpoints, credentials, token accounting, and
separate reasoning captures.

## Scaling boundary

The automatic stages support batched inputs, resumable outputs, sharding, and
concurrent API calls. Their compute and API cost therefore scale
approximately with the number of rendered pages, detected structures, and
extracted rows. Human verification is the deliberate final quality gate and
the main curation bottleneck. Larger releases should be produced as versioned
review batches rather than by treating unreviewed extraction output as
benchmark ground truth.
