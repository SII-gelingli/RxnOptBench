# RxnOptBench code

This directory contains the benchmark builders and an inference runner for
OpenAI-compatible chat-completions endpoints. Paths are resolved from the
code-repository root; no machine-specific filesystem layout is
required.

The retained extraction code documents the released pipeline boundary, but it
does not include the unreleased OCSR inference or model-specific post-processing
implementation. `pipeline/stage8_align_labels.py` consumes externally supplied
OCSR JSON and copies its SMILES fields without expanding or repairing them.

## Inference

`inference_open_source.py` works with local servers such as vLLM and with other
OpenAI-compatible providers. Real datasets are downloaded separately into an authorized HF artifact root;
prompt templates are under this code repository's `prompts/`. Pass `--data-dir`
for the downloaded native JSON directory. New inference calls your supplied
endpoint; it is not needed for archived-output score reproduction.

For a local endpoint:

```bash
python code/inference_open_source.py \
  --task-type mc_all \
  --data-dir ../RxnOptBench-data/data/original_json \
  --setting classic \
  --model auto \
  --base-url http://localhost:8008/v1 \
  --temperature 0 \
  --max-tokens 2048 \
  --reasoning-mode none \
  --output outputs/mc_all_predictions.json
```

For an OpenAI-compatible endpoint that does not accept vLLM-specific request
extensions, add `--disable-vllm-extensions`. The endpoint and credential can
also be supplied through `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Use
`--testset` or `--prompt-dir` to override the bundled paths, and `--resume` to
continue an interrupted run.

The persisted inference artifacts are deliberately limited to:

- the parsed prediction mapping at the exact `--output` path;
- `<output_stem>_raw.json`, containing raw answer text;
- `<output_stem>_parse_methods.json`, recording how each answer was parsed.

Reasoning fields, token accounting, parsing summaries, and endpoint metadata
are not written to result JSON files. Operational details may still appear in
the local run log.

## Dataset builders

`build_multiple_choice.py` and `build_fill_in_blank.py` retain the submitted
2026-05-02 V6 construction and scoring semantics. They expect the intermediate
table JSON files produced by the extraction pipeline; those inputs are distinct
from the ready-to-evaluate files in `data/original_json/`.

```bash
python code/build_multiple_choice.py \
  --article-dir /path/to/pipeline_output \
  --output-dir build_outputs

python code/build_fill_in_blank.py \
  --article-dir /path/to/pipeline_output \
  --output-dir build_outputs
```

Optional companion paths are available through `--metadata-dir` and
`--sioc-mapping-csv`. The same values can be set with
`RXNOPTBENCH_ARTICLE_DIR`, `RXNOPTBENCH_METADATA_DIR`,
`RXNOPTBENCH_SIOC_MAPPING_CSV`, and `RXNOPTBENCH_BUILD_OUTPUT_DIR`.

Each builder creates a timestamped directory and copies the corresponding
evaluation script plus the shared `evaluation/common_v6.py` snapshot into that
directory.
