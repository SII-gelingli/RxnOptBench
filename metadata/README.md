RxnOptBench Croissant metadata and loading

Dataset: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final
Code: https://github.com/SII-gelingli/RxnOptBench
Code license: MIT. Data license: RxnOptBench Noncommercial Research License v1.0.
The data license and lightweight research-use agreement apply to dataset records;
MIT code licensing does not make benchmark records unrestricted.

croissant.json is the complete core + Responsible AI metadata, not a patch.
It describes the frozen 0.2.5-eval-only release: eight subsets, 4,773 questions.
It pins data URLs to an immutable commit and supplies true SHA-256 checksums.
Record IDs are unique within each subset; use (subset, id) when combining subsets.

The eight derived TSV.gz transport files expose four fields: id, question_type,
parent_table_id, and json_record. json_record preserves the complete original
JSONL line, including all nested inputs, candidates, answers, and provenance.
The transport is only for Croissant loader compatibility. Native JSON and JSONL
remain the canonical frozen data consumed by the evaluation scripts. The full
nested structural schemas for the eight subsets are in record_schema.json.

Access
1. Open the dataset page, log in, and accept the noncommercial research-use
   agreement. Access is approved automatically after acceptance.
2. Install dependencies with: python -m pip install -r requirements.txt
3. Log in using the official Hugging Face CLI: hf auth login
   A standard cached login or HF_TOKEN is used; never put a token in metadata.
4. Download croissant.json beside load_croissant.py and run:
   python load_croissant.py croissant.json --subset multiple_choice_all_varying
   This prints a record count without printing data or credentials.

Programmatic use
  from load_croissant import load_dataset, records
  dataset = load_dataset("croissant.json")
  for question in records(dataset, "multiple_choice_all_varying"):
      # question is the complete original nested dict
      pass

The wrapper downloads the eight pinned transports with huggingface_hub, verifies
their hashes, then uses the reference mlcroissant loader's standard mapping API.
The generic mlcroissant HTTP downloader does not itself read HF_TOKEN. Anonymous
downloads of gated records require agreement/login. Public metadata remains
available without acceptance and contains no actual question records or tokens.

Validation
The official conference-linked checker tests JSON syntax, Croissant schema, RAI
completeness, and optionally record loading. Its anonymous record-loading test
cannot bypass the agreement. Separate authenticated validation checks all 24
distribution hashes, loads all 4,773 records through the unmodified reference
loader, and verifies equality with native JSON and JSONL and nested JSON Schema.
These checks verify metadata and transport integrity, not scientific correctness
of every label, upstream publisher permissions, or conference approval of gating.

Official references
https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting
https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker
https://docs.mlcommons.org/croissant/docs/croissant-spec.html
