RxnOptBench 0.2.6-target-policy Croissant metadata

Dataset: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final
Immutable data revision: b80dfbc61d338961d46034df67aca36f5b1bcf17
Publication UTC: 2026-11-01T14:07:25.449146Z

croissant.json describes eight evaluation RecordSets and 4,773 questions. Its 24
distribution hashes match the retained native JSON, JSONL and lossless TSV.gz.
record_schema.json documents the complete nested records. The json_record Text
field must be JSON-decoded; identifiers are unique within a subset.

The broader release separately supplies 777 reviewed factual support exports
from 220 source groups within the final 232-source scope. Explanatory fields are
omitted; retained structured values and final QA/scoring are unchanged. Their
separate export manifest documents the transformation. They are support records,
not exact reconstruction of the 4,773 questions. The historical 884-record ->
4,473-question construction experiment does not describe a rerun of this export.

Code uses MIT. Author-owned data/QA contributions use the RxnOptBench
Noncommercial Research License v1.0 to the extent of rights held. See upstream
notices for source-specific terms; independently granted upstream permissions
are not reduced. Source-policy evidence is supplied, without claiming every
source has individual permission or historical contracts have all been cleared.

Load after accepting the HF research-use agreement and automatic approval:
  python -m pip install -r metadata/requirements.txt
  hf auth login
  python metadata/load_croissant.py metadata/croissant.json --subset multiple_choice_all_varying

The wrapper uses standard authorized HF downloads, checks immutable hashes, and
passes a local mapping to unmodified mlcroissant. It never prints tokens or data.
Optional offline use: add --transport-dir PATH_TO_DATA/data/croissant.

Reproduce the complete authenticated metadata check from a code checkout with
the matching data checkout in a sibling RxnOptBench-data directory:
  python metadata/validate_croissant.py --metadata metadata/croissant.json --mode authenticated --all-distributions --stage ../RxnOptBench-data
The portable validator has no dependency on private packaging/build scripts.

Metadata generation is separate from verification. Read the accompanying final
validation report for actual authenticated loading/schema results. Source rights,
scientific correctness and conference acceptance of the gate are not established
by structural metadata validation. No model call or rescoring is performed.
