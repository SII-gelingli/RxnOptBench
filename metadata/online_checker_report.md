# CROISSANT VALIDATION REPORT
================================================================================
## VALIDATION RESULTS
--------------------------------------------------------------------------------
Starting validation for file: croissant.json
### JSON Format Validation
✓
The file is valid JSON.
### Croissant Schema Validation
✓
The dataset passes Croissant validation.
### Responsible AI Metadata
✓
All required Responsible AI metadata fields are present.
### Records Generation Test
?
Record set 'fill_in_blank_all_blank' failed due to generation error:

```text
An error occurred during the streaming generation of the dataset, more specifically during the operation Download(transport/fill_in_blank_all_blank)

Traceback (most recent call last):
  File "/usr/local/lib/python3.13/site-packages/mlcroissant/_src/operation_graph/execute.py", line 135, in execute_operations_in_streaming
    result = operation.call(result)
  File "/usr/local/lib/python3.13/site-packages/mlcroissant/_src/operation_graph/operations/download.py", line 232, in call
    self._download_from_http(filepath)
    ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/app/validation.py", line 58, in _hf_aware_download
    return _orig_download_from_http(self, filepath)
  File "/usr/local/lib/python3.13/site-packages/mlcroissant/_src/operation_graph/operations/download.py", line 191, in _download_from_http
    response.raise_for_status()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/usr/local/lib/python3.13/site-packages/requests/models.py", line 1028, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/fill_in_blank_all_blank.tsv.gz

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/validation.py", line 145, in validate_records
    raise result  # re-raise actual error outside timeout
    ^^^^^^^^^^^^
  File "/app/validation.py", line 119, in try_generate_record
    for i, record in enumerate(record_collection):
                     ~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/mlcroissant/_src/datasets.py", line 166, in __iter__
    yield from execute_operations_in_streaming(
    ...<2 lines>...
    )
  File "/usr/local/lib/python3.13/site-packages/mlcroissant/_src/operation_graph/execute.py", line 137, in execute_operations_in_streaming
    raise GenerationError(
    ...<2 lines>...
    ) from e
mlcroissant._src.core.issues.GenerationError: An error occurred during the streaming generation of the dataset, more specifically during the operation Download(transport/fill_in_blank_all_blank)
```
## JSON-LD REFERENCE
================================================================================
```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "dct": "http://purl.org/dc/terms/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "prov": "http://www.w3.org/ns/prov#",
    "conformsTo": "dct:conformsTo",
    "citeAs": "cr:citeAs",
    "recordSet": "cr:recordSet",
    "field": "cr:field",
    "key": "cr:key",
    "dataType": {
      "@id": "cr:dataType",
      "@type": "@vocab"
    },
    "source": "cr:source",
    "fileObject": "cr:fileObject",
    "extract": "cr:extract",
    "fileProperty": "cr:fileProperty",
    "transform": "cr:transform",
    "regex": "cr:regex",
    "isLiveDataset": "cr:isLiveDataset",
    "arrayShape": "cr:arrayShape",
    "column": "cr:column",
    "containedIn": "cr:containedIn",
    "data": {
      "@id": "cr:data",
      "@type": "@json"
    },
    "description": {
      "@container": "@language"
    },
    "name": {
      "@container": "@language"
    },
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {
      "@id": "cr:examples",
      "@type": "@json"
    },
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "jsonPath": "cr:jsonPath",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "references": "cr:references",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "separator": "cr:separator",
    "subField": "cr:subField",
    "@base": "cr_base_iri/"
  },
  "@type": "sc:Dataset",
  "name": "RxnOptBench",
  "description": "RxnOptBench 0.2.6-target-policy contains 4,773 constructed reaction-condition QA records in eight test subsets, derived from reviewed literature screening tables. Experimental measurements are literature-derived; synthetic content is the constructed QA layer. Unverified dr/rr axes remain raw metadata and are excluded from scoring. Best labels use explicitly declared released observed support. The broader release separately supplies 777 source-relevant reviewed factual support records with explanatory fields omitted; those files are outside these eight RecordSets and do not establish exact reconstruction of the final questions. Each Croissant json_record field losslessly encodes a complete native question; decode it as JSON. Pair subset name with id when combining subsets.",
  "url": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final",
  "version": "0.2.6-target-policy",
  "inLanguage": "en",
  "conformsTo": "http://mlcommons.org/croissant/1.1",
  "isLiveDataset": false,
  "license": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/LICENSE-DATA.md",
  "creator": [
    {
      "@type": "sc:Person",
      "name": "Lingli Ge"
    },
    {
      "@type": "sc:Person",
      "name": "Yubin Wang"
    },
    {
      "@type": "sc:Person",
      "name": "Junyuan Gao"
    },
    {
      "@type": "sc:Person",
      "name": "Jiahe Song"
    },
    {
      "@type": "sc:Person",
      "name": "Jiaxing Sun"
    },
    {
      "@type": "sc:Person",
      "name": "Boyu Zhu"
    },
    {
      "@type": "sc:Person",
      "name": "Haote Yang"
    },
    {
      "@type": "sc:Person",
      "name": "Jingchao Wang"
    },
    {
      "@type": "sc:Person",
      "name": "Lixin Ma"
    },
    {
      "@type": "sc:Person",
      "name": "Jiang Wu"
    },
    {
      "@type": "sc:Person",
      "name": "Yuqiang Li"
    },
    {
      "@type": "sc:Person",
      "name": "Conghui He"
    }
  ],
  "keywords": [
    "chemistry",
    "reaction condition optimization",
    "large language models",
    "benchmark",
    "test-only"
  ],
  "conditionsOfAccess": "The authors' code uses MIT. Their data/QA contributions use the RxnOptBench Noncommercial Research License v1.0 to the extent of rights they hold. Upstream permissions remain as stated in UPSTREAM_LICENSE_NOTICES.md. Data access requires a Hugging Face account and acceptance of the research-use agreement; approval is automatic after acceptance. Metadata and code are publicly available. The loader uses standard cached Hugging Face login or HF_TOKEN to download immutable, hash-checked files, then gives them to the unmodified mlcroissant local-mapping API. The generic mlcroissant HTTP downloader does not automatically apply Hugging Face authentication.",
  "citation": {
    "@type": "sc:CreativeWork",
    "name": "RxnOptBench: Benchmarking LLMs for Reaction-Condition Optimization in Organic Methodology",
    "author": [
      {
        "@type": "sc:Person",
        "name": "Lingli Ge"
      },
      {
        "@type": "sc:Person",
        "name": "Yubin Wang"
      },
      {
        "@type": "sc:Person",
        "name": "Junyuan Gao"
      },
      {
        "@type": "sc:Person",
        "name": "Jiahe Song"
      },
      {
        "@type": "sc:Person",
        "name": "Jiaxing Sun"
      },
      {
        "@type": "sc:Person",
        "name": "Boyu Zhu"
      },
      {
        "@type": "sc:Person",
        "name": "Haote Yang"
      },
      {
        "@type": "sc:Person",
        "name": "Jingchao Wang"
      },
      {
        "@type": "sc:Person",
        "name": "Lixin Ma"
      },
      {
        "@type": "sc:Person",
        "name": "Jiang Wu"
      },
      {
        "@type": "sc:Person",
        "name": "Yuqiang Li"
      },
      {
        "@type": "sc:Person",
        "name": "Conghui He"
      }
    ],
    "datePublished": "2026"
  },
  "citeAs": "@inproceedings{ge2026rxnoptbench,\n  title={RxnOptBench: Benchmarking LLMs for Reaction-Condition Optimization in Organic Methodology},\n  author={Lingli Ge and Yubin Wang and Junyuan Gao and Jiahe Song and Jiaxing Sun and Boyu Zhu and Haote Yang and Jingchao Wang and Lixin Ma and Jiang Wu and Yuqiang Li and Conghui He},\n  booktitle={Advances in Neural Information Processing Systems},\n  year={2026}\n}",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "transport/fill_in_blank_all_blank",
      "name": "fill_in_blank_all_blank.tsv.gz",
      "description": "Deterministic gzip-compressed TSV transport; json_record is the unchanged original UTF-8 JSON line. Contains exactly the same questions as the frozen native JSON and JSONL.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/fill_in_blank_all_blank.tsv.gz",
      "encodingFormat": "text/tab-separated-values",
      "sha256": "16cc5cbca5e01f0339795ffcfc45e65aff0a43f3d098b3269663326038e33192",
      "contentSize": "254519 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "jsonl/fill_in_blank_all_blank",
      "name": "fill_in_blank_all_blank.jsonl",
      "description": "One complete original question per UTF-8 JSON line. Canonical Hugging Face transport format.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/jsonl/fill_in_blank_all_blank.jsonl",
      "encodingFormat": "application/jsonlines",
      "sha256": "d084aeefc0542767206b179ee4ce99c62c34ad10c283b1fdb81c33eeff69a009",
      "contentSize": "3442389 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "native-json/fill_in_blank_all_blank",
      "name": "fill_in_blank_all_blank.json",
      "description": "Equivalent native JSON array consumed by the deterministic evaluation scripts.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/original_json/fill_in_blank_all_blank.json",
      "encodingFormat": "application/json",
      "sha256": "d071543f2eec8c0c161fa781958ac011b9b0aff91260eb5f71f6953e63657c45",
      "contentSize": "6391286 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "transport/fill_in_blank_all_blank_reference",
      "name": "fill_in_blank_all_blank_reference.tsv.gz",
      "description": "Deterministic gzip-compressed TSV transport; json_record is the unchanged original UTF-8 JSON line. Contains exactly the same questions as the frozen native JSON and JSONL.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/fill_in_blank_all_blank_reference.tsv.gz",
      "encodingFormat": "text/tab-separated-values",
      "sha256": "8350326eb10935bc348faf2ece92f141a40ae1dd00be3958b15eacf7704d4de8",
      "contentSize": "186157 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "jsonl/fill_in_blank_all_blank_reference",
      "name": "fill_in_blank_all_blank_reference.jsonl",
      "description": "One complete original question per UTF-8 JSON line. Canonical Hugging Face transport format.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/jsonl/fill_in_blank_all_blank_reference.jsonl",
      "encodingFormat": "application/jsonlines",
      "sha256": "af6fa1afea6530baa31c5a077d4ec0d05ace552d592bec9cc959027e34b443ef",
      "contentSize": "2637941 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "native-json/fill_in_blank_all_blank_reference",
      "name": "fill_in_blank_all_blank_reference.json",
      "description": "Equivalent native JSON array consumed by the deterministic evaluation scripts.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/original_json/fill_in_blank_all_blank_reference.json",
      "encodingFormat": "application/json",
      "sha256": "5bb15b81d3888c4af88dce6ad539d3742924330603828ac0802ec72ab6a0f1cf",
      "contentSize": "5187920 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "transport/multiple_choice_all_varying",
      "name": "multiple_choice_all_varying.tsv.gz",
      "description": "Deterministic gzip-compressed TSV transport; json_record is the unchanged original UTF-8 JSON line. Contains exactly the same questions as the frozen native JSON and JSONL.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/multiple_choice_all_varying.tsv.gz",
      "encodingFormat": "text/tab-separated-values",
      "sha256": "390623aab545e599909c4d856f7725a022db694edbc44964ab8b4496f623f94f",
      "contentSize": "232275 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "jsonl/multiple_choice_all_varying",
      "name": "multiple_choice_all_varying.jsonl",
      "description": "One complete original question per UTF-8 JSON line. Canonical Hugging Face transport format.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/jsonl/multiple_choice_all_varying.jsonl",
      "encodingFormat": "application/jsonlines",
      "sha256": "038e5e39196f1e74c06501e8d4504c2ec06e28fb01097cf56d6eb0aad0c43186",
      "contentSize": "3269256 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "native-json/multiple_choice_all_varying",
      "name": "multiple_choice_all_varying.json",
      "description": "Equivalent native JSON array consumed by the deterministic evaluation scripts.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/original_json/multiple_choice_all_varying.json",
      "encodingFormat": "application/json",
      "sha256": "123f3defa47e834276ec5badff973c15e6248475d87ae396f250308687ac11aa",
      "contentSize": "5925881 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "transport/multiple_choice_all_varying_reference",
      "name": "multiple_choice_all_varying_reference.tsv.gz",
      "description": "Deterministic gzip-compressed TSV transport; json_record is the unchanged original UTF-8 JSON line. Contains exactly the same questions as the frozen native JSON and JSONL.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/multiple_choice_all_varying_reference.tsv.gz",
      "encodingFormat": "text/tab-separated-values",
      "sha256": "bcea74e5b4688df58608f9008342bcec91920e5f4e811bef63e3c9d6a9fa2a30",
      "contentSize": "171774 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "jsonl/multiple_choice_all_varying_reference",
      "name": "multiple_choice_all_varying_reference.jsonl",
      "description": "One complete original question per UTF-8 JSON line. Canonical Hugging Face transport format.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/jsonl/multiple_choice_all_varying_reference.jsonl",
      "encodingFormat": "application/jsonlines",
      "sha256": "22c85aaf5945acfe396f653bcaff168807b545f149ebecabef64cf0d0a35627f",
      "contentSize": "2569158 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "native-json/multiple_choice_all_varying_reference",
      "name": "multiple_choice_all_varying_reference.json",
      "description": "Equivalent native JSON array consumed by the deterministic evaluation scripts.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/original_json/multiple_choice_all_varying_reference.json",
      "encodingFormat": "application/json",
      "sha256": "21fd64a60a9b07f6bee65542074ef0f5d83d65f091fdc6a3ccb81b33d5862c93",
      "contentSize": "4976901 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "transport/multiple_choice_all_varying_reference_control",
      "name": "multiple_choice_all_varying_reference_control.tsv.gz",
      "description": "Deterministic gzip-compressed TSV transport; json_record is the unchanged original UTF-8 JSON line. Contains exactly the same questions as the frozen native JSON and JSONL.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/multiple_choice_all_varying_reference_control.tsv.gz",
      "encodingFormat": "text/tab-separated-values",
      "sha256": "5ff59a61bc75a354c9b85cdaff919fddf0435be294778de13e07ba640d1f7591",
      "contentSize": "136652 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "jsonl/multiple_choice_all_varying_reference_control",
      "name": "multiple_choice_all_varying_reference_control.jsonl",
      "description": "One complete original question per UTF-8 JSON line. Canonical Hugging Face transport format.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/jsonl/multiple_choice_all_varying_reference_control.jsonl",
      "encodingFormat": "application/jsonlines",
      "sha256": "55d2a0874b1d6ff67a903f3b21f2acfa847d21e7c54b868bc67d9a12d5763f9e",
      "contentSize": "1771823 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "native-json/multiple_choice_all_varying_reference_control",
      "name": "multiple_choice_all_varying_reference_control.json",
      "description": "Equivalent native JSON array consumed by the deterministic evaluation scripts.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/original_json/multiple_choice_all_varying_reference_control.json",
      "encodingFormat": "application/json",
      "sha256": "1e0bb5ffacb82f4703e81c4a912ccc4563c244a6132af0fc6c91350154bd43b6",
      "contentSize": "3180654 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "transport/multiple_choice_single_varying",
      "name": "multiple_choice_single_varying.tsv.gz",
      "description": "Deterministic gzip-compressed TSV transport; json_record is the unchanged original UTF-8 JSON line. Contains exactly the same questions as the frozen native JSON and JSONL.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/multiple_choice_single_varying.tsv.gz",
      "encodingFormat": "text/tab-separated-values",
      "sha256": "e8ba232fb5938dd9f22399d247afaa50a8fc52487026a6e7e6bf75bffde1ecc5",
      "contentSize": "208777 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "jsonl/multiple_choice_single_varying",
      "name": "multiple_choice_single_varying.jsonl",
      "description": "One complete original question per UTF-8 JSON line. Canonical Hugging Face transport format.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/jsonl/multiple_choice_single_varying.jsonl",
      "encodingFormat": "application/jsonlines",
      "sha256": "5a129250a3cea7de81fc373ad029193ece256e958bb226caa5e0573df40cc7d5",
      "contentSize": "3167763 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "native-json/multiple_choice_single_varying",
      "name": "multiple_choice_single_varying.json",
      "description": "Equivalent native JSON array consumed by the deterministic evaluation scripts.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/original_json/multiple_choice_single_varying.json",
      "encodingFormat": "application/json",
      "sha256": "53b5380245b14057be3a8fc497daf20ddbddd701de95ad83b9694cbdc71b0040",
      "contentSize": "5218859 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "transport/multiple_choice_single_varying_reference",
      "name": "multiple_choice_single_varying_reference.tsv.gz",
      "description": "Deterministic gzip-compressed TSV transport; json_record is the unchanged original UTF-8 JSON line. Contains exactly the same questions as the frozen native JSON and JSONL.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/multiple_choice_single_varying_reference.tsv.gz",
      "encodingFormat": "text/tab-separated-values",
      "sha256": "1e72954e687920a1727f2e67c584e9756f17cdf2f5f0163f8e0d348f12564afa",
      "contentSize": "230991 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "jsonl/multiple_choice_single_varying_reference",
      "name": "multiple_choice_single_varying_reference.jsonl",
      "description": "One complete original question per UTF-8 JSON line. Canonical Hugging Face transport format.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/jsonl/multiple_choice_single_varying_reference.jsonl",
      "encodingFormat": "application/jsonlines",
      "sha256": "5ef93bb9b772f4cdb222533a5da0ad55a37e9bd31d1fb0606411b25cfec1e2ae",
      "contentSize": "4586814 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "native-json/multiple_choice_single_varying_reference",
      "name": "multiple_choice_single_varying_reference.json",
      "description": "Equivalent native JSON array consumed by the deterministic evaluation scripts.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/original_json/multiple_choice_single_varying_reference.json",
      "encodingFormat": "application/json",
      "sha256": "af25f74003a05f6e6d261cf38a58a602b50612b4eb58ea37bdf57cc873b1a6e4",
      "contentSize": "8752299 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "transport/multiple_choice_single_varying_reference_control",
      "name": "multiple_choice_single_varying_reference_control.tsv.gz",
      "description": "Deterministic gzip-compressed TSV transport; json_record is the unchanged original UTF-8 JSON line. Contains exactly the same questions as the frozen native JSON and JSONL.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/croissant/multiple_choice_single_varying_reference_control.tsv.gz",
      "encodingFormat": "text/tab-separated-values",
      "sha256": "e67fa4fda738f07552823f714c68a407bbe18014e3bb8b77f240b013ee52f5db",
      "contentSize": "172220 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "jsonl/multiple_choice_single_varying_reference_control",
      "name": "multiple_choice_single_varying_reference_control.jsonl",
      "description": "One complete original question per UTF-8 JSON line. Canonical Hugging Face transport format.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/jsonl/multiple_choice_single_varying_reference_control.jsonl",
      "encodingFormat": "application/jsonlines",
      "sha256": "b1f6f6604b65db798508dac0fa26c7dfb11500e344d5cf7385a31d96a26f2e26",
      "contentSize": "2611070 B"
    },
    {
      "@type": "cr:FileObject",
      "@id": "native-json/multiple_choice_single_varying_reference_control",
      "name": "multiple_choice_single_varying_reference_control.json",
      "description": "Equivalent native JSON array consumed by the deterministic evaluation scripts.",
      "contentUrl": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/data/original_json/multiple_choice_single_varying_reference_control.json",
      "encodingFormat": "application/json",
      "sha256": "896e5b413647a3994c161c2859c8571194343556bd79681cc1d6a271abf24c31",
      "contentSize": "4280807 B"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "@id": "fill_in_blank_all_blank",
      "name": "fill_in_blank_all_blank",
      "description": "Versioned test subset; id is unique within this RecordSet. Parse json_record as JSON. Nested schema: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/record_schema.json#/$defs/fill_in_blank_all_blank.",
      "key": {
        "@id": "fill_in_blank_all_blank/id"
      },
      "field": [
        {
          "@type": "cr:Field",
          "@id": "fill_in_blank_all_blank/id",
          "name": "id",
          "description": "Frozen native question identifier; pair with subset name when combining RecordSets.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/fill_in_blank_all_blank"
            },
            "extract": {
              "column": "id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "fill_in_blank_all_blank/question_type",
          "name": "question_type",
          "description": "Native task-family label.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/fill_in_blank_all_blank"
            },
            "extract": {
              "column": "question_type"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "fill_in_blank_all_blank/parent_table_id",
          "name": "parent_table_id",
          "description": "Source-table grouping identifier from meta.parent_table_id; use it to avoid treating related questions as independent tables.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/fill_in_blank_all_blank"
            },
            "extract": {
              "column": "parent_table_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "fill_in_blank_all_blank/json_record",
          "name": "json_record",
          "description": "The complete unchanged UTF-8 JSON line, including id, question_type, input, answer, meta, and applicable candidate fields. JSON-decode this text to recover all nested objects and arrays; schema is in record_schema.json.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/fill_in_blank_all_blank"
            },
            "extract": {
              "column": "json_record"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "fill_in_blank_all_blank_reference",
      "name": "fill_in_blank_all_blank_reference",
      "description": "Versioned test subset; id is unique within this RecordSet. Parse json_record as JSON. Nested schema: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/record_schema.json#/$defs/fill_in_blank_all_blank_reference.",
      "key": {
        "@id": "fill_in_blank_all_blank_reference/id"
      },
      "field": [
        {
          "@type": "cr:Field",
          "@id": "fill_in_blank_all_blank_reference/id",
          "name": "id",
          "description": "Frozen native question identifier; pair with subset name when combining RecordSets.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/fill_in_blank_all_blank_reference"
            },
            "extract": {
              "column": "id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "fill_in_blank_all_blank_reference/question_type",
          "name": "question_type",
          "description": "Native task-family label.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/fill_in_blank_all_blank_reference"
            },
            "extract": {
              "column": "question_type"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "fill_in_blank_all_blank_reference/parent_table_id",
          "name": "parent_table_id",
          "description": "Source-table grouping identifier from meta.parent_table_id; use it to avoid treating related questions as independent tables.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/fill_in_blank_all_blank_reference"
            },
            "extract": {
              "column": "parent_table_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "fill_in_blank_all_blank_reference/json_record",
          "name": "json_record",
          "description": "The complete unchanged UTF-8 JSON line, including id, question_type, input, answer, meta, and applicable candidate fields. JSON-decode this text to recover all nested objects and arrays; schema is in record_schema.json.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/fill_in_blank_all_blank_reference"
            },
            "extract": {
              "column": "json_record"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "multiple_choice_all_varying",
      "name": "multiple_choice_all_varying",
      "description": "Versioned test subset; id is unique within this RecordSet. Parse json_record as JSON. Nested schema: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/record_schema.json#/$defs/multiple_choice_all_varying.",
      "key": {
        "@id": "multiple_choice_all_varying/id"
      },
      "field": [
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying/id",
          "name": "id",
          "description": "Frozen native question identifier; pair with subset name when combining RecordSets.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying"
            },
            "extract": {
              "column": "id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying/question_type",
          "name": "question_type",
          "description": "Native task-family label.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying"
            },
            "extract": {
              "column": "question_type"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying/parent_table_id",
          "name": "parent_table_id",
          "description": "Source-table grouping identifier from meta.parent_table_id; use it to avoid treating related questions as independent tables.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying"
            },
            "extract": {
              "column": "parent_table_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying/json_record",
          "name": "json_record",
          "description": "The complete unchanged UTF-8 JSON line, including id, question_type, input, answer, meta, and applicable candidate fields. JSON-decode this text to recover all nested objects and arrays; schema is in record_schema.json.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying"
            },
            "extract": {
              "column": "json_record"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "multiple_choice_all_varying_reference",
      "name": "multiple_choice_all_varying_reference",
      "description": "Versioned test subset; id is unique within this RecordSet. Parse json_record as JSON. Nested schema: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/record_schema.json#/$defs/multiple_choice_all_varying_reference.",
      "key": {
        "@id": "multiple_choice_all_varying_reference/id"
      },
      "field": [
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying_reference/id",
          "name": "id",
          "description": "Frozen native question identifier; pair with subset name when combining RecordSets.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying_reference"
            },
            "extract": {
              "column": "id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying_reference/question_type",
          "name": "question_type",
          "description": "Native task-family label.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying_reference"
            },
            "extract": {
              "column": "question_type"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying_reference/parent_table_id",
          "name": "parent_table_id",
          "description": "Source-table grouping identifier from meta.parent_table_id; use it to avoid treating related questions as independent tables.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying_reference"
            },
            "extract": {
              "column": "parent_table_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying_reference/json_record",
          "name": "json_record",
          "description": "The complete unchanged UTF-8 JSON line, including id, question_type, input, answer, meta, and applicable candidate fields. JSON-decode this text to recover all nested objects and arrays; schema is in record_schema.json.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying_reference"
            },
            "extract": {
              "column": "json_record"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "multiple_choice_all_varying_reference_control",
      "name": "multiple_choice_all_varying_reference_control",
      "description": "Versioned test subset; id is unique within this RecordSet. Parse json_record as JSON. Nested schema: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/record_schema.json#/$defs/multiple_choice_all_varying_reference_control.",
      "key": {
        "@id": "multiple_choice_all_varying_reference_control/id"
      },
      "field": [
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying_reference_control/id",
          "name": "id",
          "description": "Frozen native question identifier; pair with subset name when combining RecordSets.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying_reference_control"
            },
            "extract": {
              "column": "id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying_reference_control/question_type",
          "name": "question_type",
          "description": "Native task-family label.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying_reference_control"
            },
            "extract": {
              "column": "question_type"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying_reference_control/parent_table_id",
          "name": "parent_table_id",
          "description": "Source-table grouping identifier from meta.parent_table_id; use it to avoid treating related questions as independent tables.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying_reference_control"
            },
            "extract": {
              "column": "parent_table_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_all_varying_reference_control/json_record",
          "name": "json_record",
          "description": "The complete unchanged UTF-8 JSON line, including id, question_type, input, answer, meta, and applicable candidate fields. JSON-decode this text to recover all nested objects and arrays; schema is in record_schema.json.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_all_varying_reference_control"
            },
            "extract": {
              "column": "json_record"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "multiple_choice_single_varying",
      "name": "multiple_choice_single_varying",
      "description": "Versioned test subset; id is unique within this RecordSet. Parse json_record as JSON. Nested schema: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/record_schema.json#/$defs/multiple_choice_single_varying.",
      "key": {
        "@id": "multiple_choice_single_varying/id"
      },
      "field": [
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying/id",
          "name": "id",
          "description": "Frozen native question identifier; pair with subset name when combining RecordSets.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying"
            },
            "extract": {
              "column": "id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying/question_type",
          "name": "question_type",
          "description": "Native task-family label.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying"
            },
            "extract": {
              "column": "question_type"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying/parent_table_id",
          "name": "parent_table_id",
          "description": "Source-table grouping identifier from meta.parent_table_id; use it to avoid treating related questions as independent tables.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying"
            },
            "extract": {
              "column": "parent_table_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying/json_record",
          "name": "json_record",
          "description": "The complete unchanged UTF-8 JSON line, including id, question_type, input, answer, meta, and applicable candidate fields. JSON-decode this text to recover all nested objects and arrays; schema is in record_schema.json.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying"
            },
            "extract": {
              "column": "json_record"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "multiple_choice_single_varying_reference",
      "name": "multiple_choice_single_varying_reference",
      "description": "Versioned test subset; id is unique within this RecordSet. Parse json_record as JSON. Nested schema: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/record_schema.json#/$defs/multiple_choice_single_varying_reference.",
      "key": {
        "@id": "multiple_choice_single_varying_reference/id"
      },
      "field": [
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying_reference/id",
          "name": "id",
          "description": "Frozen native question identifier; pair with subset name when combining RecordSets.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying_reference"
            },
            "extract": {
              "column": "id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying_reference/question_type",
          "name": "question_type",
          "description": "Native task-family label.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying_reference"
            },
            "extract": {
              "column": "question_type"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying_reference/parent_table_id",
          "name": "parent_table_id",
          "description": "Source-table grouping identifier from meta.parent_table_id; use it to avoid treating related questions as independent tables.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying_reference"
            },
            "extract": {
              "column": "parent_table_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying_reference/json_record",
          "name": "json_record",
          "description": "The complete unchanged UTF-8 JSON line, including id, question_type, input, answer, meta, and applicable candidate fields. JSON-decode this text to recover all nested objects and arrays; schema is in record_schema.json.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying_reference"
            },
            "extract": {
              "column": "json_record"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "multiple_choice_single_varying_reference_control",
      "name": "multiple_choice_single_varying_reference_control",
      "description": "Versioned test subset; id is unique within this RecordSet. Parse json_record as JSON. Nested schema: https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/resolve/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/record_schema.json#/$defs/multiple_choice_single_varying_reference_control.",
      "key": {
        "@id": "multiple_choice_single_varying_reference_control/id"
      },
      "field": [
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying_reference_control/id",
          "name": "id",
          "description": "Frozen native question identifier; pair with subset name when combining RecordSets.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying_reference_control"
            },
            "extract": {
              "column": "id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying_reference_control/question_type",
          "name": "question_type",
          "description": "Native task-family label.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying_reference_control"
            },
            "extract": {
              "column": "question_type"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying_reference_control/parent_table_id",
          "name": "parent_table_id",
          "description": "Source-table grouping identifier from meta.parent_table_id; use it to avoid treating related questions as independent tables.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying_reference_control"
            },
            "extract": {
              "column": "parent_table_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "multiple_choice_single_varying_reference_control/json_record",
          "name": "json_record",
          "description": "The complete unchanged UTF-8 JSON line, including id, question_type, input, answer, meta, and applicable candidate fields. JSON-decode this text to recover all nested objects and arrays; schema is in record_schema.json.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "transport/multiple_choice_single_varying_reference_control"
            },
            "extract": {
              "column": "json_record"
            }
          }
        }
      ]
    }
  ],
  "rai:dataLimitations": [
    "The benchmark measures selection among candidates derived from disclosed literature screening tables. A best answer is best within the explicitly declared released observational support under the utility, not a global optimum over chemical space. No direct wet-lab guidance, autonomous experimentation, safety/cost assessment, or open-ended synthesis ability is validated.",
    "This version retains 4,773 questions in eight subsets and 552 parent tables. Their union represents 4,768 distinct released observed rows, whereas the paper describes 5,451 source-table entries; 298 parent tables have all reported source rows represented and 254 do not. global_best_scope is all_released_observations_for_parent_table. A best label does not certify the optimum over unavailable original rows. Individual MC and FIB questions expose different observed supports.",
    "The fixed-target dr/rr policy requires source-identified components, a target fixed across the table, and a human-verified component-to-target mapping. No such independent mapping ledger was found in the current workspace; the author confirmed there are no additional records to use. All 65 observed parent-table/axis combinations therefore have status exclude_unverified. Their original parsed dr/rr observations remain in raw_observed_scores or raw_observed_score, while dr/rr values in the score-bearing structure are null and do not affect utility or gold labels. No major:minor normalization is used for these excluded axes. Other existing score factors retain their declared treatment. This is a conservative exclusion policy, not a claim that target mappings have been verified.",
    "Fill-in-the-blank candidate values are individually source-derived, but their Cartesian product contains unobserved joint tuples. A structurally valid tuple that does not match a released observed row is a non-Hit outside measured support, not a chemically invalid reaction or a measured failure. Score conditioned on Hit excludes those unobserved tuples.",
    "Question inputs, candidate indices, and reference/control contexts are preserved for rescoring archived predictions. Gold labels and relative scores may change under the corrected policy. Some reference rows become best after exclusion, and a small number of question variants contain ties across all options; unchanged contexts are not evidence that the corrected optimum was withheld. Reference rows and candidate rows are disjoint by source-row identity, which does not guarantee distinct full condition tuples when different source rows repeat the same conditions. The local audit reports affected items separately. Historical generations were not rerun under altered prompts.",
    "Paired reference/control subsets match candidate sets, but their released option order differs for many items. Historical provider-input ordering has not been independently verified from complete prompt-order logs. These paired comparisons therefore do not isolate a causal effect of adding reference outcomes; candidate-set matching must not be described as complete prompt identity apart from the reference field.",
    "Deterministic rescoring of retained model predictions does not reproduce historical hosted-model generations or establish invariance across endpoint versions, backends, decoding settings, or time. Submitted historical aggregates and corrected-policy results are separately versioned and must not be interchanged.",
    "Sources are recent organic-methodology literature across 16 selected journals, including publisher-confirmed first publications in 2026. They must not be described as exclusively first published in 2025. A Crossref year can reflect a later issue date rather than first online publication; not all records with a 2026 registry date are asserted to be 2026 first publications. Expert review compared accepted chemical records with source material. Source exposure and model training cutoffs are not fully observable; absence of training contamination has not been established.",
    "This descriptor covers eight evaluation subsets only. The broader release separately inventories available code, six task prompts, validators, archived model outputs and 777 source-relevant reviewed factual support records limited to the final 232 source groups. Explanatory fields are omitted from those support exports. An earlier content-association bridge links 416 original complete reviewed files to final tables; its hashes refer to the original internal files, not current public exports. It is historical support, not a new validation of the exports. Twelve final source groups lack a matching reviewed record and missing mappings are not invented. Unreleased molecular-recognition components/model-specific postprocessing, weights, publisher PDFs and source images are excluded. The 4,473-question construction result belongs to a historical full 884-record snapshot; it is not a rerun of the 777-record export. Exact final-question or raw-PDF-to-benchmark regeneration is not established.",
    "meta.doi contains frozen source-identifier strings. The MC-all file has 552 parent tables and 236 case-sensitive raw source strings; four case-variant groups reduce this to 232 case-insensitive source groups. A separate public bibliographic audit matched their 232 DOI identifiers to Crossref. Frozen strings remain unchanged. Registry matching establishes identity, not permission to redistribute source material.",
    "Public source-category and policy evidence is retained for 232 final-source DOIs across six publisher families, including separate article/SI attribution where available. The 777 reviewed support records are factual-field exports with explanatory fields omitted. These scope choices do not claim individual permission for every source, complete historical contract review, or project-wide rights clearance. Historical acquisition/subscription details and API hosting remain unknown; some source-file correspondence, SI and third-party scope are unresolved. Missing an open-license label is not treated as an automatic prohibition on releasing a factual measurement. Source terms and any upstream protectable expression remain distinct from author-owned QA contributions; the custom agreement does not restrict permissions independently granted by upstream licenses."
  ],
  "rai:dataBiases": [
    "Methodology papers disclose curated evidence supporting a narrative rather than complete optimization histories. Successful, interesting, or publishable transformations and selected comparison rows may be overrepresented. This benchmark cannot recover undisclosed experiments.",
    "The selected journals, recent-literature sampling frame, extraction feasibility, and human curation produce uneven coverage of reaction classes, substrates, catalysts, condition fields, and table formats. Tables that could not be structured reliably or lacked usable reaction/outcome information are underrepresented.",
    "Reference/control contexts were constructed using the historical utility and retained unchanged when labels were corrected. Their context-selection distribution need not match a chemist's available information. Corrected-optimal reference rows, repeated condition tuples across distinct source rows, all-option ties, and unmatched candidate orders are reported as limitations rather than evidence of a causal reference benefit.",
    "The aggregate utility prioritizes reported yield and retained selectivity factors. Missing factors, source reporting conventions, and the conservative exclusion of unverified dr/rr axes can affect rankings. It does not encode every target-isomer preference, safety constraint, cost, or chemical objective."
  ],
  "rai:personalSensitiveInformation": "Test records describe chemical reactions, conditions, outcomes, and source identifiers. No personal, demographic, medical, political, or other human-sensitive attributes are intentionally collected as benchmark features. Public author information appears in documentation. The paper reports aggregate information about six paid chemistry interns; individual annotator identities are not test-set attributes.",
  "rai:dataUseCases": [
    "Offline comparison of reaction-condition recommendation systems within a fixed, source-derived candidate space. Report MC Accuracy/relative Score and distinguish FIB observed-row Hit, Strict Accuracy, and Score conditioned on Hit. State dataset and scoring-policy versions and use subset plus id as the question key.",
    "The evaluated construct is comparative selection among disclosed conditions for the same reaction, based on source-reviewed rows and reported measurements. It is conditional on released support and the declared utility, not unrestricted chemical optimization or laboratory readiness.",
    "All eight subsets are test-only. Training or fine-tuning on these records compromises evaluation on them; no disjoint training/validation split or established fine-tuning benefit is supplied. Direct experimental deployment and outcome inference for unobserved tuples have not been validated."
  ],
  "rai:dataSocialImpact": "Structured aggregation supports reproducible comparison while reducing effort needed to retrieve and reuse chemical information, creating misuse risk and risk of confusing offline scores with experimental readiness. Provenance, disclosed-row limits, target policy and sampling limits are documented. The research-use agreement and automatic approval add accountability and friction rather than certifying safe use or source rights. Public source-policy evidence and a factual-field export boundary are supplied; historical acquisition/API hosting and remaining source/SI scope are not asserted to be fully resolved.",
  "rai:hasSyntheticData": true,
  "rai:dataCollection": "Sources are recent organic-methodology papers and supporting information from 16 selected journals, including publisher-confirmed first publications in 2026. Available code has local PDF/table-processing stages and API-dependent stages. Historical download dates, institutional subscriptions and whether those API services were locally hosted or external are unknown; code capability does not prove historical transmission. Four semi-automated stages detect and structure screening tables, followed by final human verification against source material before question construction. Public source-category/license evidence is inventoried separately, without asserting that all acquisition, processing or redistribution conditions are established. Experimental outcomes are literature-derived, not simulated or newly measured. hasSyntheticData is true only for the programmatically constructed QA layer, including masking, candidate arrangement, context construction and computed annotations.",
  "rai:dataCollectionType": [
    "Document analysis",
    "Secondary Data analysis",
    "Manual Human Curator",
    "Software Collection"
  ],
  "rai:dataAnnotationProtocol": "Four master's-level and two doctoral-level paid chemistry interns compared extracted tables with source PDFs, corrected fields, and accepted, rejected, or skipped tables using an internal review interface. The paper reports RMB 300/day for master's-level interns and RMB 400/day for doctoral-level interns. No independent multi-label annotation or inter-annotator-agreement statistic is claimed. General acceptance of a table is distinct from an independently documented dr/rr target-component mapping. No such mapping record was found; unverified dr/rr axes are excluded from scoring and preserved as raw metadata.",
  "rai:dataAnnotationPlatform": "Internal per-table expert-review interface described in the paper. Available tooling and 777 source-relevant reviewed factual support records are inventoried separately from the eight evaluation RecordSets. Explanatory fields are omitted from public support exports. Independent dr/rr target-component verification is not established.",
  "rai:dataManipulationProtocol": "The 0.2.6-target-policy version preserves question inputs, candidates, indices, source strings, and reference/control contexts from the compatible historical evaluation snapshot. It retains all provided raw observations, nulls unverified score-bearing dr/rr fields, and recomputes effective/relative scores and gold/tie sets under a consistent released-support policy. No measured yields or selectivities are invented. Separate versioned policy, relabel, per-question change, and historical-scoring manifests document the transformation. JSONL and native JSON represent equivalent records; lossless TSV.gz exists solely for standard Croissant-loader compatibility. Archived index-restored predictions are rescored against unchanged retained question inputs. Derived response artifacts and post-hoc index restoration do not independently verify original provider prompt ordering or reproduce provider calls. The separate support export contains 777 source-relevant reviewed factual records with explanatory fields omitted without changing the 4,773 evaluation questions or their scores. An earlier 884-record full-snapshot construction experiment yielded 4,473 questions; no reduced-snapshot rerun or exact review-to-final regeneration is asserted.",
  "prov:wasDerivedFrom": [
    {
      "@type": "prov:Entity",
      "@id": "https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final/tree/b80dfbc61d338961d46034df67aca36f5b1bcf17/metadata/source_rights",
      "sc:name": "Literature-source attribution and public policy evidence",
      "sc:description": "Immutable source-attribution/policy evidence accompanies the factual measurements and source/table/row back-pointers. It is not a blanket individual-permission or historical-contract-clearance claim. Publisher PDFs and source images are excluded."
    }
  ],
  "prov:wasGeneratedBy": [
    {
      "@type": "prov:Activity",
      "sc:name": "Literature selection, extraction, and final human verification",
      "sc:description": "Literature acquisition, extraction/normalization, structural checks and final expert review precede question construction. Available stages include local processing and API calls; historical hosting is unknown. The paper describes 911 candidate tables, 552 retained tables and 5,451 source-table entries; the evaluation-question union represents 4,768 distinct observed rows. Public source-category evidence is inventoried; full historical route and actual material-scope clearance is not asserted."
    },
    {
      "@type": "prov:Activity",
      "sc:name": "Constructed QA layer and fixed-target inclusion policy",
      "sc:description": "Programmatic construction forms MC/FIB questions and reference/control variants. No independent target-component ledger was available; 65 observed dr/rr table-axis combinations are excluded uniformly from scoring while their raw observations remain. Best/relative values and gold/tie sets are recomputed within declared released support. Existing question inputs are retained for honest archived-output rescoring."
    },
    {
      "@type": "prov:Activity",
      "sc:name": "Versioned Croissant packaging",
      "sc:description": "Package the existing eight lossless TSV.gz transports, native/JSONL hashes and nested schemas against an immutable published data revision. The 777 reviewed factual support exports are separately manifested and omit explanatory fields. Final QA data and scores remain unchanged by support-export/publication packaging. Validation results are recorded separately; metadata generation itself does not establish remote loading or checker approval."
    }
  ],
  "datePublished": "2026-11-01T14:07:25.449146Z"
}
```