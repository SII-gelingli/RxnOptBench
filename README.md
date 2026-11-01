# RxnOptBench

**RxnOptBench: Benchmarking LLMs for Reaction-Condition Optimization in Organic Methodology**

Lingli Ge, Yubin Wang, Junyuan Gao, Jiahe Song, Jiaxing Sun, Boyu Zhu,
Haote Yang, Jingchao Wang, Lixin Ma, Jiang Wu, Yuqiang Li, and Conghui He.

Shanghai Jiao Tong University · Shanghai Innovation Institute ·
Shanghai Artificial Intelligence Laboratory

RxnOptBench evaluates the selection of reaction conditions from alternatives
reported in organic-methodology screening tables. The benchmark contains
4,773 questions derived from 552 parent tables in 232 papers, organized into
eight multiple-choice (MC) and fill-in-the-blank (FIB) subsets. Its paired
questions evaluate condition selection with and without experimental
precedents.

This repository contains the deterministic evaluation scripts and fully
synthetic examples. The benchmark records and their prediction-format
examples are distributed separately through the
[Hugging Face dataset repository](https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final),
for non-commercial research use under its separate dataset terms and access
agreement. The code, documentation, and synthetic examples in this repository
are available under the [MIT License](LICENSE). See
[LICENSE_SCOPE.md](LICENSE_SCOPE.md) for the code/data boundary.

## Contents and scope

```text
evaluation/           Unmodified standalone MC and FIB evaluators
examples/synthetic/  Invented inputs and predictions for a runnable demo
CITATION.cff         Author list and repository citation metadata
```

The evaluators require Python 3 and use only its standard library. There is
no package installation step for evaluation.

The release provides the fixed, human-reviewed test set separately from
this code repository. Extraction, benchmark construction, annotation tools,
prompt templates, inference runners, archived model responses and predictions,
and publisher PDFs or images are outside the release. It supports scoring
supplied predictions; it does not reconstruct the data from source PDFs or
replay the paper's historical model generations.

## Run without downloading the benchmark

From the repository root:

```bash
python3 evaluation/evaluate_multiple_choice.py \
  --dataset examples/synthetic/multiple_choice.json \
  --prediction examples/synthetic/prediction_multiple_choice.json \
  --detail

python3 evaluation/evaluate_fill_in_blank.py \
  --dataset examples/synthetic/fill_in_blank.json \
  --prediction examples/synthetic/prediction_fill_in_blank.json \
  --detail
```

All toy records, labels, scores, and predictions were invented to demonstrate
the file interface. They are not chemical measurements, benchmark items, or
outputs from evaluated models. The MC demo produces Accuracy `0.5` and Score
`0.7`. The FIB demo produces Hit `2/3`, Score | Hit `0.75`, and Strict Accuracy
`1/3`. These values describe only the toy files.

## Download the benchmark

1. Open the [dataset page](https://huggingface.co/datasets/SII-gelingli/RxnOptBench-final)
   while signed in to your Hugging Face account, read its terms, and complete
   the access agreement. Download is available after access is granted.
2. With the official Hugging Face CLI available, run `hf auth login` locally
   using a token for that same authorized account. Do not place tokens in
   prediction files, source files, or repository commits.
3. Download the native JSON files consumed by these evaluators:

```bash
hf download SII-gelingli/RxnOptBench-final \
  --repo-type dataset \
  --include "data/original_json/*" \
  --local-dir .
```

For CLI setup, see the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).
The dataset's own license and access agreement govern the downloaded records;
they are not included in this code repository's license.

The native files are:

| File under `data/original_json/` | Questions |
| --- | ---: |
| `multiple_choice_single_varying.json` | 1,039 |
| `multiple_choice_all_varying.json` | 552 |
| `multiple_choice_single_varying_reference.json` | 850 |
| `multiple_choice_all_varying_reference.json` | 310 |
| `multiple_choice_single_varying_reference_control.json` | 850 |
| `multiple_choice_all_varying_reference_control.json` | 310 |
| `fill_in_blank_all_blank.json` | 552 |
| `fill_in_blank_all_blank_reference.json` | 310 |

The `reference_control` subsets are the paired no-precedents controls of
their corresponding `reference` subsets.

## Predict and evaluate

Write MC predictions as a JSON object mapping every question ID in the
chosen subset to one **0-based** option index:

```json
{
  "question_id": 0
}
```

Write FIB predictions as a JSON object mapping every question ID to a mapping
of its `query_keys` to **0-based** indices in
`input.candidate_values[condition_key]`:

```json
{
  "question_id": {
    "catalysts": 0,
    "solvents": 1
  }
}
```

The IDs and keys above illustrate the format. Use the IDs, keys, and candidate
orders from the downloaded release. Do not mix candidate indices or answer
keys from an earlier dataset version.

```bash
python3 evaluation/evaluate_multiple_choice.py \
  --dataset data/original_json/multiple_choice_single_varying.json \
  --prediction my_mc_predictions.json

python3 evaluation/evaluate_fill_in_blank.py \
  --dataset data/original_json/fill_in_blank_all_blank.json \
  --prediction my_fib_predictions.json
```

The examples distributed with the dataset demonstrate the prediction schema;
they are not evidence of any model's benchmark performance.

## Metrics and interpretation

The evaluators read the release's precomputed labels and scores. They do not
infer chemical validity, recompute selectivity from raw observations, or
search for new experimental conditions.

For MC, `exact_match_accuracy` is the fraction of evaluated predictions in
the stored optimal-answer set; `avg_relative_score` is the mean stored
relative score of the selected options. An invalid index scores zero.
`avg_yield_ratio` is an additional yield-only diagnostic, not the headline
yield-and-selectivity Score.

For FIB, `hit_rate` measures matches to observed joint condition rows. A
valid in-range combination absent from the observed rows is a non-Hit; it
is not labeled chemically invalid. `avg_relative_score` is **Score | Hit**,
the mean relative score restricted to Hits. The printed
`best_hit_rate_among_scored` also conditions on Hits. The paper's
**Strict Accuracy** is `best_hit_count / num_samples`, equivalently
`hit_rate * best_hit_rate_among_scored`; it is not the latter field alone.

**Coverage matters:** both scripts evaluate only question IDs present in
both the dataset and the predictions. Missing IDs are warned about but
excluded from the denominator; extra IDs are ignored. Submit a prediction
for every item and report `num_missing = 0` when comparing against full-set
results. Run each subset separately and preserve its identity in reports.

## Frozen labels and known limitations

The paper's main tables retain a historical evaluation snapshot and its
frozen labels. The archived utility combines reported yield with selectivity
factors. Missing selectivity axes contribute a neutral factor of one,
omitting those axes rather than establishing perfect selectivity.

In particular, archived `dr` and `rr` parsing must not be interpreted as
selectivity for one fixed target isomer. Normalizing each row as major:minor
can discard isomer identity, and numeric compound labels in ratio fields
require careful parsing. A complete source-verified, target-aware
re-adjudication of every archived label has not been established. The
camera-ready appendix discloses this limitation, a separate sensitivity
ablation removing all `dr/rr` factors, and a parser-maintenance audit. That
ablation is not an upper bound on the effect of target-aware correction.

The evaluator in this repository uses the scores supplied with the downloaded
release; it does not silently apply those ablations or relabel the data.
Fixed-data rescoring is deterministic, while new calls to hosted models may
produce different answers. The release also retains the documented
order-dependent FIB comparator as a limitation of its existing labels.

## Citation and support

The author list and paper title are recorded in [CITATION.cff](CITATION.cff).
No DOI or arXiv identifier is asserted here while final publication metadata
is pending. Please report the dataset version or revision together with
the evaluation code revision in any comparison.

For issues with these scripts or metadata, open an issue in
[this repository](https://github.com/SII-gelingli/RxnOptBench/issues).

## License

The evaluation code, documentation, and synthetic examples are licensed under
the [MIT License](LICENSE), copyright 2026 RxnOptBench authors. The separately
hosted benchmark dataset is limited to non-commercial research use under its
own terms and access agreement. The MIT license does not relicense those
records or any third-party source materials. See
[LICENSE_SCOPE.md](LICENSE_SCOPE.md).
