# Fully synthetic interface examples

Every record in this directory was invented solely to exercise the evaluator
interface. None is a benchmark item, chemical measurement, or model output.
The arbitrary labels `A0`, `A1`, `B0`, and `B1` have no chemical meaning.

Run the commands in the repository README from the repository root.

The MC file has two questions: one correct prediction and one incorrect
prediction, with mean relative score `(1.0 + 0.4) / 2 = 0.7`.

The FIB file has three questions. The predictions select a best observed
combination, a non-best observed combination, and a valid but unobserved
combination, respectively. Expected Hit is `2/3`, Score | Hit is
`(1.0 + 0.5) / 2 = 0.75`, and Strict Accuracy is `1/3`.
