# Frozen evaluation harness

Run a baseline:

```bash
python 04_eval/run_eval.py --repeats 3 --out 04_eval/results/baseline.json
```

Compare it after a change:

```bash
python 04_eval/compare.py 04_eval/results/baseline.json 04_eval/results/after.json
```

Scores are 0–100. Clustering combines inverse contamination and expected-count
agreement. Verification scores its genuine-minus-fabricated confidence gap and
gold-confidence MAE. Curriculum, evaluation, and recommendation use the
metrics named in the task once their gold labels exist. A missing label or
missing output contract produces `null`, never zero; composite weights are then
renormalized over available layers. Every metric records mean and population
standard deviation across independent repeats.

`compare.py` hard-fails for a different dataset SHA-256 or requested model ID.
It flags a change smaller than the baseline standard deviation as noise. A
comparison is also invalid if the model is not version-pinned, the tool/cache
state changes, the dataset is edited, or an agent's output contract changes
without updating the evaluator. The runner forces temperature 0 for injected
live clients and records tokens and elapsed time per repeat.
