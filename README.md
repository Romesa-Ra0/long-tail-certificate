Everything runs on a laptop CPU in a few minutes. No dataset is downloaded and no
GPU is required.

## Running the experiments

Keep all files in one directory and run them in order. `step1_train.py` must run
first, since the remaining scripts load the model it saves.

| Script | Purpose |
|---|---|
| `step1_train.py` | Trains the model and saves it |
| `step2_longtail.py` | Per-class accuracy before compression |
| `step3_prune.py` | Accuracy against sparsity, for head, tail and overall |
| `step4_margins.py` | Per-class margin, computed with and without labels |
| `step5_predict.py` | Correlation between margin and retained accuracy, three seeds |
| `step6_certificate.py` | Behaviour of the worst-case bound |
| `algo_v2.py` | Comparison of the three certification criteria |

Numbers on a different machine will differ in the third decimal place; the
pattern should be identical.

## Reproducibility

All random seeds are fixed in `common.py`. `step5_predict.py` reports each seed
separately as well as the pooled result, since a single seed is not evidence.
