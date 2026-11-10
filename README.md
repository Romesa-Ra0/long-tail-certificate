# Certifying Compression-Induced Representation Drift

Can we tell that model compression has damaged a rare-class capability, without having any labels for that class?

Where labels exist, the cost of compression is easy to measure. In a zero-shot setting it is not, because no labelled examples of the affected classes are available. This repository studies whether the geometry of the embedding space reveals the loss instead.

## Setting

A prototype-similarity classifier is used, with the same decision rule as a vision-language model. An encoder maps inputs to an embedding, and the predicted class is the fixed prototype with the highest cosine similarity. Training data is long-tailed, from 2,000 examples for the most frequent class down to 12 for the rarest. Compression is post-training magnitude pruning, applied without fine-tuning.

The data is synthetic and no pretrained vision-language model is involved. These experiments test assumptions. They are not a clinical result.

## Findings

**Compression hides its cost in the tail.** In this synthetic experiment, overall accuracy stayed close to that of the dense model while the rarest classes lost accuracy after pruning. At the highest sparsity at which overall accuracy had fallen by no more than five points, the four most frequent classes lost no accuracy, while the four rarest classes lost up to 11.4 percentage points.

**Margin tracks class frequency.** The rank correlation between the number of training examples per class and its decision margin is 0.97 to 1.00.

**The margin predicts fragility, without labels.** At 80 percent sparsity, the label-free margin of the dense model correlates with the fraction of accuracy each class retains after pruning, with a pooled Spearman correlation of 0.665 across 36 class-seed pairs. The same 12 classes recur across the three seeds, so this is reported as a descriptive correlation rather than a formal significance test. `step5_predict.py` also prints the correlation for each seed. Computing the same quantity with ground-truth labels does not do better.

**The worst-case bound fails, and this is reported rather than omitted.** Applying the Cauchy-Schwarz inequality to the exact identity gives a criterion that certifies no predictions beyond 30 percent sparsity, although `step3_prune.py` shows the model still performing well at 70 percent. Estimating the projected drift directly, with a separate threshold for each class predicted by the compressed model, certifies 83 percent of predictions at 50 percent sparsity (mean over three seeds, risk level 5 percent). Thresholds are fitted on an unlabelled half of the test data, and coverage is measured on the other half (`algo_v2.py`).

## Requirements

Python 3 with the packages in `requirements.txt`:

    pip install -r requirements.txt

Everything runs on a laptop CPU in a few minutes. No dataset is downloaded and no GPU is required.

## Running the experiments

Keep all files in one directory and run them in order. `step1_train.py` must run first, since the remaining scripts load the model it saves.

    python step1_train.py
    python step2_longtail.py
    python step3_prune.py
    python step4_margins.py
    python step5_predict.py
    python step6_certificate.py
    python algo_v2.py

| Script | Purpose |
|---|---|
| `step1_train.py` | Trains the model and saves it |
| `step2_longtail.py` | Per-class accuracy before compression |
| `step3_prune.py` | Accuracy against sparsity, for head, tail and overall |
| `step4_margins.py` | Per-class margin, computed with and without labels |
| `step5_predict.py` | Correlation between margin and retained accuracy, three seeds |
| `step6_certificate.py` | Behaviour of the worst-case bound |
| `algo_v2.py` | Comparison of the three certification criteria |

## Reproducibility

All random seeds are fixed in `common.py`. `step5_predict.py` reports each seed separately as well as the pooled result, since a single seed is not evidence.
