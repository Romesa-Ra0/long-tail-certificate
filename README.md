# Certifying Compression-Induced Representation Drift

Can we tell that model compression has destroyed a rare-class capability, without
having any labels for that class?

Compression preserves overall accuracy by sacrificing the rarest classes. Where
labels exist, this is easy to measure. In a zero-shot setting it is not, because
no labelled examples of the affected classes are available. This repository
studies whether the geometry of the embedding space reveals the loss instead.

## Setting

A prototype-similarity classifier is used, with the same decision rule as a
vision-language model: an encoder maps inputs to an embedding, and the predicted
class is the fixed prototype with the highest cosine similarity. Training data is
long-tailed, from 2,000 examples for the most frequent class down to 12 for the
rarest. Compression is post-training magnitude pruning, applied without
fine-tuning.

The data is synthetic and no pretrained vision-language model is involved. These
experiments test assumptions; they are not a clinical result.

## Findings

Averaged over three random seeds.

**Compression hides its cost in the tail.** At the highest sparsity where overall
accuracy has fallen by no more than five points, the four most frequent classes
lose no accuracy at all, while the four rarest lose up to 11.4 points.

**Margin tracks class frequency.** The rank correlation between the number of
training examples per class and its decision margin is 0.97 to 1.00.

**The margin predicts fragility, without labels.** Correlating the label-free
margin of the dense model with the accuracy each class retains at 80 percent
sparsity gives a pooled Spearman correlation of 0.665 (p < 0.0001). Computing the
same quantity with ground-truth labels does not do better.

**The worst-case bound fails, and this is reported rather than omitted.**
Applying Cauchy-Schwarz to the exact identity yields a criterion that certifies
no predictions beyond 30 percent sparsity, although the model is still correct at
70 percent. Estimating the projected drift directly, separately for each
predicted class, certifies 83 percent of predictions at 50 percent sparsity.

## Requirements

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
