# Experiment 3 — imbalanced-learn: when does explicit resampling beat class weighting?

## Goal

`sklearn-experiments/04_imbalanced_calibration` found that at 95/5 imbalance,
`class_weight="balanced"` plus threshold tuning was enough — but it never
compared against the family of explicit resamplers that
`imbalanced-learn` provides (SMOTE, ADASYN, NearMiss, etc.). Many tutorials
present SMOTE as the default move for imbalanced classification. Test that
recommendation directly: at severe imbalance (99/1), does **explicit
minority-class generation** beat (a) doing nothing, (b) class weighting?
Or does it hurt by distorting probability calibration?

## Note on what this experiment originally was

The Phase 4 plan called for a Hugging Face transformers experiment here
(zero-shot vs frozen-encoder linear probe). The sandbox blocks
`huggingface.co` despite an indication otherwise — the HF API endpoint
returns `403 x-deny-reason: host_not_allowed`, and there's no cached
model on disk. Rather than burn the slot, we pivoted to imbalanced-learn:
also a popular non-sklearn library, also pairs naturally with the prior
sklearn-experiments-4, and fully offline.

## Dataset

`sklearn.datasets.make_classification(n_samples=10000, n_features=20,
n_informative=10, n_redundant=5, n_clusters_per_class=2,
weights=[0.99, 0.01], class_sep=0.7, flip_y=0.01, random_state=42)` —
141 minority examples in 10,000 rows. 75/25 stratified train/test split.

## APIs used and why

### imbalanced-learn samplers
- **`RandomOverSampler`** — duplicates minority rows; trivial baseline.
- **`SMOTE(k_neighbors=5)`** — linear interpolation between a minority
  point and a random one of its 5 nearest minority neighbors.
- **`ADASYN`** — like SMOTE but generates *more* synthetic samples for
  hard-to-classify minority points (where local neighborhood is mostly
  majority).
- **`RandomUnderSampler`** — drops majority rows uniformly until balanced.
- **`NearMiss(version=1)`** — drops majority rows that are *far* from the
  minority class; tries to make the boundary tighter.
- **`SMOTETomek`** — SMOTE oversample, then Tomek-link cleanup (removes
  ambiguous pairs).
- **`SMOTEENN`** — SMOTE oversample, then Edited Nearest Neighbours cleanup.

### Pipeline integration
- **`imblearn.pipeline.Pipeline`** (NOT `sklearn.pipeline.Pipeline`) —
  this is the critical API choice. The imblearn version applies the
  sampler *only* on training folds inside cross-validation; the sklearn
  version would leak resampled rows into the test fold. Three-step
  pipeline: `StandardScaler → sampler → LogisticRegression`.

### Baselines & metrics
- **`LogisticRegression(class_weight="balanced", max_iter=2000)`** —
  loss reweighting baseline.
- **`LogisticRegression()` plain** — no resample, no weight.
- **Threshold-tuned F1**: walk the `precision_recall_curve` and pick
  argmax(F1). Reports both F1 @ 0.5 and F1 @ tuned threshold per sampler.
- **`average_precision_score` (AP)** is the headline imbalance metric;
  area under the PR curve, not ROC.
- **5-fold CV stability** on the top-3 by AP via `cross_val_score(...,
  scoring="average_precision")`.

## How to read the outputs

- `resampler_table.txt` — one row per sampler with AUC, AP, F1@0.5,
  precision@0.5, recall@0.5, F1@tuned, and the tuned threshold.
- `f1_default_vs_tuned.png` — bar pairs comparing F1@0.5 vs F1@tuned
  per sampler. Shows how big the threshold-tuning lever is per sampler.
- `pr_curves.png` — precision-recall curves overlaid for a representative
  subset (the full 9 would be unreadable).
- `confusion_winner.png` — confusion matrix for the F1-tuned winner.
- `winner.txt` — summary line + top-3 CV AP table.
