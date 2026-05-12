# Experiment 3 — Postmortem

## Headline

- **Agglomerative (Ward linkage) recovered the 10 digit classes best**
  (ARI = 0.664) — even though both it and KMeans got near-identical
  *silhouette* scores. Internal and external metrics disagree, in a
  textbook way.
- **DBSCAN is the wrong tool for high-dimensional digit pixels.**
  Densities are too uneven in 64-D; 82% of points get flagged as noise
  before any reasonable eps recovers the class structure.
- **t-SNE blows PCA out of the water for this kind of high-dim cluster
  structure.** PCA's 2D projection leaves the digits as one diffuse blob;
  t-SNE separates the classes into ten visible islands.
- **EllipticEnvelope topped the anomaly leaderboard at 0.988 ROC-AUC** on
  the "digit-1 vs. injected anomalies" task; Isolation Forest was right
  behind at 0.981.

## Clustering metrics

```
algorithm       silhouette    ARI
---------       ----------    ---
KMeans                0.136    0.531
Agglomerative         0.125    0.664
DBSCAN                0.342    0.029

DBSCAN eps=3.236  clusters_found=14  noise_points=1474  (of 1797)
```

The silhouette/ARI disagreement is informative:

- KMeans and Agglomerative get essentially the same silhouette (≈0.13)
  because both partition the whole space into ten compact groups, but
  Agglomerative is +0.13 ARI better — it finds the linkage structure that
  better matches actual digit shape similarity.
- DBSCAN has the *highest* silhouette by far (0.342), but only because it
  computes silhouette over the 18% of points it didn't call "noise". A
  high silhouette on a tiny core of obviously-dense points isn't
  meaningful — ARI = 0.029 reveals it's not finding the digit classes at
  all.

## DBSCAN eps selection

The classical k-distance-knee heuristic (`argmax(diff(sorted_kth_distance))`)
picked `eps = 30.5` and merged everything into one giant cluster — the
largest single jump in the curve is at the very tail, caused by a handful
of true outliers. **Replaced with a silhouette grid** over the 10th-90th
percentile of 5-NN distances; that landed on `eps = 3.236` which actually
forms multiple clusters but still relegates 82% of points to "noise".

Conclusion: DBSCAN on raw 64-D pixel space is a poor choice. To make it
work you'd need to project to a lower-dim manifold first (PCA/UMAP) and
then re-run DBSCAN there.

## Projections

`outputs/pca_tsne.png` makes the contrast vivid:

- **PCA panels** show one diffuse cloud with weak class color gradient —
  the top-2 linear directions capture only ~28% of variance.
- **t-SNE panels** show ten clearly separated islands. Coloring by
  *KMeans* labels (bottom-right) shows KMeans mostly agrees with t-SNE
  islands but mis-splits/merges a couple — the islands corresponding to
  digits "4/9" and "3/5/8" are visually closer in t-SNE space, and KMeans
  occasionally bridges them.

## Anomaly detection

Setup: 182 examples of digit-1 + 9 anomalies sampled from the other
classes (5% contamination), all standardized.

| Detector              | ROC-AUC |
|---                    |---:     |
| EllipticEnvelope      | 0.988   |
| IsolationForest       | 0.981   |
| LocalOutlierFactor    | 0.966   |
| OneClassSVM           | 0.861   |

`EllipticEnvelope` won because digit-1 pixel-intensity distributions
really *are* close to a (high-dim) elliptical Gaussian — the MCD
robust-covariance estimator is then very effective. Isolation Forest is
nearly tied and would be the safe default for unknown-distribution data.
OneClassSVM trailed at default `nu=0.05` / `gamma="scale"` — like SVR in
Experiment 2, it'd want a small grid search.

The `outputs/anomaly_grid.png` plot overlays each detector's predicted
outliers (orange) on the PCA-projected contaminated set, with red rings
marking the *true* anomalies. EllipticEnvelope and IsolationForest hit
8-9 of 9 anomalies with minimal false positives; OCSVM has noticeably
more false positives scattered across the inlier cloud.

## Surprises and gotchas

- **The k-distance knee heuristic is fragile.** Documented this in the
  code with a comment and replaced with a silhouette-grid eps picker.
- **LocalOutlierFactor's transductive API is different.** It has no
  `predict()` by default (set `novelty=True` for that), and its
  "decision score" lives on `negative_outlier_factor_`, not
  `decision_function`. Handled with a dedicated branch.
- **Don't compute silhouette on DBSCAN labels including `-1`.** The noise
  label distorts the metric. Filtered with `mask = labels != -1`.

## What I'd try next

1. Project digits to 30-D via PCA first, *then* run DBSCAN — should
   restore some density structure.
2. Use `HDBSCAN` (it's now in sklearn as `sklearn.cluster.HDBSCAN`) which
   handles varying density much better than vanilla DBSCAN.
3. Calibrate the OCSVM with a `(nu, gamma)` grid; it should match or
   exceed Isolation Forest.
4. Try `MiniBatchKMeans` for speed comparison on the same task.
