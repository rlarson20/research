# Experiment 3 — Unsupervised: clustering, dim-reduction, anomaly detection

## Goal

(1) Run three clustering algorithms on a high-dim dataset with known labels
and score them with both an internal metric (silhouette) and a supervised
metric (adjusted Rand index). (2) Visualize the structure with two
fundamentally different 2D projections — linear (PCA) vs. nonlinear (t-SNE)
— colored both by true labels and by the KMeans assignments. (3) Inject
anomalies into a single-class subset and benchmark four anomaly detectors.

## Dataset

`sklearn.datasets.load_digits()` — 1797 samples of 8×8 grayscale digit
images flattened to a 64-dim vector. Ten classes (digits 0-9), nearly
balanced. Ships with sklearn.

For the anomaly-detection sub-experiment, the "normal" set is **all digit-1
images** (~180 samples); the anomalies are a random ~5% draw from the other
nine classes mixed in, with a known ground-truth mask.

## sklearn APIs used and why

### Clustering
- `KMeans(n_clusters=10, n_init="auto", random_state=42)` — Lloyd's
  algorithm; finds a Voronoi partition that minimizes within-cluster
  variance. Centroid-based, assumes roughly spherical clusters of similar size.
  `n_init="auto"` runs ~10 inits and keeps the best inertia.
- `AgglomerativeClustering(n_clusters=10, linkage="ward")` — hierarchical
  bottom-up merging; Ward linkage merges clusters that minimize the
  variance-increase. Doesn't require choosing centroids; works well when
  clusters are elongated.
- `DBSCAN(eps=…, min_samples=5)` — density-based; clusters are
  high-density regions separated by low-density ones. Doesn't need
  `n_clusters` but is exquisitely sensitive to `eps`. We pick `eps` by
  finding the knee of the sorted 5-nearest-neighbor distance curve — a
  standard heuristic for choosing the radius.

### Cluster evaluation
- `silhouette_score(X, labels)` — internal metric, no ground truth needed.
  For each sample, `(b - a) / max(a, b)` where `a` is mean intra-cluster
  distance and `b` is mean nearest-other-cluster distance. Range
  `[-1, 1]`, higher is better.
- `adjusted_rand_score(true, pred)` — external metric, requires ground
  truth. Counts agreeing pairs corrected for chance. Range `[-1, 1]`, 0 is
  random, 1 is perfect.

### Dimensionality reduction (for visualization only)
- `PCA(n_components=2)` — linear projection onto the top-2 eigenvectors of
  the covariance matrix. Fast, deterministic, distance-preserving for the
  retained axes, but limited to capturing linear structure.
- `TSNE(n_components=2, perplexity=30, random_state=42)` — nonlinear
  manifold embedding that preserves local neighborhoods. Slow,
  stochastic, distorts global distances — but excellent at revealing
  cluster structure in dense high-dim data like digits.

### Anomaly detection (four contrasting approaches)
- `IsolationForest(contamination=0.05, random_state=42)` — random partition
  trees; anomalies are points isolated by short paths. Ensemble-based,
  scales well, works on high-dim data.
- `OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)` — fits a tight
  boundary around the bulk of the training data. Sensitive to gamma; nu
  approximately bounds the expected fraction of outliers.
- `LocalOutlierFactor(n_neighbors=20, contamination=0.05)` —
  density-based; compares a point's local density to that of its
  neighbors. Used in transductive mode (no `.predict()` on new data; you
  call `.fit_predict()` on the combined set).
- `EllipticEnvelope(contamination=0.05, support_fraction=None, random_state=42)`
  — fits a robust Gaussian (Minimum Covariance Determinant) and flags
  points with high Mahalanobis distance. Strong when normal data really
  is roughly elliptical, otherwise can fail badly.

### Reporting
- `roc_auc_score(anomaly_mask, -decision_function)` — anomaly detectors
  conventionally return higher scores for "more normal", so we negate to
  align with the convention "higher → more anomalous" expected by AUC.

## How to read the outputs

- `cluster_metrics.txt` — silhouette and ARI for the three clusterers.
- `pca_tsne.png` — 2×2 grid: (PCA, t-SNE) × (true-labels, KMeans-labels).
  Reveals whether the algorithms find structure that's visually recoverable.
- `anomaly_grid.png` — 2×2 PCA projection of the contaminated subset, one
  panel per detector; anomalies highlighted vs. inliers.
- `anomaly_auc.txt` — ROC-AUC for each detector on the known anomaly mask.
