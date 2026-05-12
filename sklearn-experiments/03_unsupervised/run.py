"""Experiment 3 — Clustering + dim-reduction + anomaly detection."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

RANDOM_STATE = 42
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def pick_dbscan_eps(X, k=5):
    """Returns (best_eps, kth_distances) chosen by a small silhouette grid.

    The classical "knee of the k-NN distance curve" trick (argmax of the
    first difference) is fragile in high-dim because the largest jump is
    usually at the tail caused by a handful of outliers - it picks an eps
    that merges everything. Instead, scan the 10th-90th percentile range
    of 5-NN distances and pick the eps with the best silhouette among
    multi-cluster solutions.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    kth = np.sort(distances[:, -1])
    candidates = np.quantile(kth, np.linspace(0.10, 0.90, 12))
    best = (None, -np.inf, None)
    for eps in candidates:
        labels = DBSCAN(eps=float(eps), min_samples=5).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
        mask = labels != -1
        if mask.sum() < 2:
            continue
        sil = silhouette_score(X[mask], labels[mask])
        if sil > best[1]:
            best = (float(eps), sil, n_clusters)
    if best[0] is None:
        return float(np.median(kth)), kth
    return best[0], kth


def clustering_section(X_scaled, y_true):
    print("[1] Clustering on standardized digits …")
    results = {}
    for name, alg in {
        "KMeans": KMeans(n_clusters=10, n_init="auto", random_state=RANDOM_STATE),
        "Agglomerative": AgglomerativeClustering(n_clusters=10, linkage="ward"),
    }.items():
        labels = alg.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ari = adjusted_rand_score(y_true, labels)
        results[name] = (labels, sil, ari)
        print(f"  {name:14s} silhouette={sil:.3f}  ARI={ari:.3f}")

    eps, kth = pick_dbscan_eps(X_scaled, k=5)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(X_scaled[mask], labels[mask]) if mask.sum() > 1 else float("nan")
    else:
        sil = float("nan")
    ari = adjusted_rand_score(y_true, labels)
    results["DBSCAN"] = (labels, sil, ari)
    print(f"  DBSCAN         eps={eps:.3f}  clusters={n_clusters}  noise={n_noise}  silhouette={sil:.3f}  ARI={ari:.3f}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(kth)
    ax.axhline(eps, color="red", linestyle="--", label=f"chosen eps = {eps:.3f}")
    ax.set_xlabel("samples sorted by 5-NN distance")
    ax.set_ylabel("distance to 5th NN")
    ax.set_title("k-distance (knee selects DBSCAN eps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "dbscan_kdist.png", dpi=120)
    plt.close(fig)
    return results, eps, n_clusters, n_noise


def projections_plot(X_scaled, y_true, kmeans_labels):
    print("[2] Computing PCA and t-SNE projections …")
    Z_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    Z_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=RANDOM_STATE).fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, Z, title in [
        (axes[0, 0], Z_pca, "PCA — true labels"),
        (axes[0, 1], Z_tsne, "t-SNE — true labels"),
        (axes[1, 0], Z_pca, "PCA — KMeans labels"),
        (axes[1, 1], Z_tsne, "t-SNE — KMeans labels"),
    ]:
        labels = y_true if "true" in title else kmeans_labels
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=6, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("2D projections of digits (64 → 2)", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "pca_tsne.png", dpi=120)
    plt.close(fig)


def write_cluster_metrics(results, eps, n_clusters_dbscan, n_noise):
    lines = ["algorithm       silhouette    ARI",
             "---------       ----------    ---"]
    for name, (_, sil, ari) in results.items():
        sil_s = "nan" if np.isnan(sil) else f"{sil:.3f}"
        lines.append(f"{name:14s} {sil_s:>10s}    {ari:.3f}")
    lines += [
        "",
        f"DBSCAN eps={eps:.3f}  clusters_found={n_clusters_dbscan}  noise_points={n_noise}",
    ]
    (OUT / "cluster_metrics.txt").write_text("\n".join(lines) + "\n")


def build_contaminated_subset(X, y, normal_class=1, anomaly_frac=0.05, rng=None):
    rng = np.random.default_rng(RANDOM_STATE) if rng is None else rng
    normal_idx = np.where(y == normal_class)[0]
    other_idx = np.where(y != normal_class)[0]
    n_anom = max(5, int(len(normal_idx) * anomaly_frac))
    anom_pick = rng.choice(other_idx, size=n_anom, replace=False)
    idx = np.concatenate([normal_idx, anom_pick])
    rng.shuffle(idx)
    is_anom = (y[idx] != normal_class).astype(int)
    return X[idx], is_anom


def anomaly_section(X, y):
    print("[3] Anomaly detection on class-1-vs-others (5% contamination) …")
    Xc, mask = build_contaminated_subset(X, y, normal_class=1, anomaly_frac=0.05)
    Xs = StandardScaler().fit_transform(Xc)
    print(f"  contaminated set: {Xs.shape}, {int(mask.sum())} anomalies of {len(mask)}")

    Z_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    detectors = {
        "IsolationForest": IsolationForest(contamination=0.05, random_state=RANDOM_STATE),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
        "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=20, contamination=0.05),
        "EllipticEnvelope": EllipticEnvelope(contamination=0.05, random_state=RANDOM_STATE),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    aucs = {}
    for ax, (name, det) in zip(axes.flatten(), detectors.items()):
        if isinstance(det, LocalOutlierFactor):
            pred = det.fit_predict(Xs)
            score = -det.negative_outlier_factor_
        else:
            det.fit(Xs)
            pred = det.predict(Xs)
            try:
                score = -det.decision_function(Xs)
            except AttributeError:
                score = -det.score_samples(Xs)
        try:
            auc = roc_auc_score(mask, score)
        except ValueError:
            auc = float("nan")
        aucs[name] = auc

        inlier = pred == 1
        outlier = pred == -1
        ax.scatter(Z_pca[inlier, 0], Z_pca[inlier, 1], c="lightgray", s=10, label="predicted inlier")
        ax.scatter(Z_pca[outlier, 0], Z_pca[outlier, 1], c="orange", s=18, edgecolors="black", linewidths=0.5, label="predicted outlier")
        truly = mask == 1
        ax.scatter(Z_pca[truly, 0], Z_pca[truly, 1], facecolors="none", edgecolors="red", s=60, linewidths=1.2, label="true anomaly")
        ax.set_title(f"{name}\nROC-AUC = {auc:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Anomaly detectors on contaminated digit-1 set (PCA-projected)", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "anomaly_grid.png", dpi=120)
    plt.close(fig)

    lines = ["detector              roc_auc",
             "--------              -------"]
    for name, auc in aucs.items():
        lines.append(f"{name:20s}  {auc:.3f}")
    (OUT / "anomaly_auc.txt").write_text("\n".join(lines) + "\n")
    print("\n".join("  " + line for line in lines))
    return aucs


def main():
    data = load_digits()
    X, y = data.data, data.target
    print(f"Dataset: digits  X={X.shape}, classes={len(set(y))}")

    Xs = StandardScaler().fit_transform(X)
    results, eps, n_clusters_dbscan, n_noise = clustering_section(Xs, y)
    write_cluster_metrics(results, eps, n_clusters_dbscan, n_noise)

    kmeans_labels = results["KMeans"][0]
    projections_plot(Xs, y, kmeans_labels)
    aucs = anomaly_section(X, y)

    summary = [
        "clustering_silhouette_ari:",
        *[f"  {name}: sil={results[name][1]:.3f} ari={results[name][2]:.3f}"
          for name in ("KMeans", "Agglomerative", "DBSCAN")],
        f"dbscan_eps: {eps:.3f}  clusters: {n_clusters_dbscan}  noise: {n_noise}",
        "anomaly_auc:",
        *[f"  {n}: {a:.3f}" for n, a in aucs.items()],
    ]
    (OUT / "summary.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
