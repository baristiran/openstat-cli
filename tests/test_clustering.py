"""Tests for clustering, MDS, discriminant analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session

_sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


@pytest.fixture()
def cluster_df():
    rng = np.random.default_rng(42)
    # 3 clear clusters
    c1 = rng.normal([0, 0], 0.5, (50, 2))
    c2 = rng.normal([5, 0], 0.5, (50, 2))
    c3 = rng.normal([2.5, 5], 0.5, (50, 2))
    X = np.vstack([c1, c2, c3])
    return pl.DataFrame({"x": X[:, 0].tolist(), "y": X[:, 1].tolist()})


@pytest.fixture()
def disc_df():
    rng = np.random.default_rng(10)
    n = 60
    X_a = rng.normal([0, 0], 1.0, (n, 2))
    X_b = rng.normal([3, 3], 1.0, (n, 2))
    X = np.vstack([X_a, X_b])
    labels = ["A"] * n + ["B"] * n
    return pl.DataFrame({
        "group": labels,
        "x1": X[:, 0].tolist(),
        "x2": X[:, 1].tolist(),
    })


@pytest.fixture()
def session_c(cluster_df):
    s = Session()
    s.df = cluster_df
    return s


class TestKMeans:
    def test_basic(self, cluster_df):
        from openstat.stats.clustering import fit_kmeans
        r = fit_kmeans(cluster_df, ["x", "y"], k=3)
        assert r["k"] == 3
        assert len(r["cluster_sizes"]) == 3
        assert r["silhouette_score"] > 0.5

    def test_inertia(self, cluster_df):
        from openstat.stats.clustering import fit_kmeans
        r = fit_kmeans(cluster_df, ["x", "y"], k=3)
        assert r["inertia"] > 0

    def test_labels_length(self, cluster_df):
        from openstat.stats.clustering import fit_kmeans
        r = fit_kmeans(cluster_df, ["x", "y"], k=3)
        assert len(r["labels"]) == 150

    def test_centroids_shape(self, cluster_df):
        from openstat.stats.clustering import fit_kmeans
        r = fit_kmeans(cluster_df, ["x", "y"], k=3)
        assert len(r["centroids"]) == 3
        assert len(r["centroids"][0]) == 2

    def test_k2(self, cluster_df):
        from openstat.stats.clustering import fit_kmeans
        r = fit_kmeans(cluster_df, ["x", "y"], k=2)
        assert r["k"] == 2


class TestHierarchical:
    def test_basic(self, cluster_df):
        from openstat.stats.clustering import fit_hierarchical
        r = fit_hierarchical(cluster_df, ["x", "y"], k=3)
        assert r["k"] == 3
        assert r["silhouette_score"] > 0.4

    def test_different_linkage(self, cluster_df):
        from openstat.stats.clustering import fit_hierarchical
        r = fit_hierarchical(cluster_df, ["x", "y"], k=3, linkage="complete")
        assert r["linkage"] == "complete"

    def test_labels_length(self, cluster_df):
        from openstat.stats.clustering import fit_hierarchical
        r = fit_hierarchical(cluster_df, ["x", "y"], k=3)
        assert len(r["labels"]) == 150


class TestMDS:
    def test_basic(self, cluster_df):
        from openstat.stats.clustering import fit_mds
        r = fit_mds(cluster_df, ["x", "y"], n_components=2)
        assert r["n_components"] == 2
        assert len(r["coordinates"]) == 150
        assert len(r["coordinates"][0]) == 2

    def test_stress_finite(self, cluster_df):
        from openstat.stats.clustering import fit_mds
        r = fit_mds(cluster_df, ["x", "y"])
        assert r["stress"] >= 0

    def test_1d(self, cluster_df):
        from openstat.stats.clustering import fit_mds
        r = fit_mds(cluster_df, ["x", "y"], n_components=1)
        assert len(r["coordinates"][0]) == 1


class TestDiscriminant:
    def test_lda_basic(self, disc_df):
        from openstat.stats.clustering import fit_discriminant
        r = fit_discriminant(disc_df, "group", ["x1", "x2"], method="lda")
        assert r["accuracy"] > 0.8
        assert r["n_classes"] == 2

    def test_qda_basic(self, disc_df):
        from openstat.stats.clustering import fit_discriminant
        r = fit_discriminant(disc_df, "group", ["x1", "x2"], method="qda")
        assert r["accuracy"] > 0.7

    def test_classes(self, disc_df):
        from openstat.stats.clustering import fit_discriminant
        r = fit_discriminant(disc_df, "group", ["x1", "x2"])
        assert set(r["classes"]) == {"A", "B"}

    def test_lda_coefs(self, disc_df):
        from openstat.stats.clustering import fit_discriminant
        r = fit_discriminant(disc_df, "group", ["x1", "x2"], method="lda")
        assert "coefficients" in r


class TestClusterCommands:
    def test_kmeans_cmd(self, session_c):
        from openstat.commands.cluster_cmds import cmd_cluster
        out = cmd_cluster(session_c, "kmeans x y k(3)")
        assert "K-Means" in out
        assert "Silhouette" in out

    def test_hierarchical_cmd(self, session_c):
        from openstat.commands.cluster_cmds import cmd_cluster
        out = cmd_cluster(session_c, "hierarchical x y k(3)")
        assert "Hierarchical" in out

    def test_unknown_method(self, session_c):
        from openstat.commands.cluster_cmds import cmd_cluster
        out = cmd_cluster(session_c, "dbscan x y")
        assert "Unknown" in out

    def test_no_args(self, session_c):
        from openstat.commands.cluster_cmds import cmd_cluster
        out = cmd_cluster(session_c, "")
        assert "Usage" in out

    def test_mds_cmd(self, session_c):
        from openstat.commands.cluster_cmds import cmd_mds
        out = cmd_mds(session_c, "x y n(2)")
        assert "MDS" in out
        assert "Stress" in out

    def test_discriminant_cmd(self):
        rng = np.random.default_rng(5)
        n = 80
        X = rng.normal(0, 1, (n, 2))
        labels = (X[:, 0] + X[:, 1] > 0).astype(str).tolist()
        df = pl.DataFrame({"g": labels, "x1": X[:, 0].tolist(), "x2": X[:, 1].tolist()})
        s = Session()
        s.df = df
        from openstat.commands.cluster_cmds import cmd_discriminant
        out = cmd_discriminant(s, "g x1 x2 method(lda)")
        assert "LDA" in out
        assert "Accuracy" in out

    def test_stores_model(self, session_c):
        from openstat.commands.cluster_cmds import cmd_cluster
        cmd_cluster(session_c, "kmeans x y k(3)")
        assert session_c._last_model is not None
