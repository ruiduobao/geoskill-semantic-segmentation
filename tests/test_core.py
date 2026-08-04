"""Core algorithm tests for semantic-segmentation."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conftest import mod


class TestFeatureMatrix:
    def test_shape(self):
        cube = np.random.uniform(0, 1, (4, 16, 20)).astype(np.float32)
        feat = mod.build_feature_matrix(cube)
        assert feat.shape == (16 * 20, 4)

    def test_2d_promoted(self):
        feat = mod.build_feature_matrix(np.zeros((8, 8), dtype=np.float32))
        assert feat.shape == (64, 1)

    def test_rejects_1d(self):
        with pytest.raises(mod.ValidationError):
            mod.build_feature_matrix(np.zeros((10,)))

    def test_values_preserved(self):
        cube = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
        feat = mod.build_feature_matrix(cube)
        # 第一个像元 (0,0) 的两个波段值
        assert feat[0, 0] == cube[0, 0, 0]
        assert feat[0, 1] == cube[1, 0, 0]


class TestTrainClassifier:
    def test_kmeans_predict(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.1, (50, 3))
        b = rng.normal(5, 0.1, (50, 3))
        feat = np.vstack([a, b])
        model = mod.train_classifier(feat, n_classes=2, method="kmeans", seed=0)
        pred = np.asarray(model.predict(feat))
        # 两簇应被分开：前 50 与后 50 标签不同
        assert len(np.unique(pred[:50])) == 1
        assert len(np.unique(pred[50:])) == 1
        assert pred[0] != pred[50]

    def test_rf_requires_labels(self):
        feat = np.random.uniform(0, 1, (20, 3))
        with pytest.raises(mod.UsageError):
            mod.train_classifier(feat, n_classes=2, method="rf")

    def test_rf_supervised(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 0.1, (40, 3))
        b = rng.normal(5, 0.1, (40, 3))
        feat = np.vstack([a, b])
        labels = np.array([0] * 40 + [1] * 40)
        model = mod.train_classifier(feat, n_classes=2, method="rf", labels=labels, seed=1)
        pred = np.asarray(model.predict(feat))
        assert np.mean(pred == labels) > 0.95

    def test_unknown_method(self):
        with pytest.raises(mod.UsageError):
            mod.train_classifier(np.zeros((10, 2)), 2, method="svm")


class TestPredictTiled:
    def test_tiled_equals_full(self):
        """分块预测与整幅预测应完全一致（同一模型）。"""
        rng = np.random.default_rng(2)
        feat = rng.normal(0, 1, (40 * 40, 4))
        model = mod.train_classifier(feat, n_classes=3, method="kmeans", seed=2)
        full = np.asarray(model.predict(feat)).reshape(40, 40)
        tiled = mod.predict_tiled(model, feat, 40, 40, tile=16)
        np.testing.assert_array_equal(full, tiled)

    def test_size_mismatch_raises(self):
        rng = np.random.default_rng(3)
        feat = rng.normal(0, 1, (100, 3))
        model = mod.train_classifier(feat, n_classes=2, method="kmeans", seed=3)
        with pytest.raises(mod.ValidationError):
            mod.predict_tiled(model, feat, 12, 12)


class TestMajorityFilter:
    def test_removes_salt_pixel(self):
        lm = np.zeros((9, 9), dtype=np.int64)
        lm[4, 4] = 7  # 孤立噪声像元
        out = mod.majority_filter(lm, size=3)
        assert out[4, 4] == 0  # 被邻域众数（0）覆盖

    def test_preserves_uniform(self):
        lm = np.full((9, 9), 3, dtype=np.int64)
        out = mod.majority_filter(lm, size=3)
        np.testing.assert_array_equal(out, lm)

    def test_size_lt2_returns_copy(self):
        lm = np.array([[1, 2], [3, 4]], dtype=np.int64)
        out = mod.majority_filter(lm, size=1)
        np.testing.assert_array_equal(out, lm)


class TestLabelAccuracy:
    def test_permuted_labels_full_accuracy(self):
        """无监督类别编号可任意置换，匹配后精度应为 1.0。"""
        truth = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([2, 2, 0, 0, 1, 1])  # 完全置换
        assert mod.label_accuracy(pred, truth) == pytest.approx(1.0)

    def test_half_correct(self):
        truth = np.array([0, 0, 0, 0])
        pred = np.array([0, 0, 1, 1])
        # 最佳匹配 0->0 命中 2 个，剩下 2 个无法命中 -> 0.5
        assert mod.label_accuracy(pred, truth) == pytest.approx(0.5)


class TestSemanticSegment:
    def test_synthetic_accuracy_high(self):
        cube, truth, _ = mod.generate_synthetic([116, 39, 117, 40], seed=5)
        label_map, info = mod.semantic_segment(cube, n_classes=3, method="kmeans", seed=5)
        acc = mod.label_accuracy(label_map, truth)
        assert acc > 0.85
        assert info["n_classes"] == 3

    def test_output_shape(self):
        cube = np.random.uniform(0, 1, (4, 32, 32)).astype(np.float32)
        label_map, info = mod.semantic_segment(cube, n_classes=3)
        assert label_map.shape == (32, 32)
        assert label_map.dtype.kind in "iu"

    def test_rf_supervised_with_labels(self):
        cube, truth, _ = mod.generate_synthetic([116, 39, 117, 40], seed=6)
        label_map, info = mod.semantic_segment(
            cube, n_classes=3, method="rf", labels=truth, seed=6)
        acc = mod.label_accuracy(label_map, truth)
        assert acc > 0.9

    def test_rf_partial_labels(self):
        """部分标注（-1 = 未标注）也能训练。"""
        cube, truth, _ = mod.generate_synthetic([116, 39, 117, 40], seed=7)
        partial = truth.copy()
        partial[::2, :] = -1  # 一半像元未标注
        label_map, _ = mod.semantic_segment(
            cube, n_classes=3, method="rf", labels=partial, seed=7)
        acc = mod.label_accuracy(label_map, truth)
        assert acc > 0.85

    def test_rf_without_labels_raises(self):
        cube = np.random.uniform(0, 1, (4, 16, 16)).astype(np.float32)
        with pytest.raises(mod.UsageError):
            mod.semantic_segment(cube, n_classes=3, method="rf")

    def test_rf_too_few_labels_raises(self):
        cube = np.random.uniform(0, 1, (4, 16, 16)).astype(np.float32)
        lab = np.full((16, 16), -1, dtype=np.int64)
        lab[0, 0] = 0
        with pytest.raises(mod.ValidationError):
            mod.semantic_segment(cube, n_classes=3, method="rf", labels=lab)


class TestSynthetic:
    def test_shapes(self):
        cube, truth, info = mod.generate_synthetic([116, 39, 117, 40], n_bands=4)
        assert cube.shape == (4, 64, 64)
        assert truth.shape == (64, 64)
        assert set(np.unique(truth)).issubset({0, 1, 2})


class TestGeoTiffIO:
    def test_roundtrip(self, tmp_path):
        arr = np.random.uniform(0, 10, (16, 16)).astype(np.float32)
        bbox = [116.0, 39.0, 117.0, 40.0]
        path = str(tmp_path / "seg.tif")
        mod.write_geotiff(path, arr, bbox)
        back, rb = mod.read_geotiff(path)
        np.testing.assert_allclose(rb, bbox, atol=1e-6)
        np.testing.assert_allclose(back[0], arr, atol=1e-4)

    def test_read_missing_raises(self):
        with pytest.raises(mod.UsageError):
            mod.read_geotiff("/nonexistent/x.tif")
