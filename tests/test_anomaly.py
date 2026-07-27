import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly_detector import AnomalyDetector


@pytest.fixture
def normal_data():
    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0, 1, 1000))


@pytest.fixture
def data_with_outliers():
    rng = np.random.RandomState(42)
    data = rng.normal(0, 1, 1000)
    data[0] = 100
    data[1] = -100
    return pd.Series(data)


class TestZScore:
    def test_no_outliers(self, normal_data):
        result = AnomalyDetector.zscore_outliers(normal_data, threshold=3.0)
        assert result.count < 50
        assert result.method == "zscore"

    def test_outliers_detected(self, data_with_outliers):
        result = AnomalyDetector.zscore_outliers(data_with_outliers, threshold=3.0)
        assert result.count >= 2
        assert 0 in result.indices
        assert 1 in result.indices

    def test_higher_threshold_fewer_outliers(self, data_with_outliers):
        r3 = AnomalyDetector.zscore_outliers(data_with_outliers, threshold=3.0)
        r5 = AnomalyDetector.zscore_outliers(data_with_outliers, threshold=5.0)
        assert r5.count <= r3.count

    def test_indices_match_scores(self, data_with_outliers):
        result = AnomalyDetector.zscore_outliers(data_with_outliers, threshold=3.0)
        assert len(result.indices) == len(result.scores)
        assert len(result.indices) == result.count


class TestIQR:
    def test_no_outliers(self, normal_data):
        result = AnomalyDetector.iqr_outliers(normal_data, factor=1.5)
        assert result.method == "iqr"
        assert result.count < 50

    def test_outliers_detected(self, data_with_outliers):
        result = AnomalyDetector.iqr_outliers(data_with_outliers, factor=1.5)
        assert result.count >= 2

    def test_higher_factor_fewer_outliers(self, data_with_outliers):
        r15 = AnomalyDetector.iqr_outliers(data_with_outliers, factor=1.5)
        r3 = AnomalyDetector.iqr_outliers(data_with_outliers, factor=3.0)
        assert r3.count <= r15.count

    def test_all_scores_non_negative(self, data_with_outliers):
        result = AnomalyDetector.iqr_outliers(data_with_outliers)
        for s in result.scores:
            assert s >= 0


class TestModifiedZScore:
    def test_no_outliers(self, normal_data):
        result = AnomalyDetector.modified_zscore(normal_data, threshold=3.5)
        assert result.method == "modified_zscore"
        assert result.count < 50

    def test_outliers_detected(self, data_with_outliers):
        result = AnomalyDetector.modified_zscore(data_with_outliers, threshold=3.5)
        assert result.count >= 2

    def test_scores_positive(self, data_with_outliers):
        result = AnomalyDetector.modified_zscore(data_with_outliers)
        for s in result.scores:
            assert s >= 0


class TestDetectAll:
    def test_returns_all_methods(self, normal_data):
        results = AnomalyDetector.detect_all(normal_data)
        assert set(results.keys()) == {"zscore", "iqr", "modified_zscore"}

    def test_all_results_have_count(self, normal_data):
        results = AnomalyDetector.detect_all(normal_data)
        for r in results.values():
            assert r.count >= 0

    def test_constant_data_no_outliers(self):
        data = pd.Series([5.0] * 100)
        result = AnomalyDetector.zscore_outliers(data)
        assert result.count == 0

    def test_modified_zscore_constant_no_outliers(self):
        data = pd.Series([5.0] * 100)
        result = AnomalyDetector.modified_zscore(data)
        assert result.count == 0
