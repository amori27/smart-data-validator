import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnomalyResult:
    indices: list[int]
    scores: list[float]
    method: str
    count: int


class AnomalyDetector:
    """Statistical anomaly detection for numeric data."""

    @staticmethod
    def zscore_outliers(
        data: pd.Series, threshold: float = 3.0
    ) -> AnomalyResult:
        values = data.dropna().astype(float)
        mean = values.mean()
        std = values.std()
        if std == 0:
            return AnomalyResult(indices=[], scores=[], method="zscore", count=0)
        z_scores = np.abs((values - mean) / std)
        mask = z_scores > threshold
        return AnomalyResult(
            indices=values.index[mask].tolist(),
            scores=z_scores[mask].tolist(),
            method="zscore",
            count=int(mask.sum()),
        )

    @staticmethod
    def iqr_outliers(
        data: pd.Series, factor: float = 1.5
    ) -> AnomalyResult:
        values = data.dropna().astype(float)
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (values < lower) | (values > upper)
        scores = []
        for v in values[mask]:
            dist = max(lower - v, v - upper, 0)
            scores.append(float(dist / iqr) if iqr > 0 else 0.0)
        return AnomalyResult(
            indices=values.index[mask].tolist(),
            scores=scores,
            method="iqr",
            count=int(mask.sum()),
        )

    @staticmethod
    def modified_zscore(
        data: pd.Series, threshold: float = 3.5
    ) -> AnomalyResult:
        values = data.dropna().astype(float)
        median = values.median()
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return AnomalyResult(indices=[], scores=[], method="modified_zscore", count=0)
        modified_z = 0.6745 * (values - median) / mad
        mask = np.abs(modified_z) > threshold
        return AnomalyResult(
            indices=values.index[mask].tolist(),
            scores=np.abs(modified_z[mask]).tolist(),
            method="modified_zscore",
            count=int(mask.sum()),
        )

    @staticmethod
    def detect_all(
        data: pd.Series,
        zscore_threshold: float = 3.0,
        iqr_factor: float = 1.5,
    ) -> dict[str, AnomalyResult]:
        return {
            "zscore": AnomalyDetector.zscore_outliers(data, zscore_threshold),
            "iqr": AnomalyDetector.iqr_outliers(data, iqr_factor),
            "modified_zscore": AnomalyDetector.modified_zscore(data),
        }
