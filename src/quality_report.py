import pandas as pd
from typing import Optional


class QualityReport:
    """Generate comprehensive data quality reports."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def overview(self) -> dict:
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }

    def missing_values(self) -> dict:
        missing = self.df.isna().sum()
        total = len(self.df)
        return {
            col: {
                "count": int(missing[col]),
                "pct": round(missing[col] / total * 100, 2) if total > 0 else 0,
            }
            for col in self.df.columns
            if missing[col] > 0
        }

    def column_stats(self, column: str) -> dict:
        col = self.df[column]
        stats: dict = {
            "dtype": str(col.dtype),
            "non_null": int(col.count()),
            "null_count": int(col.isna().sum()),
            "unique": int(col.nunique()),
        }
        if pd.api.types.is_numeric_dtype(col):
            desc = col.describe()
            stats.update(
                {
                    "mean": float(desc.get("mean", 0)),
                    "std": float(desc.get("std", 0)),
                    "min": float(desc.get("min", 0)),
                    "max": float(desc.get("max", 0)),
                    "median": float(col.median()),
                    "skewness": float(col.skew()),
                    "kurtosis": float(col.kurtosis()),
                }
            )
        elif col.dtype == "object":
            vc = col.value_counts()
            stats["top_values"] = vc.head(5).to_dict()
            stats["avg_length"] = round(col.astype(str).str.len().mean(), 2)
        return stats

    def duplicate_rows(self) -> dict:
        dup_count = self.df.duplicated().sum()
        return {
            "duplicate_count": int(dup_count),
            "duplicate_pct": round(dup_count / len(self.df) * 100, 2) if len(self.df) > 0 else 0,
        }

    def full_report(self) -> dict:
        return {
            "overview": self.overview(),
            "missing_values": self.missing_values(),
            "duplicates": self.duplicate_rows(),
            "column_stats": {col: self.column_stats(col) for col in self.df.columns},
        }

    def print_summary(self) -> None:
        overview = self.overview()
        print(f"=== Data Quality Report ===")
        print(f"Rows: {overview['rows']}, Columns: {overview['columns']}")
        print(f"Memory: {overview['memory_mb']} MB")
        missing = self.missing_values()
        if missing:
            print(f"\nMissing Values:")
            for col, info in missing.items():
                print(f"  {col}: {info['count']} ({info['pct']}%)")
        else:
            print("\nNo missing values found.")
        dups = self.duplicate_rows()
        print(f"\nDuplicate Rows: {dups['duplicate_count']} ({dups['duplicate_pct']}%)")
