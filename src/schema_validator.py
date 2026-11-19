import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import pandas as pd


@dataclass
class ColumnSchema:
    name: str
    type: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    regex: Optional[str] = None
    unique: bool = False
    nullable: bool = True
    choices: Optional[list] = None


@dataclass
class Schema:
    columns: dict[str, dict] = field(default_factory=dict)

    def add_column(self, name: str, **kwargs) -> None:
        self.columns[name] = kwargs

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(columns=d.get("columns", {}))

    @classmethod
    def from_yaml(cls, path: str) -> "Schema":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class ValidationResult:
    def __init__(self):
        self.errors: list[dict] = []
        self.passed: bool = True

    def add_error(self, column: str, message: str, row: Any = None) -> None:
        self.errors.append({"column": column, "message": message, "row": row})
        self.passed = False

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "error_count": len(self.errors),
            "errors": self.errors[:10],
        }


class SchemaValidator:
    def __init__(self, schema: Schema):
        self.schema = schema

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult()
        for col_name, constraints in self.schema.columns.items():
            if col_name not in df.columns:
                result.add_error(col_name, "Column missing from DataFrame")
                continue
            col = df[col_name]
            col_type = constraints.get("type")
            if col_type:
                self._check_type(col, col_name, col_type, result)
            if not constraints.get("nullable", True):
                self._check_nullable(col, col_name, result)
            if "min" in constraints:
                self._check_min(col, col_name, constraints["min"], result)
            if "max" in constraints:
                self._check_max(col, col_name, constraints["max"], result)
            if "regex" in constraints:
                self._check_regex(col, col_name, constraints["regex"], result)
            if constraints.get("unique"):
                self._check_unique(col, col_name, result)
            if "choices" in constraints:
                self._check_choices(col, col_name, constraints["choices"], result)
        return result

    def _check_type(self, col, name, expected, result):
        type_map = {"int": "int64", "float": "float64", "str": "object", "bool": "bool"}
        expected_dtype = type_map.get(expected)
        if not expected_dtype:
            return
        if expected == "str":
            # pandas >= 3 uses the "str" dtype for text columns; older
            # versions used "object". Both represent valid string data.
            valid = str(col.dtype) in ("object", "str") or pd.api.types.is_string_dtype(col)
        else:
            valid = str(col.dtype) == expected_dtype
        if not valid:
            result.add_error(name, f"Expected type {expected}, got {col.dtype}")

    def _check_nullable(self, col, name, result):
        null_count = col.isna().sum()
        if null_count > 0:
            result.add_error(name, f"Column has {null_count} null values")

    def _check_min(self, col, name, min_val, result):
        if pd.api.types.is_numeric_dtype(col):
            violations = (col.dropna() < min_val).sum()
            if violations > 0:
                result.add_error(name, f"{violations} values below min {min_val}")

    def _check_max(self, col, name, max_val, result):
        if pd.api.types.is_numeric_dtype(col):
            violations = (col.dropna() > max_val).sum()
            if violations > 0:
                result.add_error(name, f"{violations} values above max {max_val}")

    def _check_regex(self, col, name, pattern, result):
        matches = col.dropna().astype(str).str.match(pattern)
        violations = (~matches).sum()
        if violations > 0:
            result.add_error(name, f"{violations} values don't match regex {pattern}")

    def _check_unique(self, col, name, result):
        duplicates = col.duplicated().sum()
        if duplicates > 0:
            result.add_error(name, f"{duplicates} duplicate values found")

    def _check_choices(self, col, name, choices, result):
        invalid = ~col.dropna().isin(choices)
        if invalid.any():
            result.add_error(name, f"{invalid.sum()} values not in allowed choices")
