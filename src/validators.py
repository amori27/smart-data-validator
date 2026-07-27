import re
from abc import ABC, abstractmethod
from typing import Any, Callable
import pandas as pd


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, series: pd.Series) -> list[str]:
        """Return list of error messages. Empty if valid."""
        ...


class TypeValidator(BaseValidator):
    def __init__(self, expected_type: str):
        self.expected_type = expected_type
        self.type_map = {
            "int": "int64",
            "float": "float64",
            "str": "object",
            "bool": "bool",
            "datetime": "datetime64",
        }

    def validate(self, series: pd.Series) -> list[str]:
        expected = self.type_map.get(self.expected_type)
        if expected and str(series.dtype) != expected:
            return [f"Expected type {self.expected_type} ({expected}), got {series.dtype}"]
        return []


class RangeValidator(BaseValidator):
    def __init__(self, min_val: float = float("-inf"), max_val: float = float("inf")):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, series: pd.Series) -> list[str]:
        errors = []
        if not pd.api.types.is_numeric_dtype(series):
            return ["RangeValidator requires numeric data"]
        below = (series.dropna() < self.min_val).sum()
        above = (series.dropna() > self.max_val).sum()
        if below > 0:
            errors.append(f"{below} values below minimum {self.min_val}")
        if above > 0:
            errors.append(f"{above} values above maximum {self.max_val}")
        return errors


class RegexValidator(BaseValidator):
    def __init__(self, pattern: str):
        self.pattern = pattern

    def validate(self, series: pd.Series) -> list[str]:
        matches = series.dropna().astype(str).str.match(self.pattern)
        failures = (~matches).sum()
        if failures > 0:
            return [f"{failures} values don't match pattern '{self.pattern}'"]
        return []


class UniquenessValidator(BaseValidator):
    def validate(self, series: pd.Series) -> list[str]:
        dups = series.duplicated().sum()
        if dups > 0:
            return [f"{dups} duplicate values found"]
        return []


class CustomValidator(BaseValidator):
    def __init__(self, func: Callable[[pd.Series], bool], message: str):
        self.func = func
        self.message = message

    def validate(self, series: pd.Series) -> list[str]:
        if not self.func(series):
            return [self.message]
        return []


class ValidatorChain:
    def __init__(self):
        self.validators: dict[str, list[BaseValidator]] = {}

    def add(self, column: str, validator: BaseValidator) -> "ValidatorChain":
        self.validators.setdefault(column, []).append(validator)
        return self

    def validate(self, df: pd.DataFrame) -> dict[str, list[str]]:
        errors: dict[str, list[str]] = {}
        for col, validators in self.validators.items():
            if col not in df.columns:
                errors[col] = [f"Column '{col}' not found"]
                continue
            for v in validators:
                col_errors = v.validate(df[col])
                if col_errors:
                    errors.setdefault(col, []).extend(col_errors)
        return errors
