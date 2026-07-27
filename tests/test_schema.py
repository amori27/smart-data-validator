import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.schema_validator import Schema, SchemaValidator, ValidationResult


@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com"],
    })


class TestSchemaFromDict:
    def test_from_dict(self):
        s = Schema.from_dict({"columns": {"age": {"type": "int"}}})
        assert "age" in s.columns

    def test_add_column(self):
        s = Schema()
        s.add_column("x", type="float", min=0)
        assert s.columns["x"]["type"] == "float"


class TestSchemaValidator:
    def test_valid_data(self, simple_df):
        schema = Schema(columns={
            "age": {"type": "int", "min": 0, "max": 150},
            "name": {"type": "str"},
        })
        result = SchemaValidator(schema).validate(simple_df)
        assert result.passed

    def test_missing_column(self, simple_df):
        schema = Schema(columns={"missing_col": {"type": "int"}})
        result = SchemaValidator(schema).validate(simple_df)
        assert not result.passed
        assert "missing" in result.errors[0]["message"].lower()

    def test_type_mismatch(self, simple_df):
        schema = Schema(columns={"age": {"type": "str"}})
        result = SchemaValidator(schema).validate(simple_df)
        assert not result.passed

    def test_min_violation(self, simple_df):
        df = simple_df.copy()
        df.loc[0, "age"] = -5
        schema = Schema(columns={"age": {"min": 0}})
        result = SchemaValidator(schema).validate(df)
        assert not result.passed

    def test_max_violation(self, simple_df):
        df = simple_df.copy()
        df.loc[0, "age"] = 200
        schema = Schema(columns={"age": {"max": 150}})
        result = SchemaValidator(schema).validate(df)
        assert not result.passed

    def test_regex_valid(self, simple_df):
        schema = Schema(columns={"email": {"regex": r"^[\w.-]+@[\w.-]+\.\w+$"}})
        result = SchemaValidator(schema).validate(simple_df)
        assert result.passed

    def test_regex_invalid(self):
        df = pd.DataFrame({"email": ["not-an-email", "also-bad"]})
        schema = Schema(columns={"email": {"regex": r"^[\w.-]+@[\w.-]+\.\w+$"}})
        result = SchemaValidator(schema).validate(df)
        assert not result.passed

    def test_unique_violation(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        schema = Schema(columns={"id": {"unique": True}})
        result = SchemaValidator(schema).validate(df)
        assert not result.passed

    def test_nullable_constraint(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0]})
        schema = Schema(columns={"val": {"nullable": False}})
        result = SchemaValidator(schema).validate(df)
        assert not result.passed

    def test_choices_valid(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        schema = Schema(columns={"color": {"choices": ["red", "blue", "green", "yellow"]}})
        result = SchemaValidator(schema).validate(df)
        assert result.passed

    def test_choices_invalid(self):
        df = pd.DataFrame({"color": ["red", "purple"]})
        schema = Schema(columns={"color": {"choices": ["red", "blue", "green"]}})
        result = SchemaValidator(schema).validate(df)
        assert not result.passed

    def test_validation_result_bool(self):
        r = ValidationResult()
        assert bool(r) is True
        r.add_error("col", "err")
        assert bool(r) is False

    def test_summary(self):
        r = ValidationResult()
        r.add_error("a", "msg1")
        s = r.summary()
        assert s["error_count"] == 1
        assert not s["passed"]
