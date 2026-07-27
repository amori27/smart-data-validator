# smart-data-validator

![CI](https://github.com/amori27/smart-data-validator/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A flexible data validation framework for pandas DataFrames and CSV files.

## Features

- **Schema Validation** - Define and enforce column types, constraints, and relationships
- **Anomaly Detection** - Z-score, IQR, and isolation-based outlier detection
- **Quality Reports** - Comprehensive data quality metrics and summaries
- **Built-in Validators** - Type, range, regex, uniqueness, and custom validators

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd
from src.schema_validator import SchemaValidator, Schema
from src.anomaly_detector import AnomalyDetector
from src.quality_report import QualityReport

# Validate schema
schema = Schema(
    columns={
        "age": {"type": "int", "min": 0, "max": 150},
        "email": {"type": "str", "regex": r"^[\w.-]+@[\w.-]+\.\w+$"},
    }
)
validator = SchemaValidator(schema)
result = validator.validate(pd.DataFrame({"age": [25, 30], "email": ["a@b.com", "c@d.com"]}))

# Detect anomalies
detector = AnomalyDetector()
outliers = detector.zscore_outliers(df["value"], threshold=3.0)

# Generate quality report
report = QualityReport(df)
report.print_summary()
```

## Validators

| Validator | Description |
|-----------|-------------|
| Type | Check column data types |
| Range | Min/max value constraints |
| Regex | Pattern matching for strings |
| Uniqueness | Ensure no duplicate values |
| Custom | User-defined validation functions |

## License

MIT
