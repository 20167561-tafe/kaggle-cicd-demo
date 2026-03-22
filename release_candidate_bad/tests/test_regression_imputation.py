from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.bad_pipeline import impute_age_with_regression

MODULE_PATH = Path("src/bad_pipeline.py")


def test_age_imputation_uses_regression_class() -> None:
    source = MODULE_PATH.read_text()
    tree = ast.parse(source)
    imported_names = {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "sklearn.linear_model"}
    assert "LinearRegression" in imported_names



def test_age_imputation_matches_regression_signal() -> None:
    dataframe = pd.DataFrame(
        {
            "Pclass": [1, 2, 3, 4, 5],
            "SibSp": [0, 0, 0, 0, 0],
            "Parch": [0, 0, 0, 0, 0],
            "Fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Age": [20.0, 30.0, 40.0, np.nan, np.nan],
        }
    )
    result = impute_age_with_regression(dataframe)
    assert np.isclose(result.loc[3, "Age"], 50.0)
    assert np.isclose(result.loc[4, "Age"], 60.0)
