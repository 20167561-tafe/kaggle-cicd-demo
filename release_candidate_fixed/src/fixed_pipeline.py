from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APPROVED_DATASET_PATH = Path("data/raw/titanic_train.csv")
REQUIRED_PYTHON = ((3, 10), (3, 11))
MIN_LIBRARY_VERSIONS = {
    "pandas": "3.0.0",
    "numpy": "2.4.3",
    "scikit-learn": "1.8.0",
}
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )



def check_python_version() -> None:
    lower, upper = REQUIRED_PYTHON
    current = sys.version_info[:2]
    if not (lower <= current <= upper):
        raise RuntimeError("Python must be between 3.10 and 3.11 inclusive.")



def load_dataset(dataset_path: Path = APPROVED_DATASET_PATH) -> pd.DataFrame:
    if dataset_path != APPROVED_DATASET_PATH:
        raise ValueError("Only the approved dataset path may be used.")
    dataframe = pd.read_csv(dataset_path)
    LOGGER.info("Loaded dataset from %s with shape %s", dataset_path, dataframe.shape)
    return dataframe



def impute_age_with_regression(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    predictors = ["Pclass", "SibSp", "Parch", "Fare"]
    train_rows = result["Age"].notna()
    predict_rows = result["Age"].isna()
    if predict_rows.sum() == 0:
        return result
    model = LinearRegression()
    model.fit(result.loc[train_rows, predictors], result.loc[train_rows, "Age"])
    predicted_age = model.predict(result.loc[predict_rows, predictors])
    result.loc[predict_rows, "Age"] = predicted_age
    return result



def validate_dataframe(dataframe: pd.DataFrame) -> None:
    required_columns = {
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if dataframe.empty:
        raise ValueError("Dataset must not be empty.")



def build_model_pipeline() -> Pipeline:
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )



def train_and_evaluate(dataframe: pd.DataFrame) -> float:
    cleaned = impute_age_with_regression(dataframe)
    validate_dataframe(cleaned)

    features = cleaned[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    target = cleaned["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    pipeline = build_model_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    LOGGER.info("Validation accuracy: %.4f", accuracy)
    return accuracy



def main() -> None:
    configure_logging()
    check_python_version()
    np.random.seed(42)
    dataframe = load_dataset()
    accuracy = train_and_evaluate(dataframe)
    LOGGER.info("Fixed pipeline completed successfully with accuracy %.4f", accuracy)


if __name__ == "__main__":
    main()
