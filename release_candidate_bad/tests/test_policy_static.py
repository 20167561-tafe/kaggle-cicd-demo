from __future__ import annotations

import ast
import re
from pathlib import Path

from src.bad_pipeline import APPROVED_DATASET_PATH

MODULE_PATH = Path("src/bad_pipeline.py")
EXPECTED_DATASET = Path("data/raw/titanic_train.csv")
ABSOLUTE_PATH_PATTERN = re.compile(r"(^[A-Za-z]:\\)|(/Users/)|(/home/)|(/tmp/)")


def test_approved_dataset_path_is_used() -> None:
    assert APPROVED_DATASET_PATH == EXPECTED_DATASET



def test_no_hard_coded_local_paths() -> None:
    tree = ast.parse(MODULE_PATH.read_text())
    string_literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    offending = [value for value in string_literals if ABSOLUTE_PATH_PATTERN.search(value)]
    assert offending == []
