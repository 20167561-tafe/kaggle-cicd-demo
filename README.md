# Kaggle CI/CD Demo for a Python Data Science Project

This package contains two self-contained example projects based on the **Kaggle Titanic** workflow:

- `release_candidate_good/` — a clean release candidate that is designed to pass CI checks.
- `release_candidate_bad/` — an intentionally flawed release candidate that should fail CI and must be fixed before production release.
- `release_candidate_fixed/` — the remediated version of the bad candidate, updated to satisfy the same CI policy.

Both projects implement the same idea:

- load an approved Titanic-style dataset
- validate the environment
- preprocess the data
- impute missing values using a **regression approach**
- train a small scikit-learn model
- run automated CI checks for code quality, typing, dataset policy, path policy, and preprocessing logic

## Real-life scenario

A programmer in your team modifies a customer-risk or churn model pipeline. Before release, the CI pipeline should verify:

1. code style and typing are clean
2. only the approved dataset path is used
3. no hard-coded local machine paths are present
4. Python is between 3.10 and 3.11 inclusive
5. approved library versions are installed
6. missing-value handling for the target feature uses **regression imputation**, not mean imputation
7. the project runs on a clean machine via GitHub Actions

## Dataset basis

The project is based on the **Kaggle Titanic competition**. A small Titanic-like CSV fixture is included so the CI pipeline can run without requiring Kaggle authentication.

## Current package/version references used for the policy

The environment policy is aligned to current official package sources and reflected in the requirements and tests shipped with each project.

## Suggested demo steps

### Good release candidate

```bash
cd release_candidate_good
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
ruff check .
mypy src tests
pytest -q
python -m src.good_pipeline
```

### Fixed release candidate

```bash
cd release_candidate_fixed
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
ruff check .
mypy src tests
pytest -q
python -m src.fixed_pipeline
```

### Bad release candidate

```bash
cd release_candidate_bad
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
ruff check .
mypy src tests
pytest -q
python -m src.bad_pipeline
```

The bad candidate is expected to fail CI until fixed.
