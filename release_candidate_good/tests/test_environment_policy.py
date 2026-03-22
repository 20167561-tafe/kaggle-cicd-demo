from __future__ import annotations
#
import sys
from importlib.metadata import version
from packaging.version import Version

MIN_VERSIONS = {
    "pandas": "3.0.0",
    "numpy": "2.4.3",
    "scikit-learn": "1.8.0",
}


def test_python_version_is_between_310_and_311() -> None:
    assert (3, 10) <= sys.version_info[:2] <= (3, 11)



def test_library_versions_meet_policy() -> None:
    for package_name, min_version in MIN_VERSIONS.items():
        assert Version(version(package_name)) >= Version(min_version)
