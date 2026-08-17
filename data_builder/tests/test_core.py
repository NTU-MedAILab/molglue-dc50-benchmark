from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molglue_dc50.build import parse_activity_value, unit_to_nm  # noqa: E402
from molglue_dc50.context import normalize_cell_line, normalize_recruiter, normalize_target  # noqa: E402
from molglue_dc50.io import SnapshotMismatch, verify_file  # noqa: E402


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, 1.0), ("4", 4.0), ("0.5 uM", 500.0), ("200 pM", 0.2)],
)
def test_activity_units(value: object, expected: float) -> None:
    hint = "nM" if isinstance(value, (int, float)) or str(value).isdigit() else ""
    _, _, _, dc50_nm, _ = parse_activity_value(value, unit_hint=hint)
    assert dc50_nm == pytest.approx(expected)


def test_range_is_labeled_for_exclusion() -> None:
    relation, _, unit, value, note = parse_activity_value("1-3 nM")
    assert (relation, unit, value, note) == ("range", "nM", 2.0, "range_midpoint_excluded")


def test_context_normalization_examples() -> None:
    assert normalize_recruiter("Cereblon (CRBN)") == "CRBN"
    assert normalize_target("Helios / IKZF2") == "IKZF2"
    assert normalize_cell_line("HEK-293T cells") == "HEK293T"


def test_snapshot_mismatch_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_bytes(b"changed")
    expected = hashlib.sha256(b"frozen").hexdigest()
    with pytest.raises(SnapshotMismatch):
        verify_file(path, expected)


def test_unit_helper() -> None:
    assert unit_to_nm(1, "mM") == 1_000_000

