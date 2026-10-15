"""The analyst tool and the farmer report must agree on a field's history.

Until 2026-10-16 src/assess_field.py averaged a field's OTHER seasons,
later ones included, while the report used only seasons harvested before
sowing. The same field could be "no record" in one tool and "history-based"
in the other, the second built partly from future harvests.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import assess_field as af   # noqa: E402
import farmer_report as fr  # noqa: E402


def test_same_history_in_both_tools():
    rows = af.load_rows()
    a = af.plot_history(rows)
    b = fr.causal_history(fr.load())
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)])


def test_history_never_uses_a_later_season():
    rows = af.load_rows()
    hist = af.plot_history(rows)
    for i, r in enumerate(rows):
        later = [j for j, q in enumerate(rows)
                 if q["plot"] == r["plot"] and j != i
                 and not q["harv"] < r["sow"]]
        prior = [j for j, q in enumerate(rows)
                 if q["plot"] == r["plot"] and q["harv"] < r["sow"]]
        if not prior:
            assert not np.isfinite(hist[i]), (
                f"{r['uid']} has no earlier harvest but a history value")
        else:
            assert np.isclose(hist[i], np.mean([rows[j]["yield"]
                                                for j in prior]))
        assert i not in later


def test_cold_start_field_is_reported_as_satellite_by_the_analyst_tool():
    """Field-season 2164 has no earlier harvest; both tools must say so."""
    out = subprocess.run([sys.executable, "-X", "utf8",
                          str(PROJ / "src/assess_field.py"), "--uid", "2164"],
                         capture_output=True, text=True, encoding="utf8",
                         cwd=str(PROJ)).stdout
    assert "[source: satellite]" in out
    assert "NOT shown to beat the area average" in out


def test_same_satellite_estimate_in_both_tools():
    """A cold-start field gets the same number from the analyst tool as from
    the report: both use farmer_report.satellite."""
    import csv
    import tempfile
    rows = fr.load()
    sat = fr.satellite(rows)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.csv"
        subprocess.run([sys.executable, "-X", "utf8",
                        str(PROJ / "src/assess_field.py"), "--split", "test",
                        "--csv", str(out)], capture_output=True, text=True,
                       encoding="utf8", cwd=str(PROJ), check=True)
        got = {r["uid"]: float(r["satellite_estimate"])
               for r in csv.DictReader(out.open(encoding="utf8"))}
    for i, r in enumerate(rows):
        if r["uid"] in got:
            assert abs(got[r["uid"]] - round(float(sat[i]), 1)) < 0.06, r["uid"]
