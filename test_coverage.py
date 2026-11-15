from datetime import date
from pathlib import Path

from electricity_forecasting.data.coverage import audit_dir


def test_coverage_reports_missing_days(tmp_path: Path):
    (tmp_path / "curva_pbc_20240101.1").write_text("x")
    result = audit_dir(tmp_path, "curva_pbc", date(2024, 1, 1), date(2024, 1, 2))
    assert result["days_present"] == 1
    assert result["missing_days"] == ["20240102"]
    assert result["status"] == "FAILED"
