from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from electricity_forecasting.market.clearing_audit import audit_interval

TS = pd.Timestamp("2025-09-30 10:00", tz="UTC")


def _rec(side: str, flag: str, energy: float, price: float, otype: str | None = None) -> dict:
    return {"country": "MI", "side": side, "curve_flag": flag, "energy_mwh": energy, "price_eur_mwh": price, "offer_type": otype}


def _rows() -> pd.DataFrame:
    # Offered: cheap simple supply, a cheap complex offer that is rejected, and expensive supply.
    # Matched: the simple cheap supply plus a coupling import at the clearing price.
    return pd.DataFrame([
        _rec("V", "O", 100, 10, "S"),
        _rec("V", "O", 100, 20, "C02"),
        _rec("V", "O", 100, 50, "S"),
        _rec("C", "O", 250, 100, "S"),
        _rec("V", "C", 100, 10, "S"),
        _rec("V", "C", 100, 50, "S"),
        _rec("V", "C", 50, 50, "Imp FR"),
        _rec("C", "C", 250, 100, "S"),
    ])


def test_audit_interval_identifies_rejected_complex_supply_and_coupling() -> None:
    out = audit_interval(_rows(), ts=TS, official_price=50.0)
    assert out["p_offered"] == 50.0
    assert out["p_matched"] == 50.0
    # 300 MWh offered at <= 50, 250 MWh matched (incl. 50 MWh import) -> 50 MWh net unmatched.
    assert out["supply_unmatched_at_or_below_p_mwh"] == pytest.approx(50.0)
    assert out["complex_supply_unmatched_at_or_below_p_mwh"] == pytest.approx(100.0)
    assert out["matched_coupling_supply_mwh"] == pytest.approx(50.0)
    assert out["simple_supply_unmatched_strictly_below_p_mwh"] == pytest.approx(0.0)
    assert out["matched_supply_above_p_mwh"] == 0.0


def test_audit_interval_offered_crossing_is_below_official_when_cheap_offer_rejected() -> None:
    rows = _rows()
    rows.loc[rows.index[3], "energy_mwh"] = 180  # offered demand 180 -> crossing at 20 on offered curves
    rows.loc[rows.index[7], "energy_mwh"] = 180
    out = audit_interval(rows, ts=TS, official_price=50.0)
    assert out["p_offered"] == 20.0
    assert out["p_offered"] < out["price_es_eur_mwh"]


def test_audit_interval_without_typology_column() -> None:
    rows = _rows().drop(columns="offer_type")
    out = audit_interval(rows, ts=TS, official_price=50.0)
    assert out["has_offer_type"] is False
    assert "complex_supply_unmatched_at_or_below_p_mwh" not in out


def _load_extension_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_curve_extension.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("run_curve_extension", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trailing_median_uses_only_previous_days() -> None:
    ext = _load_extension_module()
    days = pd.date_range("2024-01-01", periods=10, freq="D")
    hist = pd.DataFrame({"delivery_date": days, "hour": 12, "discrepancy": np.arange(10, dtype=float)})
    target = pd.DataFrame({"delivery_date": [days[5]], "hour": [12]})
    # Window of 3 days before 2024-01-06: discrepancies 2, 3, 4 (same-day value 5 excluded).
    assert ext.trailing_hour_median(hist, target, 3)[0] == pytest.approx(3.0)


def test_adjustment_curve_state_uses_forecast_not_observed_curves() -> None:
    ext = _load_extension_module()
    frame = pd.DataFrame({"compressed_curve_clearing_eur_mwh": [99.0], "supply_00": [1.0], "demand_00": [2.0]})
    fc = ext.forecast_curve_state(frame, pd.Series([42.0]), pd.DataFrame({"s": [5.0]}), pd.DataFrame({"d": [7.0]}))
    assert fc[["curve_price", "curve_supply_total", "curve_demand_total"]].iloc[0].tolist() == [42.0, 5.0, 7.0]
    obs = ext.observed_curve_state(frame, ["supply_00"], ["demand_00"])
    assert obs[["curve_price", "curve_supply_total", "curve_demand_total"]].iloc[0].tolist() == [99.0, 1.0, 2.0]


def test_adjustment_features_contain_no_same_day_post_auction_values() -> None:
    ext = _load_extension_module()
    cols = ext.adjustment_feature_columns([1, 2, 7])
    post_auction = {"discrepancy", "unmatched_supply", "demand_gap", "price_es_eur_mwh", "observed_curve_clearing_eur_mwh",
                    "compressed_curve_clearing_eur_mwh"}
    assert not post_auction & set(cols)
    assert all("_lag_" in c for c in cols if c.split("_lag_")[0] in {"discrepancy", "unmatched_supply", "demand_gap"})


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dm_test_detects_consistent_loss_difference_and_sign() -> None:
    dm = _load_script("extension_significance")
    rng = np.random.default_rng(0)
    better = -1.0 + 0.5 * rng.standard_normal(200)  # model A has lower loss every day on average
    stat, p = dm.dm_test(better, lag=7)
    assert stat < 0 and p < 0.01
    stat0, p0 = dm.dm_test(rng.standard_normal(200), lag=7)
    assert p0 > 0.01


def test_decomposition_identities_hold_exactly() -> None:
    ext = _load_extension_module()
    rng = np.random.default_rng(1)
    n = 50
    off = rng.normal(80, 20, n)
    frame = pd.DataFrame({
        "price_es_eur_mwh": off,
        "observed_curve_clearing_eur_mwh": off - 20 + rng.normal(0, 5, n),
        "compressed_curve_clearing_eur_mwh": off - 18 + rng.normal(0, 5, n),
        "curve_elastic_net_prediction": off - 15 + rng.normal(0, 10, n),
        "adjustment_elastic_net": rng.normal(15, 3, n),
        "bias_constant_elastic_net": 14.0,
        "obs_adjustment_elastic_net": rng.normal(16, 3, n),
    })
    rows = {r["component"]: r for r in ext.decomposition(frame, "elastic_net")}
    s = lambda key: next(v["signed_mean"] for k, v in rows.items() if k.startswith(key))  # noqa: E731
    assert s("1_") + s("2_") + s("3_") == pytest.approx(s("total_unadjusted"))
    assert s("3_") + s("4_adjustment_estimation[oof_g]") == pytest.approx(s("total_adjusted[oof_g]"))
    # MAE is NOT additive in general.
    mae = lambda key: next(v["mae"] for k, v in rows.items() if k.startswith(key))  # noqa: E731
    assert mae("total_unadjusted") != pytest.approx(mae("1_") + mae("2_") + mae("3_"))
