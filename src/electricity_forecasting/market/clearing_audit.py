"""Ex-post audit of why naive crossing of OMIE *offered* curves misses the official price.

Everything here uses post-auction information (matched curves, official prices, offer
typology). It is a diagnostic and a source of *historical* training targets / lagged
summaries only. Same-day values computed here must never be used as forecasting
features for that day; callers lag them by at least one delivery day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_forecasting.market.clearing import approximate_clearing
from electricity_forecasting.market.curves import CurvePair, build_demand_curve, build_supply_curve

COUPLING_PREFIXES = ("IMP", "EXP")


def _cross(ts: pd.Timestamp, country: str, supply: pd.DataFrame, demand: pd.DataFrame) -> float:
    if supply.loc[supply["side"] == "V"].empty or demand.loc[demand["side"] == "C"].empty:
        return float("nan")
    pair = CurvePair(ts, country, build_supply_curve(supply), build_demand_curve(demand))
    value = approximate_clearing(pair).estimated_price_eur_mwh
    return float("nan") if value is None else float(value)


def _vol(frame: pd.DataFrame, mask: pd.Series | None = None) -> float:
    if mask is not None:
        frame = frame.loc[mask]
    return float(frame["energy_mwh"].sum())


def audit_interval(
    rows: pd.DataFrame,
    *,
    ts: pd.Timestamp,
    official_price: float,
    preferred_country: str = "ES",
    fallback_country: str = "MI",
) -> dict[str, object]:
    """Offered vs matched curve diagnostics for one delivery interval."""
    countries = set(rows.loc[rows["curve_flag"] == "O", "country"])
    country = preferred_country if preferred_country in countries else fallback_country
    g = rows.loc[rows["country"] == country]
    off, mat = g.loc[g["curve_flag"] == "O"], g.loc[g["curve_flag"] == "C"]
    oS, oD = off.loc[off["side"] == "V"], off.loc[off["side"] == "C"]
    mS, mD = mat.loc[mat["side"] == "V"], mat.loc[mat["side"] == "C"]
    p = float(official_price)
    eps = 1e-9

    out: dict[str, object] = {
        "interval_start_utc": ts,
        "curve_country": country,
        "price_es_eur_mwh": p,
        # Crossings: offered/offered (the gated method), matched/matched, and the two mixes.
        "p_offered": _cross(ts, country, oS, oD),
        "p_matched": _cross(ts, country, mS, mD),
        "p_matched_supply_offered_demand": _cross(ts, country, mS, oD),
        "p_offered_supply_matched_demand": _cross(ts, country, oS, mD),
        # Consistency of the matched curves with a uniform clearing price.
        "matched_supply_max_price": float(mS["price_eur_mwh"].max()) if len(mS) else np.nan,
        "matched_demand_min_price": float(mD["price_eur_mwh"].min()) if len(mD) else np.nan,
        "matched_supply_above_p_mwh": _vol(mS, mS["price_eur_mwh"] > p + eps),
        "matched_demand_below_p_mwh": _vol(mD, mD["price_eur_mwh"] < p - eps),
        "matched_supply_mwh": _vol(mS),
        "matched_demand_mwh": _vol(mD),
        "offered_supply_mwh": _vol(oS),
        "offered_demand_mwh": _vol(oD),
        "offered_supply_at_or_below_p_mwh": _vol(oS, oS["price_eur_mwh"] <= p + eps),
        "offered_supply_below_p_mwh": _vol(oS, oS["price_eur_mwh"] < p - eps),
        "offered_supply_at_or_below_0_mwh": _vol(oS, oS["price_eur_mwh"] <= 0.0),
        "offered_demand_at_or_above_p_mwh": _vol(oD, oD["price_eur_mwh"] >= p - eps),
    }
    out["supply_unmatched_at_or_below_p_mwh"] = out["offered_supply_at_or_below_p_mwh"] - out["matched_supply_mwh"]
    out["supply_unmatched_strictly_below_p_mwh"] = out["offered_supply_below_p_mwh"] - _vol(
        mS, mS["price_eur_mwh"] < p - eps
    )
    out["demand_matched_not_in_offered_mwh"] = out["matched_demand_mwh"] - out["offered_demand_at_or_above_p_mwh"]

    has_type = "offer_type" in g.columns and g["offer_type"].notna().any()
    out["has_offer_type"] = bool(has_type)
    if has_type:
        otype = g["offer_type"].fillna("").astype(str).str.upper()
        is_cpl = otype.str.startswith(COUPLING_PREFIXES)
        is_simple = otype.eq("S")
        is_complex = otype.str.match(r"^C\d")
        o_s = off.index.isin(g.index[is_simple])
        o_cx = off.index.isin(g.index[is_complex])
        oS_simple = oS.loc[oS.index.isin(g.index[is_simple])]
        oS_cx = oS.loc[oS.index.isin(g.index[is_complex])]
        mS_cx = mS.loc[mS.index.isin(g.index[is_complex])]
        mS_cpl = mS.loc[mS.index.isin(g.index[is_cpl])]
        mD_cpl = mD.loc[mD.index.isin(g.index[is_cpl])]
        mS_simple = mS.loc[mS.index.isin(g.index[is_simple])]
        out.update(
            {
                "offered_rows_simple": int(o_s.sum()),
                "offered_rows_complex": int(o_cx.sum()),
                "offered_complex_supply_at_or_below_p_mwh": _vol(oS_cx, oS_cx["price_eur_mwh"] <= p + eps),
                "matched_complex_supply_mwh": _vol(mS_cx),
                "offered_simple_supply_strictly_below_p_mwh": _vol(oS_simple, oS_simple["price_eur_mwh"] < p - eps),
                "matched_simple_supply_strictly_below_p_mwh": _vol(mS_simple, mS_simple["price_eur_mwh"] < p - eps),
                "matched_coupling_supply_mwh": _vol(mS_cpl),
                "matched_coupling_demand_mwh": _vol(mD_cpl),
                # Counterfactual crossings (diagnostic only; use post-auction information).
                "p_simple_supply_offered_demand": _cross(ts, country, oS_simple, oD),
                "p_offered_plus_coupling": _cross(
                    ts, country, pd.concat([oS, mS_cpl]), pd.concat([oD, mD_cpl])
                ),
                "p_simple_plus_coupling": _cross(
                    ts, country, pd.concat([oS_simple, mS_cpl]), pd.concat([oD, mD_cpl])
                ),
            }
        )
        unmatched_cx = out["offered_complex_supply_at_or_below_p_mwh"] - out["matched_complex_supply_mwh"]
        out["complex_supply_unmatched_at_or_below_p_mwh"] = unmatched_cx
        out["simple_supply_unmatched_strictly_below_p_mwh"] = (
            out["offered_simple_supply_strictly_below_p_mwh"] - out["matched_simple_supply_strictly_below_p_mwh"]
        )
    return out


def audit_frame(
    curve_frame: pd.DataFrame,
    prices: pd.Series,
    *,
    preferred_country: str = "ES",
    fallback_country: str = "MI",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ts, group in curve_frame.groupby("interval_start_utc", sort=True):
        rows.append(
            audit_interval(
                group, ts=pd.Timestamp(ts), official_price=float(prices.loc[ts]),
                preferred_country=preferred_country, fallback_country=fallback_country,
            )
        )
    return rows
