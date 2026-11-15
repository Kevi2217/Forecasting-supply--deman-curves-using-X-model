import numpy as np
import pandas as pd

from electricity_forecasting.data.parse_omie import parse_curve_text, parse_price_text
from electricity_forecasting.market.clearing import approximate_clearing
from electricity_forecasting.market.curves import build_curve_pair


def test_parse_to_curve_to_clearing_pipeline():
    # Synthetic fixture in the documented OMIE field layout.
    curve_text = """HEADER;\nHora;Fecha;Pais;Unidad;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);\n1;15/01/2024;MI;;V;5,0;10,0;O;\n1;15/01/2024;MI;;V;10,0;20,0;O;\n1;15/01/2024;MI;;C;8,0;30,0;O;\n1;15/01/2024;MI;;C;3,0;15,0;O;\n"""
    price_text = """MARGINALPDBC;\n2024;01;15;1;20.0;20.0;\n*\n"""
    curves = parse_curve_text(curve_text)
    prices = parse_price_text(price_text)
    ts = prices.loc[0, "interval_start_utc"]
    pair = build_curve_pair(curves, interval_start_utc=ts, country="MI", curve_flag="O")
    result = approximate_clearing(pair)
    assert result.estimated_price_eur_mwh == 20.0


def test_build_observed_clearing_table_from_fixture_file(tmp_path):
    from electricity_forecasting.data.dataset import build_observed_clearing_table

    path = tmp_path / "curva_pbc_20240101.1"
    path.write_text(
        "1;01/01/2024;ES;;C;10,0;100,0;O;\n"
        "1;01/01/2024;ES;;C;10,0;50,0;O;\n"
        "1;01/01/2024;ES;;V;10,0;0,0;O;\n"
        "1;01/01/2024;ES;;V;10,0;50,0;O;\n",
        encoding="utf-8",
    )
    out = build_observed_clearing_table([path])
    assert len(out) == 1
    assert out.iloc[0]["curve_country"] == "ES"
    assert np.isfinite(out.iloc[0]["observed_curve_clearing_eur_mwh"])
