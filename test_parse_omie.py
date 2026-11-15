from datetime import date

import pandas as pd
import pytest

from electricity_forecasting.data.parse_omie import (
    _physical_interval_map,
    parse_curve_text,
    parse_omie_number,
    parse_price_text,
)


def test_parse_spanish_number_convention():
    assert parse_omie_number("2.651,8") == pytest.approx(2651.8)
    assert parse_omie_number("180,30") == pytest.approx(180.30)
    assert parse_omie_number("59.65") == pytest.approx(59.65)


def test_parse_curve_format_example():
    text = """OMEL - Mercado de electricidad;Fecha Emisión :27/02/2021 - 10:18;;28/02/2021;Mercado diario;;;;\nHora;Fecha;Pais;Unidad;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);\n1;28/02/2021;MI;;C;30,0;180,30;O;\n1;28/02/2021;MI;;V;20,0;10,50;O;\n;;;;;;;;\n"""
    df = parse_curve_text(text)
    assert len(df) == 2
    assert set(df["side"]) == {"C", "V"}
    assert df.loc[df["side"] == "C", "energy_mwh"].iloc[0] == pytest.approx(30.0)
    assert str(df["interval_start_utc"].dtype) == "datetime64[ns, UTC]"


def test_parse_price_format_example():
    text = """MARGINALPDBC;\n2011;02;05;1;59.65;59.65;\n2011;02;05;2;55.34;55.34;\n*\n"""
    df = parse_price_text(text)
    assert list(df["period"]) == [1, 2]
    assert df.loc[0, "price_es_eur_mwh"] == pytest.approx(59.65)


def test_dst_period_counts_europe_madrid():
    # EU DST spring transition: 23 physical hours.
    assert len(_physical_interval_map(date(2024, 3, 31))) == 23
    # EU DST autumn transition: 25 physical hours.
    assert len(_physical_interval_map(date(2024, 10, 27))) == 25
    assert len(_physical_interval_map(date(2024, 1, 15))) == 24
