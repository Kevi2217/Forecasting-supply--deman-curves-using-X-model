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


def test_malformed_curve_data_row_fails_closed():
    text = "Hora;Fecha;Pais;Unidad;Tipo Oferta;Energia;Precio;Flag;\n1;15/01/2024;MI;;C;abc;100,0;O;\n"
    with pytest.raises(ValueError, match="malformed"):
        parse_curve_text(text)


def test_vectorised_number_parsing_matches_scalar():
    from electricity_forecasting.data.parse_omie import _parse_number_series

    values = ["2.651,8", "180,30", "59.65", "1.500,00", "-500,00", "0,1", "3.000,00"]
    vec = _parse_number_series(pd.Series(values)).tolist()
    assert vec == pytest.approx([parse_omie_number(v) for v in values])
