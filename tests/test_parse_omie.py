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


def test_offer_type_column_is_parsed_when_present() -> None:
    from electricity_forecasting.data.parse_omie import parse_curve_text

    text = (
        "OMIE - Mercado de electricidad;Fecha Emisión :29/09/2025 - 11:56;;30/09/2025;Mercado diario;;;;\n"
        "\n"
        "Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia Compra/Venta;Precio Compra/Venta;"
        "Ofertada (O)/Casada (C);Tipología de Oferta;\n"
        "1;30/09/2025;MI;;V;100,0;10,00;O;S;\n"
        "1;30/09/2025;MI;;V;50,0;20,00;O;C02;\n"
        "1;30/09/2025;MI;;V;30,0;80,00;C;Imp FR;\n"
    )
    frame = parse_curve_text(text)
    assert list(frame["offer_type"]) == ["S", "C02", "Imp FR"]
    assert frame["energy_mwh"].tolist() == [100.0, 50.0, 30.0]


def test_offer_type_absent_in_legacy_files() -> None:
    from electricity_forecasting.data.parse_omie import parse_curve_text

    text = (
        "Hora;Fecha;Pais;Unidad;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);\n"
        "1;15/01/2024;MI;;C;1.500,0;3.000,00;O;\n"
    )
    frame = parse_curve_text(text)
    assert frame["offer_type"].isna().all()
    assert frame["energy_mwh"].iloc[0] == 1500.0
