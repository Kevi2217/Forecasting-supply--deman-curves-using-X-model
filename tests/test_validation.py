from electricity_forecasting.data.parse_omie import parse_curve_text, parse_price_text
from electricity_forecasting.data.validate import validate_curve_records, validate_price_records


def test_documented_format_fixtures_validate():
    curve_text = """HEADER;\nHora;Fecha;Pais;Unidad;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);\n1;15/01/2024;MI;;C;10,0;100,0;O;\n1;15/01/2024;MI;;V;10,0;0,0;O;\n"""
    price_text = """MARGINALPDBC;\n2024;01;15;1;50.0;50.0;\n*\n"""
    c = validate_curve_records(parse_curve_text(curve_text))
    p = validate_price_records(parse_price_text(price_text))
    # These are partial-day fixtures, so validators correctly flag unexpected daily counts.
    assert c.status == "FAILED"
    assert p.status == "FAILED"
    assert any("Unexpected" in issue for issue in c.issues)
    assert any("Unexpected" in issue for issue in p.issues)


def test_dst_day_with_wrong_period_count_is_rejected():
    # 2024-03-31 is a 23-hour day in Europe/Madrid; 24 rows would be invalid but the
    # parser rejects period 24 directly. A 25-hour day (2024-10-27) given only 24 periods
    # must be caught by the validator.
    rows = "".join(f"2024;10;27;{h};50.0;50.0;\n" for h in range(1, 25))
    p = validate_price_records(parse_price_text("MARGINALPDBC;\n" + rows + "*\n"))
    assert p.status == "FAILED"
    assert any("DST-aware" in issue for issue in p.issues)
