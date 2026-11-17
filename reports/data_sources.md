# Data sources and provenance

Last updated: 2026-11-16

This repository is an independent public portfolio project. It uses only public market information and independently developed code. Raw OMIE files are **not** committed to the repository.

## 1. OMIE day-ahead aggregated supply and demand curves — PRIMARY

- **Official provider:** OMI, Polo Español S.A. (OMIE)
- **Dataset:** Aggregate supply and demand curves of Day-ahead market
- **Official directory:** https://www.omie.es/en/file-access-list?dir=Aggregate+supply+and+demand+curves+of+Day-ahead+market&parents=%2FDay-ahead+Market%2F3.+Curves&realdir=curva_pbc
- **Observed naming convention:** `curva_pbc_YYYYMMDD.N` (N = revision; `.1` for all 731 days in 2023–2024)
- **Target coverage:** 2023-01-01 through 2024-12-31 (production pipeline). An extension adds 2025-01-01 – 2025-09-30, the last hourly period before the 15-minute MTU, as a confirmation sample.
- **Coverage status:** VERIFIED. 731/731 days for 2023–2024 and 273/273 days for 2025-01 – 2025-09, all revision `.1`. See `reports/metrics/raw_coverage.json` and `reports/extension/raw_coverage.json`.
- **Schema change on 2025-03-19 (verified on real files):** the header becomes `Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C);Tipología de Oferta;`.
  - Periods remain hourly integers, so MW × 1 h = MWh.
  - The new 9th column labels rows `S` (simple), `C01`/`C02`/`C04` (offers with complex conditions; codes not decoded from official documentation), and coupling rows in matched curves (`Imp FR`, `Exp FR`, `Imp/Exp ES/PT`).
  - The parser keeps this column as `offer_type`; it is `None` for earlier files.
- **2025-10-01 onwards:** quarter-hour periods (`H1Q1`, …). This is out of scope, and the parser fails closed on it.
- **Download route used by OMIE File Access links:** `https://www.omie.es/en/file-download?filename=<filename>&parents=curva_pbc`
- **Documented file format for the 2023–2024 hourly regime:** public text records separated by semicolons. The public-format documentation identifies fields for hour, date, country, unit, bid side, energy, price, and offered/matched flag.
- **Relevant variables:**
  - Hour / period: 1–25 for hourly day-ahead files
  - Date: DD/MM/YYYY
  - Country: `MI` (MIBEL), `ES`, `PT`
  - Unit: blank in the non-unit public curve file
  - Bid side: `C` = purchase, `V` = sale
  - Energy: MWh
  - Price: EUR/MWh
  - Curve flag: `O` = offered curve, `C` = matched programme
- **Ordering documented by OMIE:** purchase records are ordered from higher to lower price; sale records from lower to higher price. Multiple records may occur at the same price and must not be treated as cumulative values.
- **Project interpretation:** the raw energy field is treated as an incremental contribution. Cumulative supply and demand functions are constructed by summing increments at/through price levels. This must still be validated against a real downloaded file before empirical use.
- **Cleaning planned:** decode robustly, parse locale-formatted numbers, retain source filename, aggregate same-price increments, preserve 23/24/25-hour days, reject non-finite/negative quantities, and record country/curve flags.
- **Missing observations:** none (machine-verified).

### Official format documentation used

OMIE, *Modelo de Ficheros para la distribución pública de Información del mercado de electricidad*, version 1.35 (13 June 2024):
https://www.omie.es/sites/default/files/2024-06/Formato_ficheros_inf_pub_135_1.pdf

This version is appropriate to the 2023–2024 hourly market representation. It states the general public-file convention and the `curva_pbc` schema.

## 2. OMIE day-ahead marginal prices — PRIMARY TARGET

- **Official provider:** OMIE
- **Dataset:** Day-ahead market hourly prices in Spain (`MARGINALPDBC`)
- **Official directory:** https://www.omie.es/en/file-access-list?dir=+Day-ahead+market+hourly+prices+in+Spain&parents=%2FDay-ahead+Market%2F1.+Prices&realdir=marginalpdbc
- **Observed naming convention:** `marginalpdbc_YYYYMMDD.N` (N = revision; 2023-05-28 is published only as `.2`)
- **Target coverage:** 2023-01-01 through 2024-12-31, plus the 2025-01-01 – 2025-09-30 extension.
- **Coverage status:** VERIFIED: 731/731 and 273/273 days. From 2025-10-01 the file has 96 quarter-hour periods, which is out of scope.
- **Download route:** `https://www.omie.es/en/file-download?filename=<filename>&parents=marginalpdbc`
- **Documented variables:** year, month, day, hour/period, Portuguese marginal price, Spanish marginal price.
- **Units:** EUR/MWh.
- **Primary target:** Spanish marginal price (`MarginalES`).
- **Alignment key:** physical delivery interval, represented internally by a timezone-aware UTC interval start plus local delivery date/period.

The 2024 OMIE integrated report confirms that the 2024 day-ahead market still used 24 hourly products on ordinary days, with the 15-minute transition planned for 2025. The project therefore does not mix post-transition quarter-hour data into the 2023–2024 sample.

## 3. OMIE monthly curves including bid units — SECONDARY

- **Official provider:** OMIE
- **Dataset:** Monthly files with aggregate supply and demand curves of Day-ahead market including bid units
- **Official directory:** https://www.omie.es/en/file-access-list?dir=Ficheros+mensuales+con+curvas+agregadas+de+oferta+y+demanda+del+mercado+diario+incluyendo+unidades+de+oferta&parents=%2FMercado+Diario%2F3.+Curvas&realdir=curva_pbc_uof
- **Observed naming convention:** `curva_pbc_uof_YYYYMM.zip`
- **Verified availability:** the directory contains monthly archives including `curva_pbc_uof_202301.zip` through at least early/mid 2024 files visible in the current listing.
- **Purpose:** interpretation checks and deeper bid-structure diagnostics only. The MVP does not depend on bid-unit-level information.
- **Confidentiality timing:** OMIE documentation states these unit-labelled curves are published after the applicable confidentiality period.

## 4. ENTSO-E Transparency Platform — OPTIONAL

- **Provider:** ENTSO-E
- **Website:** https://transparency.entsoe.eu/
- **Candidate variables:** day-ahead load forecast, wind generation forecast, solar generation forecast.
- **Status:** NOT VERIFIED / NOT INGESTED for the MVP yet.
- **Rule:** only historical forecast vintages demonstrably available before the chosen day-ahead forecast origin may be used. Realised load/generation must not be substituted for unavailable forecasts.

## Market timing and timestamp convention

OMIE states that the day-ahead market session closes at **12:00 CET** for delivery on the following day. For 2023–2024 the product is hourly. The parser represents physical intervals using `Europe/Madrid` civil-time transitions and stores UTC interval identifiers internally. This implementation handles 23-hour and 25-hour DST days, but the exact raw-file convention remains a real-data validation item.

Official market overview: https://www.omie.es/en/mercado-de-electricidad

## Licence / reuse

OMIE's legal notice states that information on the public website which is public and free of charge may be used freely provided the original content is respected, and that OMIE should be cited when information is used in its original format. OMIE also disclaims responsibility for downstream use and may change site structure without prior notice.

Official notice: https://www.omie.es/en/legal-warning

**Repository policy:** do not redistribute bulk raw OMIE files by default. Store source URLs, retrieval dates and checksums locally; publish code and derived research outputs with clear source attribution. This is a conservative project policy rather than a claim that OMIE prohibits redistribution.

## Verification status

The earlier build environment could not download the binary files. In a normal internet-connected Python environment the official download endpoint works directly via `requests`. As of 2026-11-16:

- official file existence and schema: **VERIFIED** on real files (2026-11-16);
- parser/downloader implementation: **VERIFIED** on all 1 462 files for 2023–2024;
- inspection of representative real raw observations: **VERIFIED** (encoding ISO-8859-1, CRLF, `;` separator; Spanish number format in curve files; Spain = column 6 of `marginalpdbc`);
- full 2023–2024 completeness audit: **VERIFIED**, 731/731 days per dataset;
- 2025-01 – 2025-09 extension: **VERIFIED**, 273/273 days per dataset, all files pass schema and DST validation (`reports/extension/data_quality.json`);
- empirical curve-clearing validation: **EXECUTED, gate FAILED**; see `results.md`.
