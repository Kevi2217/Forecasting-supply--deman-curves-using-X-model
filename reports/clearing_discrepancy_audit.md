# Clearing discrepancy audit

Run date: 2026-11-16. Data: official OMIE `curva_pbc` and `marginalpdbc` files, 2023-01-01 – 2025-09-30. That is 24 095 hourly intervals and 1 004 + 1 004 files, all of which pass the coverage, schema and DST gates.

Scripts and outputs:
- `scripts/audit_clearing_mechanism.py`, which uses `src/electricity_forecasting/market/clearing_audit.py`.
- Per-interval table: `data/processed/clearing_mechanism_audit.csv`.
- Summary: `reports/metrics/clearing_mechanism_audit.json`.

**The quantities in this audit use post-auction information:** matched curves, official prices, and offer typology. They are diagnostics. In the adjusted forecasting model they appear only as **historical training targets or as values lagged by at least one delivery day**.

## 1. Curve definitions and geographical scope (rechecked on raw bytes)

| Item | Finding |
|---|---|
| Header | `Hora;Fecha;Pais;Unidad;Tipo Oferta;Energía Compra/Venta;Precio Compra/Venta;Ofertada (O)/Casada (C)` |
| `Unidad` | Always empty. The aggregate curves are anonymised, so bids cannot be linked to units or conditions in 2023–2024. |
| `Tipo Oferta` | `C` = purchase (demand), `V` = sale (supply) |
| `Ofertada (O)/Casada (C)` | `O` = all offered bids; `C` = the matched (accepted) segments |
| `Pais` | `MI` (Iberian) in unsplit hours; `ES` and `PT` only in market-split hours. Verified as before: 0 intervals with an `MI` curve during an ES/PT price split. |
| **Schema change on 2025-03-19** | The header becomes `Periodo;…;Potencia Compra/Venta;…;Tipología de Oferta`. Periods are still hourly integers (1–23/24/25), so MW × 1 h = MWh. A 9th column labels each row: `S` (simple), `C01`/`C02`/`C04` (offers with complex conditions), and coupling rows in matched curves (`Imp FR`, `Exp FR`, and `Imp/Exp ES/PT` in split hours). |
| 2025-10-01 | Periods become quarter-hours (`H1Q1`…). This is outside the study, and the parser fails closed on it. |

The parser was extended to keep the optional `offer_type` column; legacy files give `None`. Every 2023–2025 file still validates.

Code meanings: I could not find an official public description of `C01`, `C02` and `C04`. CNMC's consultation on the 15-minute-MTU market rules describes a new day-ahead offer typology that *replaces* the previous complex (minimum-income) offers. The codes are therefore used here only as "non-simple offer", and no specific condition is attributed to any code.

## 2. Offered and matched interpretation: verifying the implementation

The **same** stepwise clearing function was applied to different curve combinations and compared with the official Spanish price:

| Supply curve × demand curve | MAE | Bias | Share \|err\| < 1 |
|---|---|---|---|
| offered × offered (the gated method) | 24.64 | −23.21 | 0.11 |
| **matched × matched** | **0.43** | −0.39 | **0.87** |
| matched × offered | 5.63 | −5.37 | 0.52 |
| offered × matched | 35.07 | −33.37 | 0.13 |

The pattern holds in every year:

| Year | offered × offered | matched × matched | matched × offered | offered × matched |
|---|---|---|---|---|
| 2023 | 32.27 | 0.45 | 7.49 | 46.17 |
| 2024 | 20.02 | 0.35 | 3.60 | 29.99 |
| 2025 (Jan–Sep) | 20.63 | 0.51 | 5.85 | 27.04 |

Uniform-price consistency of the matched curves:
- The official price lies within [highest matched sell price, lowest matched buy price] in **99.5 %** of intervals.
- No matched demand is ever priced below p\*.
- In 113 intervals (0.5 %), a small amount of matched supply (at most ~130 MWh, 6 963 MWh in total) is priced above p\*. This is consistent with intertemporal unit conditions, but **not verified**.
- Matched supply equals matched demand to within 1 MWh in all but 24 intervals.

**Conclusion.** The clearing implementation and the curve interpretation are correct: applied to the matched curves, the method reproduces the official price to within the step-function indeterminacy (median AE 0). The discrepancy comes from the **content of the offered curves**, not from parsing, the geography choice or the crossing algorithm.

## 3. Market constraints vs data representation vs implementation

| Candidate explanation | Evidence | Verdict |
|---|---|---|
| Implementation (crossing rule, parsing, units) | Matched × matched MAE 0.43; production table identical to the audit (max difference 0.0) | Ruled out as the main cause |
| Geography (MI vs ES) | 0 MI intervals during a split; ES curves used in all 1 058 (2023–24) split hours | Ruled out |
| **Selective rejection of cheap supply** (market constraints) | Swapping offered supply for matched supply cuts MAE from 24.6 to 5.6. From 2025-03-19, **simple** offers priced below p\* are matched in full (mean unmatched 0.0 MWh), so all unmatched cheap supply carries complex-condition codes. | Main mechanical factor (see §5 for the causal limits) |
| Removing *all* complex offers | Simple-only supply overshoots: bias **+23.9**, MAE 27.2 (2025-03-19 – 09-30) | Rejection is *selective*. Many complex offers are accepted, so no fixed filtering rule works. |
| Coupling flows (data representation) | Adding the realised matched coupling rows (`Imp/Exp`) to the offered curves does **not** reduce the error (MAE 11.65 → 12.99). The demand-side gap is only weakly associated with the error (r = −0.20). | Secondary; does not explain the gap on its own |

## 4. Quantified association with rejected volume

U = offered supply at or below p\* minus matched supply. Its mean is 4 106 MWh and its median 3 479 MWh. 84 % of 2023–24 intervals have U above 500 MWh.

| Statistic | Value |
|---|---|
| Pearson r(error, U) | −0.709 |
| Spearman ρ(error, U) | −0.761 |
| r(error, official price) | −0.409 |
| r within price regimes: ≤0 / 0–20 / 20–60 / 60–100 / >100 | 0.09 / −0.40 / −0.73 / −0.73 / −0.76 |
| OLS error on U and demand gap (GWh) | −4.53 per GWh U, −3.12 per GWh gap, R² 0.53 |
| Plus the official price as a control | −3.70 per GWh U, −5.81 per GWh gap, −0.17 per EUR/MWh, R² 0.59 |
| Mean error by U decile (lowest → highest) | +6.3, −3.4, −10.7, −14.6, −17.3, −23.7, −34.1, −36.1, −42.6, −55.9 |

The association is strong, monotone, and survives conditioning on the price level. **About 40 % of the error variance is not explained by these two volumes.**

## 5. Association vs causality

- **Established (accounting identity).** Replacing offered supply with matched supply removes about 77 % of the MAE. From 2025-03-19, every unmatched cheap MWh carries a complex-condition code. These are statements about the curves themselves.
- **Not established.**
  - *Why* each offer was rejected. The 2023–24 data are anonymised and the C-codes are undecoded.
  - Whether rejections would change if the price changed. They are jointly determined with p\* inside the algorithm, so U is endogenous.
  - Whether a forecastable rejected volume would transfer to forecast curves.
- U is not a valid same-day predictor: it requires p\* and the matched curves. The adjusted model therefore uses only lagged U and lagged discrepancies.

## 6. Non-stationarity: relevant to the 2025 confirmation period

Monthly MAE of the offered crossing (`offered_crossing_error_by_month` in the JSON):

| Period | Monthly MAE range | Mean U (MWh) |
|---|---|---|
| 2023-01 – 2023-04 | 40.5 – 45.6 | 6 100 – 7 000 |
| 2024-03 – 2024-05 | 8.5 – 13.3 | 2 700 – 4 800 |
| 2024-10 – 2024-12 | 22.1 – 28.6 | 3 700 – 4 400 |
| 2025-01 – 2025-02 | 38.2 – **51.1** | ~5 400 |
| 2025-04 – 2025-09 | **9.8 – 13.4** | 1 060 – 2 680 |

The discrepancy roughly halves from April 2025, the month after OMIE starts publishing the new offer typology (2025-03-19). The January–September 2025 confirmation period is comparable in data format, hourly resolution and market scope. It is **not** stationary in the clearing discrepancy, so it tests robustness to a structural change. This was found after the protocol was frozen (`preregistration_curve_extension.md`), and the protocol was not changed.

## 7. Background sources

- OMIE public file download (`curva_pbc`, `marginalpdbc`), as in `data_sources.md`.
- CNMC, public consultation on OMIE's MTU15 market-rule proposal and new offer typology (DCOOR/DE/009/24, 2024–2025).
- SDAC: the 15-minute MTU went live for delivery day 2025-10-01.
