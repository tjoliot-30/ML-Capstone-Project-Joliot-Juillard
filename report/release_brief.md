# Operations & Decisions Portfolio — Release Brief

**Pair:** Axel Juillard & Timothée Joliot  
**Repository Commit (Prior Frozen Six-Session Baseline):** `7175f83` (Final immutable tag pending controlled instructor receipts)  
**Limit:** Four equivalent pages, excluding references

---

## 1. Decision context and release bounds

Hospital In-House Pharmacies (PUI) face chronic supply chain volatility for Major Therapeutic Interest Medicines (MITM), notably injectable antibiotics used in intensive care units (ICU). Under-ordering risks acute clinical rationing and patient harm, whereas over-ordering inflates cold-chain holding costs and batch expiration waste. In this capstone, the **Chief Hospital Pharmacist** acts as the accountable operational owner, supported by our two-strand analytical evaluation:

- **Time Series strand (`TS-01`):** 28-day daily antibiotic demand forecasting aligned with centralized procurement lead times.
- **Reinforcement Learning strand (`RL-01`):** Sequential daily reordering under a discrete two-regime Markov demand environment (`InventoryControlEnv`) reflecting asymmetric clinical stockout penalties.

**Explicit non-integration statement:** Forecast origins from `TS-01` and simulated episodes from `RL-01` constitute **separate evaluation populations**. Forecasts produced in TDo1 were never fed as state observations into the TDo2 RL environment. Consequently, **no end-to-end forecast-to-controller performance or causal hospital benefit is claimed**.

---

## 2. Forecasting release (`TS-01`)

Tracing TD1 through TDo1 on `daily_demand_development.csv` (`1,433` rows, `SHA256: b275d65b...`):

- **Baselines and rolling comparison:** On TD1's frozen 28-day development block, lag-7 seasonal-naive (`MAE 9.61`, `sMAPE 10.13%`) beat historical mean (`12.56`) and last-value (`12.11`), confirming strong weekly seasonality (weekday admission peaks vs. weekend troughs). Across TD2's three expanding origins `(1250, 1310, 1370)` with a 14-day horizon (`9` fold rows, `0` warnings), `SARIMA(1,1,1)(1,0,1,7)` (`mean MAE 6.14`, `std 1.03`, `sMAPE 6.70%`) and additive ETS (`6.17`, `std 1.17`) outperformed seasonal-naive (`7.29`).
- **Failure period and diagnostics:** At origin `1370`, both SARIMA (`MAE 7.21`) and ETS (`7.40`) underperformed seasonal-naive (`6.93`), demonstrating sensitivity to late-series regime shifts. Ljung-Box tests did not reject white noise at lag 7 (`p=0.54`) or lag 14 (`p=0.52`), though non-rejection on finite samples never proves residual independence.
- **Interval release and public replay:** Across the `42` dependent rolling cases in TDo1, SARIMA reached `85.71%` empirical 80%-coverage (mean width `23.90`) and `100.00%` 95%-coverage (mean width `36.55`). Passing the predeclared 5% development gate, SARIMA was refit once on all `1,433` public rows to freeze `artifacts/final_forecast.csv` (`SHA256: 8fe96aba...`, `28` dates `2026-12-04` to `2026-12-31`, forecasts `85.05–109.40`, strictly nested bounds). Public replay (`evidence/replay/forecasting/result.json`) verified schema, hash, and seasonal-naive checks (`3.23s`). **Protected TDo1 receipt (`forecast_holdout_receipt.json`) remains pending instructor handoff**; `TS-01` is therefore held as `partial`.
- **Literature context:** Our methodology aligns with Lim et al. (2024), who demonstrated that SARIMA models effectively capture weekly seasonality in French hospital medicine consumption over rolling moving-average baselines, while emphasizing vulnerability to sudden epidemic surges and data distortion during drug shortages (`stockout_flag=1` right-censoring).

---

## 3. RL policy release (`RL-01`)

Tracing TD3 through TDo2 on `InventoryControlEnv` (`order=0.45`, `holding=0.30`, `unmet=3.00`, capturing the 10× clinical penalty of ICU shortages over refrigerated storage):

- **Baselines and explicit training:** On TD3's five public seeds `(11..71)`, the regime-aware `reorder` heuristic (`targets 2/4`) reduced mean cost by `53.3%` vs. `random` (`41.16` vs. `88.20`) and lifted fill rate from `81.83%` to `99.17%`. In TD4, explicit tabular Q-learning (`1,750` episodes, `alpha=0.18`, `gamma=0.96`, linear $\epsilon$ `0.85->0.05`, truncation bootstrapping) produced frozen `artifacts/q_table.npy` (`(9,2,3)`, `SHA256: 61cce191...`).
- **Development vs. public safety replay:** On TD4's five disjoint development seeds `(100011..100071)`, Q-learning cut cost `51.2%` vs. `random` (`42.78` vs. `87.75`, fill rate `98.50%`) but slightly trailed `reorder` (`42.48`, fill rate `98.83%`, unmet `1.4` vs. `1.8`) due to a learned under-ordering boundary action at `(stock=2, regime=1)`. On TDo2's public safety replay (`200011..200071`, reproduced in `evidence/replay/reinforcement_learning/result.json` in `2.29s`), Q-learning achieved `38.22` mean cost (`95% CI [32.49, 43.95]`) and `98.50%` fill rate (`[0.9715, 0.9985]`) vs. `reorder` (`39.03`, `98.67%` fill) and `random` (`75.33`, `88.83%` fill). **Controlled TDo2 receipt remains pending instructor evaluation**; `RL-01` is held as `partial`.
- **Literature context:** Abu Zwaida et al. (2021) support formalizing hospital drug replenishment with heavy shortage penalties to prevent critical stockouts. Critically, Stranieri et al. (2025) confirm that classical base-stock heuristics remain highly competitive against RL under simple low-dimensional dynamics—directly explaining why `reorder` matches or beats tabular Q-learning here—and highlight that real pharmaceutical chains require modeling drug expiration/perishability.

---

## 4. Operational decision (`OPS-01`) and safety gate

- **Decision:** **Conditional HOLD** (`OPS-01`) pending instructor-returned protected receipts for both strands. Even if both controlled receipts later pass, authorized scope is strictly **advisory shadow-mode decision support** reviewed daily by the Chief Hospital Pharmacist—never autonomous hospital ordering.
- **Release gate & transparent fallback:**
  1. **RL Service & Cost Gate:** Require controlled `mean_fill_rate >= 0.98` and `mean_cost <= 1.05 * reorder_cost`. If violated (or given Q-learning's TD4 dev shortfall at `(stock=2, regime=1)`), trigger automatic fallback to the transparent regime-aware `reorder` heuristic (`targets 2/4`).
  2. **Forecast Gate & Override:** If protected TDo1 evaluation degrades beyond development expectations, fall back to additive ETS or lag-7 seasonal-naive. Any active `stockout_flag=1` event or national ANSM shortage alert overrides algorithmic quantities with emergency central-depot procurement and manual pharmacist buffer allocation.

---

## 5. Risks, refusal boundaries and next evidence

- **Refused claims:** We explicitly refuse claims of (a) autonomous ICU procurement or clinical safety, (b) tested forecast-to-RL integration, (c) calibrated production prediction intervals from `n=3` dependent rolling origins or `n=5` small-sample RL episodes, and (d) direct transfer from synthetic course data to live hospital formularies.
- **Key residual risks:** Unmodeled stockout censoring (`stockout_flag=1`) biases historical consumption downward; `SARIMA` omits known `promotion` and `holiday_period` descriptors; `InventoryControlEnv` omits supplier lead-time delays, batch expiration waste, and multi-ward dependencies.
- **Next minimum evidence required:** (1) Return and SHA256 verification of protected TDo1 and TDo2 instructor receipts; (2) A coupled simulation feeding uncensored rolling demand intervals into an expanded perishable-drug MDP; (3) A retrospective shadow-mode validation on genuine hospital PUI dispensation records.

---

## References

1. Lim, S. S. Y., Phouratsamay, S.-L., Yahouni, Z., & Gascard, E. (2024). *Medicine consumption demand forecasting in French hospitals using Seasonal Auto-Regressive Integrated Moving Average (SARIMA) models*. International Conference on Control, Automation and Diagnosis (ICCAD).
2. Abu Zwaida, T., Pham, C., & Beauregard, Y. (2021). *Optimization of Inventory Management to Prevent Drug Shortages in the Hospital Supply Chain*. Applied Sciences, 11(6), 2726. DOI: `10.3390/app11062726`.
3. Stranieri, F., Kouki, C., van Jaarsveld, W., & Stella, F. (2025). *Classical and Deep Reinforcement Learning Inventory Control Policies for Pharmaceutical Supply Chains with Perishability and Non-Stationarity*. arXiv preprint `arXiv:2501.10895`.
