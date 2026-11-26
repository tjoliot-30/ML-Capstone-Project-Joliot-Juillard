# Data card — daily demand

- **Purpose:** rolling-origin point and interval forecast evaluation in TDo1.
- **Canonical scope:** 1,461 consecutive daily observations (four years including leap day).
- **Public development split:** the first 1,433 targets.
- **Final horizon:** the last 28 dates and known descriptors are public; targets are instructor-controlled and excluded from student exports.
- **Target:** recorded `demand_units`; known descriptors are `promotion`, `holiday_period`, and `stockout_flag`.
- **Patterns:** trend, weekly and annual seasonality, intermittent promotion/holiday effects, permanent level shifts at generator indices 850 and 1,200, noise, and two stockout-censoring windows.
- **Generation:** deterministic instructor-maintained offline process; no public target generator is distributed.
- **Protocol warning:** no random split and no future-informed preprocessing. All models use identical forecast origins and horizons.
- **Limits:** recorded demand during `stockout_flag=1` is censored sales, not latent unconstrained demand. One synthetic series cannot support universal superiority claims; finite-sample interval coverage is uncertain.
- **Privacy/terms:** synthetic and free of personal/proprietary records; internal ESILV module use only pending the written licence decision in `NOTICE.md`.
