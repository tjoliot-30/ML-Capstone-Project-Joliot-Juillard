# Data and environment provenance

| Strand | Dataset/environment ID | Version or SHA256 | Declared replay input(s) | Access/licence | Unit and target/reward | Evaluation boundary | Known limitation |
|---|---|---|---|---|---|---|---|
| forecasting | `daily_demand:development` | `b275d65b07f44795908b8941703664f69188a8767e7c84bb049e79435b6c8ba6` | `shared/datasets/daily_demand_development.csv` | synthetic course data | daily antibiotic demand | frozen rolling origins and public replay window | synthetic series; one protected 28-day horizon; censored demand during stockouts |
| reinforcement_learning | `InventoryControlEnv:v1` | `shared.inventory_env` | `shared/inventory_env.py` | synthetic course environment | inventory cost and fill rate | frozen policy on independent public seeds (11, 23, 37, 51, 71) | discrete simplified dynamics; 2-regime Markov model |

Do not copy protected demand targets, controlled RL seeds, private episode rows,
personal data or restricted datasets.
