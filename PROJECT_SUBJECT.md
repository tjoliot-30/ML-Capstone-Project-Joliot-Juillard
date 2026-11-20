# Project Subject: Hospital Pharmacy Supply Chain & Critical Drug Inventory

- **Context:** Managing hospital supply chains for Major Therapeutic Interest Medicines (MITM), specifically critical intensive care antibiotics subject to national shortage risks.
- **Time Series Forecasting:** Predicting daily antibiotic demand over a 28-day replenishment cycle, capturing hospital weekly dynamics (high admissions on weekdays vs. reduced weekend staffing) while handling censored demand caused by historical stockouts (`stockout_flag=1`).
- **Reinforcement Learning Policy:** A tabular Q-learning agent learns an optimal sequential reordering policy under Markovian demand shifts (epidemic surges vs. baseline periods), trading off cold-chain holding costs against critical clinical penalties for ICU drug shortages.
- **Operational Synthesis & Safety:** Delivering a bounded operational decision with safety buffers and a clear manual fallback procedure to secure uninterrupted patient care without overstocking.
