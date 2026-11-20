# Operations & Decisions Portfolio — Hospital Pharmacy & Critical Drug Inventory

> **Module:** Advanced Machine Learning — ESILV 2026–2027  
> **Application Domain:** Santé / Pharmacie à Usage Intérieur (PUI) Hospitalière  
> **Focus:** Supply Chain de Médicaments d'Intérêt Thérapeutique Majeur (MITM)  
> **Project overview:** Voir [`PROJECT_SUBJECT.md`](PROJECT_SUBJECT.md)

```text
advanced-ml-portfolio/
├── README.md               # Vue d'ensemble du dépôt binôme
├── PROJECT_SUBJECT.md      # Cadrage métier du sujet Santé (5-6 lignes)
├── submission.yaml         # Déclarations officielles de soumission
├── claims_evidence.csv     # Index des preuves scientifiques (TS-01, RL-01, OPS-01)
├── AI_USE_LOG.md           # Journal de transparence sur l'usage d'IA
├── CONTRIBUTIONS.csv       # Rôles alternés par session (experiment-lead / evidence-steward)
├── configs/replay.yaml     # Configuration de replay en deux commandes figées
├── data/README.md          # Provenance et empreintes cryptographiques des données
├── artifacts/              # Artefacts figés (final_forecast.csv, q_table.npy)
├── evidence/               # Reçus contrôlés et sorties de replay
├── report/release_brief.md # Rapport final de synthèse opérationnelle (≤ 4 pages)
└── scripts/quick_replay.py # Exécuteur de replay fail-closed
```

## Sujet du Capstone en bref
Ce projet modélise la chaîne logistique critique d'une Pharmacie à Usage Intérieur (PUI) hospitalière confrontée aux tensions d'approvisionnement sur les Médicaments d'Intérêt Thérapeutique Majeur (MITM). Le volet **Time Series** prévoit la demande sur l'horizon de réapprovisionnement de 28 jours (en isolant les périodes de rupture censurées `stockout_flag=1`), tandis que le volet **Reinforcement Learning** apprend une politique de commande par Q-learning arbitrant entre coût de stockage frigorifique et risque vital de rupture en soins intensifs.

## Métadonnées d'environnement binôme

- **Système d'exploitation :** Windows (x86_64)
- **Environnement Python :** Python 3.12.16 verrouillé via `uv`
- **CPU de référence :** Intel Core i7 / AMD Ryzen (exécution locale CPU garantie)
- **Statut des preuves TD1 :** baseline saisonnière lag-7 figée (`TS-01` dans `claims_evidence.csv`, MAE 9.61 vs 12.56)

## Quick replay

Depuis la racine du dépôt :

```bash
python scripts/quick_replay.py --config configs/replay.yaml
```
