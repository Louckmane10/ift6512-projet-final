# DRO for Feature Acquisition with Missing Data

[![Julia](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Projet Final IFT6512 — Programmation Stochastique**  
> Université de Montréal, Décembre 2025

## 📋 Description

Ce projet implémente une approche **distributionnellement robuste (DRO)** pour l'apprentissage supervisé avec **données manquantes** et **acquisition de caractéristiques sous contrainte de budget**.

### Problématique

En diagnostic médical, les données manquantes suivent souvent un mécanisme **MNAR** (Missing Not At Random) : un médecin prescrit un test parce qu'il suspecte un problème, donc le fait qu'une valeur soit manquante dépend de la valeur elle-même. Ce mécanisme est **non-identifiable**, ce qui rend les méthodes d'imputation classiques vulnérables.

### Solution proposée

Notre approche utilise la **DRO avec ensemble d'ambiguïté de Wasserstein** pour :
1. Considérer un ensemble de distributions plausibles autour de la distribution de référence
2. Optimiser pour le **pire cas** dans cet ensemble
3. Décider quelles caractéristiques acquérir sous contrainte de budget

## 🚀 Installation

### Prérequis

- Julia 1.9 ou supérieur

### Installation

```bash
git clone https://github.com/VOTRE_USERNAME/dro-missing-data.git
cd dro-missing-data
julia run_experiments.jl
```

## 📁 Structure du projet

```
dro-missing-data/
├── src/
│   ├── scenario_generation.jl   # Génération de scénarios (Gaussien, MICE)
│   ├── dro_objective.jl         # Évaluation de l'objectif DRO
│   ├── acquisition_solver.jl    # Optimisation combinatoire
│   ├── saa_dro.jl               # Algorithme SAA-DRO principal
│   ├── baselines.jl             # Méthodes de comparaison
│   └── utils.jl                 # Utilitaires (données, métriques)
├── rapport/
│   └── projet_final.tex         # Rapport LaTeX (~50 pages)
├── run_experiments.jl           # Script principal
├── README.md
└── .gitignore
```

## 💻 Utilisation

```bash
julia run_experiments.jl all    # Toutes les expériences
julia run_experiments.jl 1      # Exp 1: Comparaison mécanismes
julia run_experiments.jl 2      # Exp 2: Sensibilité à epsilon
julia run_experiments.jl 3      # Exp 3: Valeur de l'information
julia run_experiments.jl 4      # Exp 4: Convergence SAA
julia run_experiments.jl 5      # Exp 5: Sensibilité au budget
```

## 📊 Résultats principaux

| Méthode | MCAR | MNAR+ | Δ robustesse |
|---------|------|-------|--------------|
| Greedy-NR | 0.462 | 0.538 | 0.076 |
| **DRO (ε=0.1)** | **0.472** | **0.498** | **0.026** |

- ✅ DRO surpasse les baselines de **7-9%** sous MNAR
- ✅ Écart de robustesse réduit de **3×**

## 📐 Formulation

```
min_z  max_{P ∈ P_ε}  E_P[ℓ(y, f_θ(x̃(z, ξ)))]
s.t.   Σ c_j z_j ≤ B,  z_j ∈ {0,1}
```

## 📚 Références

- Kuhn et al. (2024). *Distributionally robust optimization*. Acta Numerica.
- Le Morvan et al. (2021). *What's a good imputation?* NeurIPS.

## 📄 License

MIT License - Usage académique.

---
**Auteur:** Louck | **Cours:** IFT6512 | **Date:** Décembre 2025
