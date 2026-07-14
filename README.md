<div align="center">

# Heart Disease Prediction

### Machine Learning · Clinical Data · Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-Latest-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)]()

<br/>

**Accuracy: 80.43%** &nbsp;|&nbsp; **ROC-AUC: 0.898** &nbsp;|&nbsp; **Algorithm: Logistic Regression**

</div>

---

## Overview

An end-to-end ML pipeline that takes raw clinical patient data and produces an interpretable heart disease prediction model — from data cleaning to final evaluation.

```
heart.csv  →  Clean  →  EDA  →  Train  →  Evaluate  →  Insight
```

- Handles missing values and encodes categorical features
- Performs exploratory data analysis with visualizations
- Trains a Logistic Regression classifier on clinical features
- Evaluates with Accuracy, Confusion Matrix, and ROC-AUC
- Ranks features by their influence on the prediction

---

## Tech Stack

| Layer | Tools | Role |
|-------|-------|------|
| Language | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat-square) | Core runtime |
| Data | ![pandas](https://img.shields.io/badge/-pandas-150458?logo=pandas&logoColor=white&style=flat-square) ![numpy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat-square) | Wrangling & computation |
| Visuals | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) ![Seaborn](https://img.shields.io/badge/-Seaborn-4C9BE8?style=flat-square) | EDA & charts |
| ML | ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square) | Model training & evaluation |

---

## Dataset

Source: `heart.csv`

The binary target is derived from the `num` column:

| `num` Value | Interpretation |
|-------------|----------------|
| `0` | No Disease ✅ |
| `> 0` | Disease Present ⚠️ |

---

## Quick Start

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Heart_Disease.py
```

> `heart.csv` is included in the repository — no additional setup required.

---

## Visualizations

### 1 · Heart Disease Distribution
Class balance between healthy and at-risk patients.

<img width="1240" height="606" alt="image" src="https://github.com/user-attachments/assets/5c4f08e6-72a1-4816-86e1-2b4bd90b39a8" />


### 2 · Correlation Heatmap
Feature-to-feature and feature-to-target relationships.

<img width="1160" height="606" alt="image" src="https://github.com/user-attachments/assets/f42c652a-c857-4ada-9aed-38ca519e093e" />


### 3 · Age Distribution
Spread of patient ages across the dataset.

<img width="1278" height="617" alt="image" src="https://github.com/user-attachments/assets/4f16b56f-9eed-4bf8-87e2-07e3b2e21bde" />


### 4 · Confusion Matrix
Correct predictions vs. misclassifications.

<img width="1110" height="610" alt="image" src="https://github.com/user-attachments/assets/11fe952c-f2b4-43db-9b02-295ecdf64be6" />


### 5 · ROC Curve
Model's ability to separate disease from no disease.

<img width="1267" height="623" alt="image" src="https://github.com/user-attachments/assets/f1212099-cc83-4757-b26c-62550548abdb" />


### 6 · Feature Importances
Top clinical variables driving the model's predictions.

<img width="1322" height="615" alt="image" src="https://github.com/user-attachments/assets/d79b16ac-f987-4e49-abe8-e2bb8a85abf0" />


---

## Model Performance

<div align="center">

| Metric | Score |
|--------|-------|
| Accuracy | **80.43%** |
| ROC-AUC | **0.898** |

</div>

### Feature Importances (Logistic Regression Coefficients)

Positive values push toward **Disease ⚠️** — negative values push toward **Healthy ✅**.

| Rank | Feature | Coefficient |
|------|---------|-------------|
| 1 | `sex` | +1.4259 |
| 2 | `ca` | +1.2378 |
| 3 | `exang` | +0.9820 |
| 4 | `oldpeak` | +0.7121 |
| 5 | `thal` | +0.5309 |
| 6 | `fbs` | +0.3990 |
| 7 | `restecg` | −0.1086 |
| 8 | `slope` | −0.2089 |
| 9 | `cp` | −0.5876 |
| 10 | `dataset` | −1.5058 |

---

## Disclaimer

This project is for **educational purposes only** and is not a diagnostic tool.

---

## Author
Nouman Majeed
