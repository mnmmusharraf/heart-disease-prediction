# Heart Disease Prediction

> End-to-end machine learning pipeline for binary classification of heart disease presence using 14 clinical patient features — achieving **0.9551 ROC-AUC** via XGBoost and ensemble methods.

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189fdd?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9551-2ea44f?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E2-20beff?style=flat-square&logo=kaggle&logoColor=white)

---

## Results at a Glance

| Metric | Value |
|---|---|
| Best single-model ROC-AUC | **0.9551** (XGBoost) |
| Ensemble ROC-AUC | **0.9552** (HGB + XGBoost) |
| Cross-validation std dev | ±0.0004 |
| Dataset size | 630,000 patient records |
| Features | 14 clinical parameters |
| Class balance | 55.2% negative · 44.8% positive |

---

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [Feature Engineering](#feature-engineering)
- [Feature Importance](#feature-importance)
- [Repository Structure](#repository-structure)
- [Future Work](#future-work)

---

## Dataset

Sourced from the [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) competition. 630,000 patient records with 14 clinical features.

| Feature | Description | Range | Type |
|---|---|---|---|
| Age | Patient age | 29 – 77 years | Numerical |
| Sex | Gender | 0 = female, 1 = male | Binary |
| Chest pain type | Type of chest pain | 1 – 4 | Categorical |
| BP | Resting blood pressure | 94 – 200 mmHg | Numerical |
| Cholesterol | Serum cholesterol | 126 – 564 mg/dl | Numerical |
| FBS over 120 | Fasting blood sugar > 120 mg/dl | 0, 1 | Binary |
| EKG results | Resting electrocardiographic results | 0 – 2 | Categorical |
| Max HR | Maximum heart rate achieved | 71 – 202 | Numerical |
| Exercise angina | Exercise-induced angina | 0, 1 | Binary |
| ST depression | ST depression (exercise vs rest) | 0 – 6.2 | Numerical |
| Slope of ST | Peak exercise ST segment slope | 1 – 3 | Categorical |
| Number of vessels fluro | Major vessels coloured by fluoroscopy | 0 – 3 | Numerical |
| Thallium | Thallium stress test result | 3, 6, 7 | Numerical |
| **Heart Disease** | **Target variable** | **0 = Absence, 1 = Presence** | **Target** |

---

## Installation

**Prerequisites:** Python 3.8+, pip

```bash
# Clone the repository
git clone https://github.com/mnmmusharraf/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

```
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
```

---

## Usage

### Running the Notebook

```bash
jupyter notebook heart-disease-eda-ipynb.ipynb
```

### Making Predictions with the Trained Model

```python
import joblib
import pandas as pd
import xgboost as xgb

# Load model and feature schema
model = xgb.XGBClassifier()
model.load_model("heart_xgb.json")
feature_columns = joblib.load("xgb_columns.pkl")

# Construct a patient record
sample_data = pd.DataFrame({
    'Age': [55],
    'Sex': [1],
    'Chest pain type': [4],
    'BP': [140],
    'Cholesterol': [250],
    # ... include all features
})

# Align feature order and predict
sample_data = sample_data.reindex(columns=feature_columns, fill_value=0)
probability = model.predict_proba(sample_data)[:, 1]
print(f"Heart disease probability: {probability[0]:.3f}")
```

---

## Model Development

Five classifiers were evaluated via 5-fold stratified cross-validation. XGBoost achieved the best single-model performance; a weighted ensemble with HistGradientBoosting provided a marginal additional gain.

### Algorithm Comparison

| Model | ROC-AUC | Std Dev |
|---|---|---|
| Logistic Regression | 0.9505 | ±0.0008 |
| Random Forest | 0.9521 | ±0.0006 |
| Gradient Boosting | 0.9538 | ±0.0005 |
| HistGradientBoosting | 0.9545 | ±0.0005 |
| **XGBoost** | **0.9551** | **±0.0004** |
| **Ensemble (HGB + XGB)** | **0.9552** | **±0.0004** |

### Ensemble Strategy

Final predictions used a weighted average of predicted probabilities:

```
Final score = 0.45 × HistGradientBoosting + 0.55 × XGBoost
```

### XGBoost Configuration

```python
XGBClassifier(
    n_estimators     = 300,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 20,
    reg_alpha        = 0.0,
    reg_lambda       = 1.0,
    tree_method      = "hist",
    eval_metric      = "auc",
    random_state     = 42,
)
```

---

## Feature Engineering

Seven interaction and binning features were constructed on top of the raw clinical variables:

| Feature | Description |
|---|---|
| `Age_group` | Binned: Young (≤40), Mid (40–55), Old (≥55) |
| `Chol_cat` | Binned: Low (<200), Mid (200–240), High (>240) |
| `Age_BP_interaction` | Age × Blood Pressure |
| `Age_Chol` | Age × Cholesterol |
| `BP_Chol` | Blood Pressure × Cholesterol |
| `Chol_per_Age` | Cholesterol ÷ Age ratio |
| `MaxHR_per_Age` | Max Heart Rate ÷ Age ratio |

> **Note:** Minimal feature engineering was required — the raw clinical features already carry strong predictive signal.

---

## Feature Importance

Ranked by Pearson correlation with the target variable:

| Rank | Feature | Correlation |
|---|---|---|
| 1 | Thallium | +0.606 |
| 2 | Chest pain type | +0.461 |
| 3 | Exercise angina | +0.442 |
| 4 | Max HR | −0.441 |
| 5 | Number of vessels fluro | +0.439 |
| 6 | ST depression | +0.431 |
| 7 | Slope of ST | +0.415 |
| 8 | Sex | +0.342 |
| 9 | EKG results | +0.219 |
| 10 | Age | +0.212 |

**Key findings:**
- The thallium stress test is the single strongest predictor (ρ = 0.606)
- Chest pain type and exercise-induced angina are highly indicative
- Maximum heart rate is inversely correlated — higher max HR is associated with disease *absence*
- XGBoost outperformed other models due to its ability to capture non-linear feature interactions

---

## Repository Structure

```
heart-disease-prediction/
├── heart-disease-eda-ipynb.ipynb   # EDA and full training pipeline
├── heart_xgb.json                  # Saved XGBoost model (JSON format)
├── xgb_columns.pkl                 # Feature column names for inference
├── submission.csv                  # Kaggle competition submission
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Future Work

- [ ] Deploy model as a REST API using FastAPI
- [ ] Build an interactive web interface for patient predictions
- [ ] Add SHAP explanations for model interpretability
- [ ] Experiment with neural network architectures
- [ ] Automated hyperparameter search with Optuna
- [ ] Validate on external clinical datasets

---

## Acknowledgements

- [Kaggle](https://www.kaggle.com) for providing the dataset via Playground Series S6E2
- The open-source community behind scikit-learn, XGBoost, and pandas

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

**Mohammed Musharraf**  
GitHub: [@mnmmusharraf](https://github.com/mnmmusharraf)  
Project: [heart-disease-prediction](https://github.com/mnmmusharraf/heart-disease-prediction)

---

*Last updated: March 2026*
