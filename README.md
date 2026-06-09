# Wine Quality Classification

A machine learning project comparing 4 classification algorithms on the **UCI Wine Dataset** (via scikit-learn). Includes confusion matrices, F1-score breakdowns, ROC curves, and a model comparison dashboard.

---

## Results 

| Model | Test Macro-F1 | CV F1 | CV Std |
|---|---|---|---|
| **Random Forest** | **1.000** | **0.979** | ±0.029 |
| KNN | 1.000 | 0.951 | ±0.037 |
| Logistic Regression | 0.971 | 0.980 | ±0.027 |
| SVM | 0.941 | 0.993 | ±0.015 |

> **Random Forest** wins overall — perfect test accuracy *and* the most stable cross-validation score.

---

## Project Structure

```
classification/
├── wine_classification.py       # Main script — training, evaluation, plots
├── requirements.txt             # Python dependencies
├── outputs/
│   ├── 1_confusion_matrices.png
│   ├── 2_f1_scores.png
│   ├── 3_roc_curves.png
│   └── 4_summary_dashboard.png
└── README.md
```

---

## Models Compared

| Model | Key Hyperparameters |
|---|---|
| SVM | RBF kernel, C=10, gamma=scale |
| KNN | k=7, Minkowski distance |
| Random Forest | 200 trees, unlimited depth |
| Logistic Regression | C=1.0, max_iter=1000 |

SVM, KNN, and Logistic Regression use **StandardScaler**. Random Forest is scale-invariant and uses raw features.

---

## Visualizations

### 1. Confusion Matrices
Normalized heatmaps showing per-class prediction accuracy for each model.

### 2. F1-Score Analysis
- Per-class F1 grouped bar chart
- 5-fold cross-validated Macro-F1 with error bars

### 3. ROC Curves
One-vs-Rest ROC curves with AUC scores for all three wine classes per model.

### 4. Summary Dashboard
- Radar chart comparing all metrics across models
- Random Forest feature importances (top 8)
- Score summary table

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/classification.git
cd classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python wine_classification.py
```

Output plots are saved to the `outputs/` folder.

---

## 📖 Dataset

The [UCI Wine Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) contains 178 samples of wine with 13 chemical features (alcohol, malic acid, flavanoids, etc.), classified into 3 cultivar classes. Loaded directly via `sklearn.datasets.load_wine()` — no download needed.

---

## 🔍 Why Random Forest Wins

- **Ensemble voting** across 200 trees cancels individual variance
- Naturally captures **non-linear feature interactions** (flavanoids × proline, etc.)
- **Scale-invariant** — no preprocessing bias
- KNN matched on test set but had a weaker, more variable CV score (likely lucky split)
- SVM had the best CV score but showed a test-time gap on Class 2

---
