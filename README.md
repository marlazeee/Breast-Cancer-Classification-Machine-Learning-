# Breast Cancer Classification — Machine Learning

A machine learning project that classifies breast tumors as **malignant or benign** using the Wisconsin Breast Cancer dataset. Five classification algorithms are trained, evaluated, and compared.

---

## Dataset

| Property | Details |
|---|---|
| Source | `data.csv` (Wisconsin Breast Cancer Dataset) |
| Samples | 569 |
| Features | 30 numerical features |
| Target | Binary — Malignant (M) / Benign (B) |
| Train / Test split | 80% / 20% (455 train, 114 test, `random_state=42`) |

### Features

30 features derived from digitized images of fine needle aspirate (FNA) of breast masses. For each of 10 cell nucleus characteristics, three values are recorded: **mean**, **standard error**, and **worst** (largest).

The 10 characteristics are: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.

---

## Preprocessing Pipeline

1. Drop irrelevant columns (`id`, `Unnamed: 32`)
2. Encode target labels with `LabelEncoder` (M → 1, B → 0)
3. Impute missing values with `SimpleImputer`
4. Scale all features with `StandardScaler` (zero mean, unit variance)
5. Wrap steps in a `Pipeline` + `ColumnTransformer`

---

## Models

| Model | Notes |
|---|---|
| SGDClassifier | Stochastic Gradient Descent |
| LinearSVC | Linear Support Vector Machine |
| SVC (Polynomial) | Degree-2 polynomial kernel |
| DecisionTreeClassifier | Tuned with `max_depth` and `max_features` to reduce overfitting |
| RandomForestClassifier | 900 estimators, `max_leaf_nodes=5` |

---

## Results

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|
| SGDClassifier | 0.96 | 0.93 | 0.98 | 0.95 |
| Linear SVM | 0.96 | 0.93 | 0.98 | 0.95 |
| **Polynomial SVM (deg=2)** | **0.98** | **1.00** | **0.95** | **0.98** |
| Decision Tree (depth=3) | 0.95 | 0.93 | 0.93 | 0.93 |
| Decision Tree (depth=4) | 0.95 | 0.93 | 0.93 | 0.93 |
| Random Forest | 0.96 | 0.98 | 0.93 | 0.95 |

**Best model:** Polynomial SVM — 98% accuracy, 100% precision, 0.98 F1 on the test set.

---

## Evaluation Metrics

- Confusion matrix (train and test)
- Accuracy, Precision, Recall, F1-score
- Cross-validation with `cross_val_predict`

High recall is prioritized since false negatives (missed malignant tumors) carry the highest clinical cost.

---

## Libraries

```
pandas, matplotlib, scikit-learn
  └── model_selection: train_test_split, cross_val_predict
  └── preprocessing: LabelEncoder, StandardScaler
  └── pipeline: Pipeline, ColumnTransformer
  └── impute: SimpleImputer
  └── linear_model: SGDClassifier
  └── svm: LinearSVC, SVC
  └── tree: DecisionTreeClassifier
  └── ensemble: RandomForestClassifier
  └── metrics: confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
```

---

## Usage

1. Clone the repo and open the notebook:
   ```bash
   git clone https://github.com/marlazeee/Breast-Cancer-Classification-Machine-Learning-.git
   cd Breast-Cancer-Classification-Machine-Learning-
   jupyter notebook Breast_Cancer_Classification.ipynb
   ```

2. Place `data.csv` in the same directory (Wisconsin Breast Cancer Dataset).

3. Run all cells top to bottom.

---

## Key Findings

- All models achieved strong test accuracy (95–98%), confirming the dataset is well-suited for binary classification.
- Polynomial SVM delivered the best balance of precision and recall, with **perfect precision (1.00)** on the test set.
- Decision Trees were regularized (`max_depth`, `max_features`) to reduce overfitting and improve generalization.
- All models maintained high recall (0.93–0.98), which is critical for medical diagnosis tasks where missing a malignant case is unacceptable.
