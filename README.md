# Breast Cancer Classification
**An individual Applied AI university project exploring how data analysis, visualisation and model evaluation can be used to distinguish benign and malignant breast tumour samples.**

Rather than focusing on one algorithm, I explored the dataset, compared **four classification models**, tuned different parameters and looked at *why* models performed differently — especially when predicting malignant cases.

**At a glance:** 569 samples · 30 features · 4 models · 80/20 train/test split · 5-fold cross-validation for model tuning · **97.37% best test accuracy**

> **Scope:** This is an educational machine learning project using an established research dataset. It is **not** a clinical diagnostic tool, and the reported results should not be interpreted as clinical performance.

## Project goal

Use the **Breast Cancer Wisconsin (Diagnostic)** dataset to explore relationships between tumour cell characteristics and classify each sample as:

- **0 — Benign:** 357 samples (62.7%)
- **1 — Malignant:** 212 samples (37.3%)

The 30 numerical features describe cell-nucleus characteristics, including radius, texture, perimeter, area, smoothness, concavity and symmetry. I sourced the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) using `ucimlrepo`.

The important question was not just *"Which model gets the highest accuracy?"* but also *"How often does it miss a malignant case?"* That influenced how I interpreted recall, confusion matrices and ROC-AUC.

## Results: comparing four models

The table below reflects results saved in my [original Jupyter notebook](Data_Science_Breast_Cancer.ipynb), evaluated on **114 held-out samples** (71 benign, 43 malignant).

| Model | Training accuracy | Test accuracy | Malignant recall | Malignant cases missed (test) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Support Vector Classifier (SVC)** | 98.24% | **97.37%** | 0.98 | **1** | **1.00** |
| **Multi-Layer Perceptron (MLP)** | 99.56% | **97.37%** | 0.98 | **1** | **0.99** |
| **XGBoost** | 99.78% | **97.37%** | 0.95 | 2 | See notebook ROC plot |
| **K-Nearest Neighbours (KNN)** | 97.14% | 95.61% | 0.93 | 3 | See notebook ROC plot |

**What I took from these results:** Although SVC, MLP and XGBoost had the *same test accuracy*, they did not make exactly the same mistakes. SVC and MLP each missed **one** malignant case on the test set, while XGBoost missed **two**. KNN missed **three**. This reinforced why relying on accuracy alone would be misleading for a healthcare-related classification problem.

The SVC achieved the highest reported ROC-AUC (**1.00**) in this particular hold-out evaluation. That is a result on a **small test set**, not proof that the model is perfect or generalises to real clinical settings.

## My approach: OSEMN

I structured the work around the **OSEMN** data science methodology: **Obtain → Scrub → Explore → Model → Interpret**.

### 1. Obtain and understand the data

- Fetched the UCI dataset using `ucimlrepo` and inspected its **569 rows, 30 predictor features and target labels**.
- Checked class distribution and considered the imbalance between benign and malignant samples.
- Reviewed summary statistics and the scales of the different features before modelling.

### 2. Clean and prepare

- Checked for missing values; **none were found** in the predictors or target.
- Used descriptive statistics and boxplots to investigate outliers, particularly in **radius** and **area**.
- **Kept the extreme values** rather than deleting them automatically: unusually large measurements might represent meaningful cases, not simply bad data.
- Encoded diagnosis labels from **M → 1** and **B → 0**.
- Split the data **80% training / 20% testing** (`random_state=42`).
- Applied `StandardScaler` because features such as area and smoothness have very different numerical ranges.

### 3. Explore and visualise

I used **Matplotlib, Seaborn and Plotly** for visual exploration, including:

- **Correlation heatmaps** to see which measurements moved together.
- **Pairplots** to compare selected features across the two classes.
- **Histograms / density plots** to inspect class overlap and feature distributions.
- **Boxplots** to investigate possible outliers.
- **Class-distribution charts** to understand imbalance.

**Notable patterns:** radius, perimeter and area showed strong relationships; malignant samples often had larger perimeter, area and concave-point values. Other features, such as smoothness and compactness, showed more overlap between classes. These patterns helped inform model selection and evaluation — they were not treated as standalone clinical rules.

### 4. Model: trials, comparisons and decisions

This was not a case of training one model and accepting its score. I tested different modelling approaches and compared how they behaved.

#### Trial A — K-Nearest Neighbours (KNN)

I explored `KNeighborsClassifier` with **10 neighbours** as a simpler comparison model.

- **Test accuracy:** 95.61%
- **Malignant recall:** 0.93
- **Malignant cases missed:** 3

KNN provided a useful point of comparison, but its lower recall for malignant samples made it less attractive than the better-performing alternatives in this experiment.

#### Trial B — Feedforward neural network / MLP

I trained an `MLPClassifier` and used **GridSearchCV with 5-fold cross-validation** to compare **180 hyperparameter combinations (900 fits)**. The search included:

- Hidden-layer structures: `(50,)`, `(100,)` and `(100, 50)`
- Activations: `relu` and `tanh`
- Solvers: `adam` and `lbfgs`
- Different initial learning rates and maximum-iteration settings
- Early stopping

**Best configuration in the notebook:**

```python
MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1000,
    early_stopping=True,
    random_state=42,
)
```

**Outcome:** 97.37% test accuracy, **0.99 ROC-AUC** and **1 malignant case missed** on the test set.

**What I learned from comparing settings:** My report discusses how `tanh` showed slightly different performance in some comparisons but produced **7 false negatives for malignant cases on a training comparison**. This made me pay more attention to class-specific errors rather than choosing a setting from accuracy alone. The final grid search selected `relu`.

#### Trial C — Support Vector Classifier (SVC)

I compared SVC kernels (`linear`, `rbf`, `poly`, `sigmoid`), regularisation values (`C`), gamma choices and class weights using **GridSearchCV with 5-fold cross-validation**.

**Best configuration in the notebook:**

```python
SVC(
    C=0.01,
    kernel="linear",
    gamma="scale",
    class_weight="balanced",
    probability=True,
)
```

The `balanced` class weights addressed the unequal number of benign and malignant samples. The notebook also evaluates probabilities using a **0.5 decision threshold**.

**Outcome:** 97.37% test accuracy, **1.00 ROC-AUC** and **1 malignant case missed** on the test set.

This model matched MLP's test accuracy with a different modelling approach. I also created **PCA-based 2D decision-boundary visualisations** to explore how a separately fitted linear SVC separates classes in two dimensions. These visualisations illustrate the decision boundary; they are **not** the decision surface of the full 30-feature model.

#### Trial D — XGBoost

I also tested `XGBClassifier` with settings aimed at controlling complexity and handling imbalance, including:

- `n_estimators=400`, `learning_rate=0.04`, `max_depth=5`
- `subsample=0.8`, `colsample_bytree=0.8`
- `min_child_weight=3`, `gamma=0.2`
- `scale_pos_weight=2`, `reg_lambda=5`

**Outcome:** 97.37% test accuracy and **99.78% training accuracy**, but **2 malignant cases missed** on the test set.

This was a useful reminder that a model can fit training data very closely without clearly improving its performance on unseen samples. XGBoost equalled the headline test accuracy of MLP and SVC but had lower malignant recall in this run.

### 5. Interpret and evaluate

I compared the models using **accuracy, precision, recall, F1-score, confusion matrices and ROC curves**, rather than reporting a single score.

The **false-negative rate for malignant cases** was especially important to discuss: in the context of this dataset, predicting a malignant sample as benign is the mistake I most wanted to investigate.

I also considered the balance between model complexity, computational effort and performance. For this dataset, the **linear SVC was a strong practical choice**: it matched the highest test accuracy, obtained the highest recorded ROC-AUC and was simpler than the tuned MLP.

## Challenges, trial and error, and lessons

| Decision / challenge | What I tried or noticed | What it taught me |
| --- | --- | --- |
| Outlier handling | Used boxplots; kept extreme values instead of dropping them automatically | Investigate the meaning of unusual data before cleaning it away |
| Features on different scales | Applied standardisation after inspecting descriptive statistics | Preprocessing decisions can affect model comparisons |
| Choosing a model | Compared KNN, MLP, SVC and XGBoost rather than relying on one method | A more complex model does not automatically give better results |
| Hyperparameter choices | Tested multiple MLP architectures/activations and SVC kernels/regularisation settings | Tuning is experimental; explain the trade-offs behind the selected settings |
| Similar headline accuracies | Found three models with 97.37% test accuracy but different malignant recall | Inspect confusion matrices and class-specific metrics, not just accuracy |
| Visualising decisions | Used PCA to create a 2D SVC decision-boundary view | Visual explanations are helpful, but simplifications should be labelled clearly |

## Limitations and what I would improve next

There are several things I would strengthen if I developed this beyond the original coursework:

1. **Use a scikit-learn `Pipeline` for scaling within cross-validation.** The original notebook standardises the training data *before* `GridSearchCV`; placing scaling inside each CV fold would avoid information leakage between training and validation folds.
2. **Use stratified or repeated validation** and report uncertainty rather than relying on a single 114-sample test split.
3. **Tune decision thresholds based on validation data**, with particular attention to malignant recall and false negatives.
4. **Explore feature selection and class-imbalance strategies** (such as RFE or SMOTE) and check whether they improve performance without introducing leakage.
5. **Evaluate on an independent external dataset** before making any broader claims about generalisation. The current work is a classroom experiment, not a medical device.

These are proposed next steps, **not additional experiments I am claiming to have completed**.

## Tools used

**Python · Jupyter Notebook · Pandas · NumPy · Matplotlib · Seaborn · Plotly · scikit-learn · XGBoost · ucimlrepo**

Key methods: **EDA, data cleaning, feature scaling, 5-fold cross-validation, grid search, model comparison, ROC-AUC, confusion matrices, PCA visualisation**.
- Compare the accuracy–recall trade-off when choosing classification thresholds.
