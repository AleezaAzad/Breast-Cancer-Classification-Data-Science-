# Breast Cancer Classification
This is an individual data science project I completed as part of my **BSc (Hons) Applied Artificial Intelligence** degree at the **University of Bradford**. I used the **Breast Cancer Wisconsin (Diagnostic)** dataset to explore patterns in cell-nucleus measurements and compare two machine-learning approaches for classifying samples as **benign** or **malignant**.

The project covers the full data science workflow, from understanding and cleaning the data to visualising patterns, training models and evaluating their results.

> **Note:** This is an academic machine-learning project using an existing labelled dataset. It is not a clinical diagnostic tool and should not be used to make healthcare decisions.

## Project at a glance

| | Details |
|---|---|
| **Dataset** | Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository |
| **Samples** | 569 (357 benign, 212 malignant) |
| **Features** | 30 numerical measurements of cell nuclei |
| **Task** | Binary classification: benign (0) vs malignant (1) |
| **Models** | Support Vector Classifier (SVC) and Multi-Layer Perceptron (MLP) |
| **Evaluation** | 80/20 train/test split, 5-fold cross-validation, Grid Search |
| **Best reported test accuracy** | **97.37%** (both models) |
| **Best reported ROC-AUC** | **1.00** (SVC on the held-out test set) |

## Objectives

- Explore the dataset to understand the relationships between features and diagnosis.
- Prepare the data for modelling through cleaning, encoding and scaling.
- Compare machine-learning models rather than relying on a single approach.
- Evaluate results using more than accuracy, particularly **recall** and **false negatives**, which matter when the positive class represents malignant cases.

## Dataset

**Source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

The dataset contains 30 numerical features describing characteristics of cell nuclei, including **radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry** and **fractal dimension**. The target is the diagnosis: **M (malignant)** or **B (benign)**.

The original diagnosis labels were encoded as **malignant = 1** and **benign = 0** for modelling.

## Approach: OSEMN methodology

I followed the **OSEMN** framework to keep the project organised while allowing room to revisit earlier stages when an approach needed changing.

### 1. Obtain and understand the data

- Reviewed the dataset structure, feature definitions and distribution of benign and malignant cases.
- Checked the relationship between the target label and the 30 input features.

### 2. Scrub and preprocess

- Checked data quality and investigated outliers, rather than automatically removing observations that could contain meaningful medical information.
- Encoded diagnosis labels into binary values.
- Used an **80/20 train/test split** and **StandardScaler** to standardise features with different numerical ranges.

### 3. Explore and visualise

Used **Matplotlib, Seaborn and Plotly** for data exploration, including:

- **Correlation heatmaps** to identify related measurements.
- **Pair plots** to compare feature relationships across the two diagnosis classes.
- **Histograms and other distribution plots** to explore differences between benign and malignant samples.

The analysis found that measurements such as **radius, perimeter, area and concave points** showed useful differences between the classes, while other features had more overlapping distributions.

### 4. Model

Compared two approaches:

- **Multi-Layer Perceptron (MLP):** A feedforward neural network capable of learning non-linear patterns.
- **Support Vector Classifier (SVC):** A classical machine-learning classifier; the reported selected configuration used a **linear kernel** and balanced class weights.

I used **GridSearchCV with 5-fold cross-validation** to compare hyperparameter choices. K-nearest neighbours (KNN) was also explored during experimentation, but the final comparison focused on MLP and SVC.

### 5. Interpret and evaluate

Evaluated the models using **confusion matrices, accuracy, precision, recall, F1-score and ROC-AUC**, paying particular attention to malignant cases that could be incorrectly classified as benign.

## Results

The following results were reported in the project evaluation:

| Model | Test accuracy | Test ROC-AUC | False negatives on test set | False positives on test set |
|---|---:|---:|---:|---:|
| **MLP** | **97.37%** | **0.99** | 1 | 2 |
| **SVC** | **97.37%** | **1.00** | 1 | 2 |

Both models performed well on the held-out test set and made the same number of classification errors. The SVC achieved the higher reported ROC-AUC. However, this was a **small dataset**, so the metrics should be interpreted as project results rather than evidence of real-world clinical readiness.

## Tools and technologies

- **Python** — data preparation, analysis and modelling
- **Pandas, NumPy** — working with numerical and tabular data
- **Matplotlib, Seaborn, Plotly** — exploratory analysis and visualisation
- **Scikit-learn** — preprocessing, SVC, MLP, Grid Search and evaluation
- **Jupyter Notebook** — experiments and results

## View or run the project

The main notebook is [`Data_Science_Breast_Cancer.ipynb`](./Data_Science_Breast_Cancer.ipynb).

1. Clone this repository:

   ```bash
   git clone https://github.com/AleezaAzad/Breast-Cancer-Classification-Data-Science-.git
   cd Breast-Cancer-Classification-Data-Science-
   ```

2. Set up a Python environment and install the main libraries:

   ```bash
   python -m pip install notebook pandas numpy matplotlib seaborn plotly scikit-learn
   ```

3. Open the notebook:

   ```bash
   jupyter notebook Data_Science_Breast_Cancer.ipynb
   ```

4. Run the cells in order. **The repository currently contains the notebook and README, but not a separate dataset file.** If your local copy of the notebook expects a CSV file, download the [UCI dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and update the file path or data-loading step before running it.

> The setup steps list the main libraries used in the report. Depending on your Python environment and the notebook's current data-loading code, an additional package or path adjustment may be needed.

## What I learned

This project gave me practical experience in **data cleaning, exploratory analysis, testing different models and interpreting results**. One of the most important lessons was that a high accuracy score does not tell the full story — you also need to consider class balance, false negatives, the choice of evaluation metric and how reliably the model performs on unseen data.

## Possible next steps

- Explore **feature selection** to reduce redundant inputs.
- Investigate additional approaches to class imbalance, such as **SMOTE**, while applying them only within training folds.
- Evaluate stability across repeated splits or on a separate external dataset.
- Compare the accuracy–recall trade-off when choosing classification thresholds.
