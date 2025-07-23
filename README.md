
# DonorsChoose Project Approval Prediction

This repository contains the end-to-end workflow for building a predictive model to automate the vetting of classroom project proposals for DonorsChoose.org. By leveraging project text, metadata, and resource information, we aim to predict whether a proposal will be approved (1) or rejected (0), improving scalability, consistency, and volunteer resource allocation.



---

## Problem Statement

DonorsChoose.org anticipates over 500,000 project proposals next year, creating challenges around:

* **Scalability:** Manual screening cannot keep up with projected volume.
* **Consistency:** Volunteer-based vetting introduces variability.
* **Resource Allocation:** Volunteers should focus on proposals needing the most review.

**Goal:** Build a classification model to predict `project_is_approved` (0 or 1) using proposal text and metadata, thereby streamlining initial screening, enforcing consistent criteria, and prioritizing borderline cases for human review.

---

## Data Description

### `train_data.csv`

* **id:** Unique project application ID
* **teacher\_id:** Teacher submitting the proposal
* **teacher\_prefix:** Title (Ms., Mr., etc.)
* **school\_state:** US state of the school
* **project\_submitted\_datetime:** Timestamp of submission
* **project\_grade\_category:** Grade levels
* **project\_subject\_categories:** High-level category (e.g., Music & The Arts)
* **project\_subject\_subcategories:** Sub-category (e.g., Visual Arts)
* **project\_title:** Title of the project
* **project\_essay\_1`–`4:** Four text fields describing the proposal
* **project\_resource\_summary:** Summary of requested resources
* **teacher\_number\_of\_previously\_posted\_projects:** Past submissions count
* **project\_is\_approved:** Target variable (0 = rejected, 1 = accepted)

### `resources.csv`

* **id:** Unique resource entry ID
* **description:** Resource description
* **price:** Unit price
* **quantity:** Number of units requested

---

## File Structure

```
├── Donor_Choose_Feature_Engineering&cleaning.ipynb   # Data cleaning & feature engineering
├── Donor_Choose_Model_Training.ipynb               # Model training & evaluation
├── train_data.csv                                  # Raw project-level data
├── resources.csv                                   # Raw resource-level data
└── README.md                                       # This file
```

---

## Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/donorschoose-approval-prediction.git
   cd donorschoose-approval-prediction
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Key libraries:* pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib, lightgbm, xgboost, mlflow

---

## Methodology

### Data Cleaning & Feature Engineering

* **Missing Value Handling:** Filled or dropped missing essays and resource entries.
* **Text Vectorization:** Transformed `project_essay_1–4` and `project_resource_summary` using `TfidfVectorizer` from scikit-learn.
* **Categorical Encoding:**

  * One-hot encoding for `project_grade_category` and `school_state`.
  * `MultiLabelBinarizer` for multi-select fields: `project_subject_categories` and `project_subject_subcategories`.
* **Numeric Scaling:** Standardized numeric features (e.g., total resource cost) with `StandardScaler`.
* **Class Imbalance:** Applied `SMOTE` (Synthetic Minority Over-sampling Technique) to balance approved vs. rejected classes.

### Exploratory Data Analysis

* **Univariate Analyses:** Approval rate distribution, essay length distribution, resource cost distribution.
* **Bivariate Analyses:** Approval rate by subject category, state-level approval heatmap, cost vs. approval scatter plots.
* **Missing & Outlier Checks:** Identified and handled missing values and extreme cost outliers.

### Modeling Approaches

1. **Train-Test Split:** `train_test_split` with stratification on the target.
2. **Baseline Model:** Logistic Regression.
3. **Advanced Models:**

   * Decision Tree (`DecisionTreeClassifier`)
   * Random Forest (`RandomForestClassifier`) with hyperparameter tuning (`RandomizedSearchCV`)
   * Gaussian Naive Bayes (`GaussianNB`)
   * XGBoost (`XGBClassifier`)
   * LightGBM (`LGBMClassifier`)
4. **Evaluation Metrics:**

   * **Accuracy, Precision, Recall, F1-score**
   * **ROC AUC**
   * **Classification Report & Confusion Matrix**
5. **Model Tracking:** Logged experiments and best parameters using `mlflow`.

---

## Model Training & Evaluation

The `Donor_Choose_Model_Training.ipynb` notebook contains:

* Loading preprocessed features and target
* Resampling with SMOTE
* Training each classifier
* Comparing performance via cross-validation and held-out validation set
* Selecting the best model based on F1-score and ROC AUC

---

## Deployment & Tracking

* **Model Serialization:** Best model saved with MLflow’s `model.log()` functionality.
* **API Integration (Future Work):** Plan to wrap the model in a Flask or FastAPI endpoint for real-time predictions.

---

## Usage

Run the notebooks interactively in the following order:

1. **Feature Engineering & Cleaning**

   ```bash
   jupyter notebook Donor_Choose_Feature_Engineering&cleaning.ipynb
   ```
2. **Model Training & Evaluation**

   ```bash
   jupyter notebook Donor_Choose_Model_Training.ipynb
   ```

---

## Acknowledgements

* DonorsChoose.org for the dataset and mission.
* scikit-learn, imbalanced-learn, XGBoost, LightGBM teams for the open-source tools.

---

*This README was generated to outline the methods and workflow used in the DonorsChoose project approval prediction.*
