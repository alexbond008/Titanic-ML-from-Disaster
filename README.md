
# 🚢 Titanic - Machine Learning from Disaster

Welcome to the Titanic ML project! This repository contains a complete, hands-on solution for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition. The goal: predict which passengers survived the Titanic shipwreck using modern machine learning techniques.

<p align="center">
	<img src="titanic_sinking.png" alt="Titanic Sinking" width="500"/>
</p>

---

## 🌊 Project Overview

- **Data:** Passenger demographics and information for 891 of the 2224 people aboard the Titanic.
- **Objective:** Predict survival based on features like age, sex, class, fare, and more.
- **Approach:** Data cleaning, feature engineering, and training multiple ML models for robust predictions.

---

## 📁 Repository Structure

```text
.
├── titanic.ipynb                # Main notebook: EDA, cleaning, logistic regression
├── random_forest_titanic.ipynb  # Random Forest model notebook
├── xgboost.ipynb                # XGBoost model notebook
├── requirements.txt             # Python dependencies
├── titanic_data/                # Data and submission files
│   ├── train.csv
│   ├── test.csv
│   ├── train_cleaned.csv
│   ├── test_cleaned.csv
│   ├── random_forest_submission.csv
│   ├── xgboost_submission.csv
│   ├── logistic_regression_submission.csv
│   └── my_submission.csv
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Titanic-ML-from-Disaster.git
cd Titanic-ML-from-Disaster
```

### 2. Install Dependencies

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Download the Data

- Download `train.csv` and `test.csv` from the [Kaggle Titanic competition page](https://www.kaggle.com/competitions/titanic/data).
- Place them in the `titanic_data/` directory.

### 4. Run the Notebooks

You can run the analysis and model training in Jupyter Notebook:

```bash
jupyter notebook titanic.ipynb
```

---

## 🧠 Workflow & Notebooks

### 1. **Data Exploration & Cleaning** (`titanic.ipynb`)

- **Exploration:** Visualize missing data, distributions, and relationships using `seaborn` and `matplotlib`.
- **Cleaning:**
	- Drop columns with excessive missing values (e.g., `Cabin`).
	- Impute missing `Age` based on `Pclass`.
	- Fill missing `Embarked` values with the most common port.
	- Fill missing `Fare` values in test set with median.
- **Feature Engineering:**
	- Encode categorical variables (`Sex`, `Embarked`) using one-hot encoding.
	- Create a `Family` feature by combining `SibSp` and `Parch`.
	- Correlation analysis to select relevant features.
- **Export:** Save cleaned datasets as `train_cleaned.csv` and `test_cleaned.csv` for modeling.

### 2. **Logistic Regression Model** (`titanic.ipynb`)

- Train a logistic regression model on the cleaned data.
- Evaluate with accuracy and classification report.
- Generate predictions and save as `logistic_regression_submission.csv`.

### 3. **Random Forest Model** (`random_forest_titanic.ipynb`)

- Train a Random Forest classifier (`n_estimators=100`).
- Evaluate with classification report and feature importances.
- Generate predictions and save as `random_forest_submission.csv`.

### 4. **XGBoost Model** (`xgboost.ipynb`)

- Train an XGBoost classifier for improved performance.
- Evaluate with classification report.
- Generate predictions and save as `xgboost_submission.csv`.

---

## 🧹 Data Preprocessing & Cleaning Highlights

- **Missing Values:**
	- Visualized with heatmaps.
	- Imputed `Age` using median values per `Pclass`.
	- Dropped `Cabin` due to high missingness.
	- Filled `Embarked` and `Fare` with sensible defaults.
- **Feature Engineering:**
	- One-hot encoding for categorical features.
	- Created `Family` size feature.
	- Dropped irrelevant columns (`Ticket`, `Name`, `PassengerId`).

---

## 🤖 Machine Learning Models Used

| Model                | Notebook                    | Output File                        |
|----------------------|-----------------------------|------------------------------------|
| Logistic Regression  | `titanic.ipynb`             | `logistic_regression_submission.csv`|
| Random Forest        | `random_forest_titanic.ipynb`| `random_forest_submission.csv`      |
| XGBoost              | `xgboost.ipynb`             | `xgboost_submission.csv`            |

All models are trained on the cleaned data and evaluated with classification metrics.

---

## 📊 Example Submissions

Each notebook outputs a submission file in `titanic_data/`, ready for Kaggle upload:

- `logistic_regression_submission.csv`
- `random_forest_submission.csv`
- `xgboost_submission.csv`

---

## 📚 References & Resources

- [Kaggle Titanic Competition Overview](https://www.kaggle.com/competitions/titanic/overview)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/)

---

## 📝 License

This project is for educational purposes.
