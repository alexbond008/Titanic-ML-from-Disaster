# Titanic - Machine Learning from Disaster

This project is a solution for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition. The goal is to predict which passengers survived the Titanic shipwreck using machine learning techniques.

![alt text](titanic_sinking.png)


## Project Overview

- **Data:** The dataset contains demographic and passenger information from 891 of the 2224 passengers and crew on board the Titanic.
- **Objective:** Predict survival on the Titanic based on features such as age, sex, class, fare, and more.
- **Approach:** Data preprocessing, feature engineering, and training a Random Forest classifier to make predictions.

## Repository Structure

```
.
├── titanic.ipynb           # Main Jupyter notebook with code and analysis
├── requirements.txt        # Python dependencies
├── titanic_data/           # Folder for train/test data and submission file
│   ├── train.csv
│   ├── test.csv
│   └── my_submission.csv
└── README.md               # Project documentation
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Titanic-ML-from-Disaster.git
cd Titanic-ML-from-Disaster
```

### 2. Install Dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Download the Data

- Download `train.csv` and `test.csv` from the [Kaggle Titanic competition page](https://www.kaggle.com/competitions/titanic/data).
- Place them in the `titanic_data/` directory.

### 4. Run the Notebook

You can run the analysis and model training in Jupyter Notebook:

```bash
jupyter notebook titanic.ipynb
```

## Notebook Workflow

- **Data Exploration:** Load and inspect the data.
- **Preprocessing:** Handle missing values, encode categorical variables, and select features.
- **Model Training:** Train a Random Forest classifier.
- **Evaluation:** Check training accuracy and feature importances.
- **Prediction:** Generate predictions for the test set and save the submission file.

## Example Submission

The notebook will output a `my_submission.csv` file in the `titanic_data/` folder, ready for submission to Kaggle.

## References

- [Kaggle Titanic Competition Overview](https://www.kaggle.com/competitions/titanic/overview)
- [scikit-learn documentation](https://scikit-learn.org/stable/)

## License

This project is for educational purposes.
