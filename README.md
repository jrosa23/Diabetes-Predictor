# Diabetes-Predictor-Model

## About
This project is designed to create a data machine learning model for predicting diabetes using the Logistic Regression algorithm, and a Flask web application for interactive risk assessment. The model analyzes patient health data from a diabetes health indicator dataset (link can be found below). The model also aims to provide insights into diabetes risk factors and assist in early detection. Again, our model uses logistic regression algorithm as well as Standard Scalar and regularization to find an optimized model.

Our goal is to create a machine learning model that can be used in a predictor to provide a risk score if an individual is at risk of developing diabetes. Being able to predict allows us to them to know whether or not the individual is at risk, which can aid early intervention efforts and stray away from the path of developing diabetes. Our model was optimized by testing our models with different hyperparameters. To which we tested the model at different C-Values(Regularization). 

We used a dataset found on Kaggle that containing diabetes health indicators. There are 3 target variable classes. 0=No diabetes, 1=prediabetes, 2=diabetes. As well as 21 feature variables. Some of those are described below. We converted the data to SQLite which allowed for easier retrieval with queries.

## Project Features
- **Machine Learning Model**: Optimized Logistic Regression with hyperparameter tuning
- **Web Interface**: User-friendly Flask application for risk assessment
- **Risk Factors Analysis**: Identifies key contributors to diabetes risk

## Featured Variables
- HighBP: Whether or not the patient has high blood pressure.
- CholCheck: Whether or not the patient has had a cholesterol check in the last 5 years.
- BMI: The BMI value of the patient.
- HvyAlcoholConsump: Wheter or not the patient is a heavy drinker. (For males it is more than 14 drinks a week, for females it is more than 7 drinks a week.
- AnyHealthcare: Whether or not the patient has healthcare coverage.
- Sex
- Age
- etc.

## Installation
1. Clone the repository using SSH key:
   ```sh
   git clone git@github.com:jrosa23/Diabetes-Predictor.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Diabetes-Predictor
   ```
3. Install dependencies (if applicable):
   ```sh
   pip install numpy pandas scikit-learn pyspark findspark joblib sqlalchemy
   ```

## Usage

The final notebook can be found in the Final folder and final data machine learning model can be found in the Resources folder under the name diabetes_logi_regress_model.pkl 
Our data can also be found under data folder in the rescources folder.

## Tableau Workbook

Here is the link to our [Tableau Work Book](PlaceLinkHere.com)

## Software/Tools/Languages Used

Python, Tableau, SQLite

## Data Source

Diabetes Health Indicators Dataset
- Dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)


## Ethical Consideration
This machine learning model is intended for educational and informational purposes only. Ethical considerations include:
- **Data Privacy**: Ensuring that patient data is anonymized and used responsibly.
- **Bias and Fairness**: The model should be tested for biases to ensure fair predictions across different demographics.
- **Interpretability**: Users should be aware of the model’s limitations and avoid making critical medical decisions based solely on its predictions.
- **Accountability**: This tool does not replace professional medical advice, and predictions should be validated by healthcare professionals.

