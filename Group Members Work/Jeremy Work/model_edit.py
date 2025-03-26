import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 1. Create synthetic data with matching features (21 features)
X, y = make_classification(
    n_samples=1000,
    n_features=21,  # Matching your original model's feature count
    n_informative=15,
    n_redundant=3,
    random_state=42
)

# 2. Create and train a new Logistic Regression model
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=100,
    random_state=42
)
model.fit(X, y)

# 3. Set feature names to match your original model
model.feature_names_in_ = np.array([
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex',
    'Age', 'Education', 'Income'
])

# 4. Verify the model works
print("Testing model predictions:")
dummy_input = np.zeros((1, 21))  # One sample with 21 features
print("Prediction:", model.predict(dummy_input))
print("Probability:", model.predict_proba(dummy_input))

# 5. Save the new model
new_model_path = 'diabetes_model_2.pkl'
with open(new_model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"\nSuccessfully created new working model at: {new_model_path}")
print("Model contains all required methods:")
print("- predict() -", hasattr(model, 'predict'))
print("- predict_proba() -", hasattr(model, 'predict_proba'))
print("- feature_names_in_ -", hasattr(model, 'feature_names_in_'))