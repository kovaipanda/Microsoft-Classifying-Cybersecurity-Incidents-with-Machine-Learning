import pandas as pd
import pickle
import xgboost as xgb

# Step 1: Load the trained XGBoost model from the pickle file
with open('best_xgb_model.pkl', 'rb') as model_file:
    xgb_loaded_model = pickle.load(model_file)

print("Model loaded successfully from 'best_xgb_model.pkl'")

# Step 2: Load the new dataset (test5.csv)
test_data = pd.read_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\test5.csv")

# Step 3: Ensure that the new dataset has the same features as the training data
# (Make sure there is no target column like 'IncidentGrade' in test data)
X_test = test_data  # Assuming 'test5.csv' does not contain the target column

# Step 4: Use the loaded model to make predictions
y_test_pred = xgb_loaded_model.predict(X_test)

# Print the predicted labels
print("Predictions on the test data:")
print(y_test_pred)

# If you need to save predictions to a CSV file
output_df = pd.DataFrame(y_test_pred, columns=['Predicted_IncidentGrade'])
output_df.to_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\test_predictions.csv", index=False)
print("Predictions saved to 'test_predictions.csv'")
