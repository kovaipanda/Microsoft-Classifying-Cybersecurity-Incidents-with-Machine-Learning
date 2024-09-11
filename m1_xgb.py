import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the test dataset
test_df = pd.read_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train5.csv")

# Assuming the test set has the same feature columns as the train set, except for 'IncidentGrade'
# Remove 'IncidentGrade' if it exists in the test set
if 'IncidentGrade' in test_df.columns:
    X_test = test_df.drop('IncidentGrade', axis=1)
    y_test = test_df['IncidentGrade']
else:
    X_test = test_df  # Test data without target labels

# Load the trained XGBoost model from the pickle file
with open('best_xgb_model.pkl', 'rb') as model_file:
    xgb_loaded_model = pickle.load(model_file)

# Predict using the loaded model on the test dataset
y_pred_test = xgb_loaded_model.predict(X_test)

# (Optional) If you have ground truth 'IncidentGrade' in the test set, evaluate the model
if 'IncidentGrade' in test_df.columns:
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_test)
    macro_f1 = f1_score(y_test, y_pred_test, average='macro')
    precision = precision_score(y_test, y_pred_test, average='macro')
    recall = recall_score(y_test, y_pred_test, average='macro')

    # Classification report
    report = classification_report(y_test, y_pred_test, target_names=[str(i) for i in set(y_test)])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix for Test Data')
    plt.show()
    
    # Print evaluation results
    print("Test Set Classification Report:\n", report)
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1 Score: {macro_f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
else:
    print("Predictions on the test data:\n", y_pred_test)
