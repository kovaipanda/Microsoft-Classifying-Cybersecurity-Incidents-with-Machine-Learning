import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model from the pickle file
with open('best_lgb_model.pkl', 'rb') as model_file:
    loaded_lgb_model = pickle.load(model_file)
print("Model loaded from 'best_lgb_model.pkl'")

# Load the new test dataset
test_df = pd.read_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\test5.csv")

# Prepare the test dataset (remove target variable if present, assume 'IncidentGrade' as target)
X_test = test_df.drop('IncidentGrade', axis=1, errors='ignore')  # Remove target variable if present

# Make predictions on the test dataset using the loaded model
y_test_pred = loaded_lgb_model.predict(X_test)

# If the test set contains actual labels, you can compare the predictions to the ground truth
# Assuming the test set contains the true 'IncidentGrade' column for evaluation
if 'IncidentGrade' in test_df.columns:
    y_test_actual = test_df['IncidentGrade']
    
    # Evaluate the model
    accuracy = accuracy_score(y_test_actual, y_test_pred)
    report = classification_report(y_test_actual, y_test_pred)
    
    # Calculate precision, recall, and F1 scores (macro)
    precision = precision_score(y_test_actual, y_test_pred, average='macro')
    recall = recall_score(y_test_actual, y_test_pred, average='macro')
    f1 = f1_score(y_test_actual, y_test_pred, average='macro')

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)
    print(f"Macro Precision: {precision}")
    print(f"Macro Recall: {recall}")
    print(f"Macro F1-Score: {f1}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test_actual, y_test_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=set(y_test_actual), yticklabels=set(y_test_actual))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# If there's no actual 'IncidentGrade' column in the test set, just print the predictions
else:
    print("Predictions on test data:\n", y_test_pred)
