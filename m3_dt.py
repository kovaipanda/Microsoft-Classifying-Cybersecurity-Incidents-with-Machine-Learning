import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train5.csv")

# Split the dataset into features and target variable
X = df.drop('IncidentGrade', axis=1)
y = df['IncidentGrade']

# Split into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model
dt_model.fit(X_train, y_train)

# Save the trained model using pickle
with open('best_dt_model.pkl', 'wb') as model_file:
    pickle.dump(dt_model, model_file)
print("Decision Tree model saved to 'best_dt_model.pkl'")

# Evaluate the Decision Tree model on the validation set
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    
    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    
    # Macro F1 Score, Precision, Recall
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    precision = precision_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    
    # Classification Report
    report = classification_report(y_val, y_pred, target_names=[str(i) for i in set(y_val)])
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=set(y_val), yticklabels=set(y_val))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.show()
    
    return accuracy, macro_f1, precision, recall, report

# Evaluate Decision Tree model
dt_accuracy, dt_f1, dt_precision, dt_recall, dt_report = evaluate_model(dt_model, X_val, y_val)
print("Decision Tree Classification Report:\n", dt_report)
print(f"Accuracy: {dt_accuracy}")
print(f"Macro F1 Score: {dt_f1}")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
