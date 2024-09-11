import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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

# Define model parameters for hyperparameter tuning
lgb_params = {
    'n_estimators': [100], #Best Hyperparameters: {'subsample': 0.8, 'num_leaves': 100, 'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.2, 'colsample_bytree': 0.8}
    'learning_rate': [ 0.2],
    'max_depth': [-1],
    'num_leaves': [100],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}

# Define LightGBM model
lgb_model = lgb.LGBMClassifier()

# Cross-validation settings
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Helper function to run RandomizedSearchCV and catch errors
def run_random_search(model, param_grid, X_train, y_train):
    try:
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, scoring='accuracy', cv=kfold, n_jobs=2, verbose=1, random_state=42)
        search.fit(X_train, y_train)
        print(f"Best Hyperparameters: {search.best_params_}")
        return search.best_estimator_, search.best_params_
    except Exception as e:
        print(f"Error with model {model.__class__.__name__}: {e}")
        return None, None

# Run Randomized Search for LightGBM
lgb_best, best_params = run_random_search(lgb_model, lgb_params, X_train, y_train)

# Save the best model using pickle
if lgb_best is not None:
    with open('best_lgb_model.pkl', 'wb') as model_file:
        pickle.dump(lgb_best, model_file)
    print("Model saved to 'best_lgb_model.pkl'")

# Evaluate the LightGBM model on the validation set
def evaluate_model(model, X_val, y_val):
    if model is None:
        return None, None, None, None, None
    
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

# Evaluate LightGBM model
lgb_accuracy, lgb_f1, lgb_precision, lgb_recall, lgb_report = evaluate_model(lgb_best, X_val, y_val)
if lgb_accuracy is not None:
    print("LightGBM Classification Report:\n", lgb_report)
    print(f"Accuracy: {lgb_accuracy}")
    print(f"Macro F1 Score: {lgb_f1}")
    print(f"Precision: {lgb_precision}")
    print(f"Recall: {lgb_recall}")
