import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load your dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train4.csv")

# Separate features and target variable
X = df.drop(columns=['IncidentGrade'])
y = df['IncidentGrade']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Print the feature importances DataFrame
print(feature_importances)

# Repeat the process for test4
