import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Load your dataset
file_path = r"C:\Users\DELL\Downloads\cyber_security_microsoft\train4.csv"

df = pd.read_csv(file_path)

# Separate features and target variable
X = df.drop(columns=['IncidentGrade'])
y = df['IncidentGrade']

# Normalize features using Min-Max Scaling
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Convert normalized data back to DataFrame
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Add the target variable back to the normalized features
df_normalized = pd.concat([X_normalized_df, y.reset_index(drop=True)], axis=1)

# Save the cleaned and normalized data to a new CSV file
output_file_path = r"C:\Users\DELL\Downloads\cyber_security_microsoft\train5.csv"
df_normalized.to_csv(output_file_path, index=False)

print(f"Data normalized successfully. Saved to {output_file_path}")

#repeat the whole process for test4 and save in test5
