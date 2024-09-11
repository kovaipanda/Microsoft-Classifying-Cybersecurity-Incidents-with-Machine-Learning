import pandas as pd

# Load your dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train.csv")

# Get a list of columns with missing values
missing_columns = df.columns[df.isnull().any()]

# Define columns to keep (including IncidentGrade) and drop
columns_to_keep = ['IncidentGrade']
columns_to_drop = [col for col in missing_columns if col not in columns_to_keep] + ['MitreTechniques'] # + ['Usage']

# Drop those columns
df_cleaned = df.drop(columns=columns_to_drop)

# Drop rows where 'IncidentGrade' has missing values
df_cleaned = df_cleaned.dropna(subset=['IncidentGrade'])

# Save the cleaned DataFrame to a new CSV file called 'train1.csv'
df_cleaned.to_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train1.csv", index=False) 

# Verify saved file by printing first few rows (optional)
print("DataFrame saved as test1.csv")
print(df_cleaned.head())
print(df.isnull().sum())


#This whole process is repeated for test dataset and saved in test1
