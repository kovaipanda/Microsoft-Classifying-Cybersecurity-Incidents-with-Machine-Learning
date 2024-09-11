import pandas as pd

# Load your dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train1.csv")

# Convert Timestamp to datetime object
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract date and time components
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df['Second'] = df['Timestamp'].dt.second
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6
df['WeekOfYear'] = df['Timestamp'].dt.isocalendar().week
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Drop the original Timestamp column if it's no longer needed
df = df.drop(columns=['Timestamp'])

# Save the updated DataFrame to a new CSV file
df.to_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train2.csv", index=False)

# Verify saved file by printing first few rows (optional)
print("DataFrame with new features saved as train_with_features.csv")
print(df.head())
print(df.isnull().sum())
# Get unique values for the specified columns
unique_values = {
    'Category': df['Category'].unique(),
    'IncidentGrade': df['IncidentGrade'].unique(),
    'EntityType': df['EntityType'].unique(),
    'EvidenceRole': df['EvidenceRole'].unique()
}

# Print the unique values for each column
for column, values in unique_values.items():
    print(f"Unique values in {column}:")
    print(values)
    print()

# this whole process is repeated for test1 and saved in test2 file
