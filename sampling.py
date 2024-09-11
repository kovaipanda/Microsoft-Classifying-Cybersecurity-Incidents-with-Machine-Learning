import pandas as pd
from sklearn.utils import resample

# Load your dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train3.csv")

# Identify the target column
target_column = 'IncidentGrade'

# Get the counts of each class in the target column
class_counts = df[target_column].value_counts()

# Find the smallest class size
min_class_size = class_counts.min()

# Separate the dataset into classes
classes = [df[df[target_column] == label] for label in class_counts.index]

# Undersample each class to the smallest class size
resampled_classes = [resample(cls, 
                              replace=False, 
                              n_samples=min_class_size, 
                              random_state=42) for cls in classes]

# Combine the resampled classes into a balanced DataFrame
df_balanced = pd.concat(resampled_classes)

# Shuffle the DataFrame to mix the rows
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced DataFrame to a new CSV file
df_balanced.to_csv(r"C:\Users\DELL\Downloads\cyber_security_microsoft\train4.csv", index=False)

# Verify saved file by printing first few rows and class distribution (optional)
print("DataFrame with balanced classes saved as train_balanced.csv")
print(df_balanced[target_column].value_counts())

#Repeat the whole process for test3 and save in test4
