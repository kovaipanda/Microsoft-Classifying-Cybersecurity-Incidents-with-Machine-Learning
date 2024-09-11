import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load your dataset
df = pd.read_csv(r"C:\Users\Jeffery\Downloads\cyber_security_microsoft\train2.csv")

# Initialize encoders
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')  # Drop first to avoid multicollinearity
label_encoder = LabelEncoder()

# One-Hot Encoding for 'Category'
category_encoded = one_hot_encoder.fit_transform(df[['Category']])
category_encoded_df = pd.DataFrame(category_encoded, columns=one_hot_encoder.get_feature_names_out(['Category']))
df = pd.concat([df, category_encoded_df], axis=1)

# Label Encoding for 'IncidentGrade'
df['IncidentGrade'] = label_encoder.fit_transform(df['IncidentGrade'])
# Create a mapping of encoded values to original labels
encoding_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

# Print the mapping
print("IncidentGrade Encoding Mapping:")
for encoded_value, original_label in encoding_mapping.items():
    print(f"{encoded_value} -> {original_label}")
    
# One-Hot Encoding for 'EntityType'
entity_type_encoded = one_hot_encoder.fit_transform(df[['EntityType']])
entity_type_encoded_df = pd.DataFrame(entity_type_encoded, columns=one_hot_encoder.get_feature_names_out(['EntityType']))
df = pd.concat([df, entity_type_encoded_df], axis=1)

# Label Encoding for 'EvidenceRole'
df['EvidenceRole'] = label_encoder.fit_transform(df['EvidenceRole'])

# Drop the original categorical columns if no longer needed
df = df.drop(columns=['Category', 'EntityType', 'EvidenceRole'])
# Drop the extra features in the train dataset because taining dataset has 85 features whereas test dataset has 82 features
df = df.drop(columns=['EntityType_KubernetesCluster','EntityType_GoogleCloudResource','Category_Weaponization'])

# Save the updated DataFrame to a new CSV file
df.to_csv(r"C:\Users\Jeffery\Downloads\cyber_security_microsoft\train3.csv", index=False)

# Verify saved file by printing first few rows (optional)
print("DataFrame with encoded features saved as train_encoded.csv")
print(df.head())

#Repeate the whole process for test2 and save in test3
