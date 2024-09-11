Cybersecurity Incident Classification Using Machine Learning

This project involves classifying cybersecurity incidents into three categories (BenignPositive, FalsePositive ,TruePositive ) using machine learning models.

Here’s a summary and guide for each step and script involved:

1. Exploratory Data Analysis (EDA) - eda1.py
   
Objective:

Clean the dataset by removing columns and rows with missing values and irrelevant information.

Key Steps:

Load Dataset:

Load train.csv using Pandas.

Identify Missing Values:

Determine which columns have missing values.

Column Selection:

Keep only relevant columns, e.g., IncidentGrade.

Drop columns with missing values that are not crucial.

Data Cleaning:

Drop rows where IncidentGrade is missing.

Save Cleaned Data:

Save cleaned dataset as train1.csv.

Apply similar cleaning to test.csv and save as test1.csv.

Outputs:

Print a summary of missing values and first few rows of cleaned data.

Usage:

Update file paths if necessary and run the script to prepare the data.

2. Feature Engineering and Dataset Preparation - eda2.py
   
Objective:

Enhance the dataset with additional time-based features from the Timestamp field.

Key Steps:

Load Cleaned Dataset:

Load train1.csv.

Timestamp Conversion:

Convert Timestamp to a datetime object.

Extract Time Features:

Extract year, month, day, hour, etc.

Create binary feature IsWeekend.

Drop Original Timestamp Column:

Remove the Timestamp column after extraction.

Save Enhanced Dataset:

Save as train2.csv.

Apply similar feature extraction to test1.csv and save as test2.csv.

Outputs:

Print unique values for important categorical columns and the first few rows of enhanced data.

Usage:

Ensure train1.csv and test1.csv are in place and run the script to add time-based features.

3. Feature Encoding - eda3.py
   
Objective:

Encode categorical variables using One-Hot and Label Encoding.

Key Steps:

Load Dataset:

Load train2.csv.

Initialize Encoders:

One-Hot Encode: For nominal variables.

Label Encode: For ordinal variables.

One-Hot Encoding:

Apply to Category and EntityType.

Label Encoding:

Apply to IncidentGrade and EvidenceRole.

Drop Original Categorical Columns:

Remove original columns after encoding.

Dropping Additional Features:

Remove extra features that are not in the test dataset.

Save Encoded Dataset:

Save as train3.csv.

Apply similar encoding to test2.csv and save as test3.csv.

Outputs:

Print encoded dataset and encoding mappings.

Usage:

Run the script with train2.csv and test2.csv to prepare encoded datasets.

4. Class Balancing - sampling.py
   
Objective:

Address class imbalance in the dataset using undersampling.

Key Steps:

Load Dataset:

Read train3.csv.

Check Class Distribution:

Identify the class with the smallest number of samples.

Undersampling:

Resample each class to match the size of the smallest class.

Combine Resampled Data:

Concatenate and shuffle the balanced dataset.

Save Balanced Dataset:

Save as train4.csv.

Apply similar balancing to test3.csv and save as test4.csv.

Outputs:

Print the class distribution of the balanced dataset.

Usage:

Execute the script to balance your dataset before model training.

5. Feature Importance - features.py
   
Objective:

Evaluate feature importance using a Random Forest Classifier.

Key Steps:

Load Dataset:

Load train4.csv.

Separate Features and Target:

Define X (features) and y (target).

Train-Test Split:

Split the data into training and validation sets.

Train Random Forest Classifier:

Train using X_train and y_train.

Calculate Feature Importance:

Use feature_importances_ to get feature importance scores.

Create Feature Importance DataFrame:

Create and sort DataFrame by importance.

Print Results:

Display the feature importance DataFrame.

Usage:

Run this script to identify key features influencing the target variable.

6. Data Normalization - normalization.py
   
Objective:

Normalize feature values using Min-Max Scaling.

Key Steps:

Load Dataset:

Load train4.csv.

Separate Features and Target:

Define X and y.

Min-Max Scaling:

Apply scaling to features between 0 and 1.

Recombine Target Variable:

Merge normalized features with the target variable.

Save Normalized Data:

Save as train5.csv.

Apply normalization to test4.csv and save as test5.csv.

Usage:

Use this script to normalize your dataset for algorithms that require scaled features.

7. Model Training and Evaluation - m1_xgb.py
   
Objective:

Train an XGBoost model and evaluate it on the test dataset.

Key Steps:

Loading the Test Dataset:

Load test5.csv and separate features and target if available.

Loading the XGBoost Model:

Load pre-trained model from best_xgb_model.pkl.

Making Predictions:

Predict using the model.

Model Evaluation:

Compute metrics like accuracy, F1 score, precision, and recall if ground truth is available.

Display confusion matrix.

Usage:

Run the script to evaluate the XGBoost model's performance on the test set.

8. Predictions with XGBoost - m1_xgb_test.py
   
Objective:

Generate predictions using a pre-trained XGBoost model on new test data.

Key Steps:

Load Trained Model:

Load model from best_xgb_model.pkl.

Load Test Dataset:

Load test5.csv.

Ensure Feature Compatibility:

Ensure the test data has the same features as training data.

Make Predictions:

Predict labels for the test dataset.

Output Predictions:

Print and save predictions to test_predictions.csv.

Usage:

Use this script to generate and save predictions for new test data.

9. Hyperparameter Tuning and Model Evaluation - m2_lg.py
    
Objective:

Perform hyperparameter tuning for a LightGBM model and evaluate its performance.

Key Steps:

Load Dataset:

Load train5.csv.

Split Data:

Create training and validation sets.

Define Hyperparameters:

Set up hyperparameters for LightGBM.

Initialize and Tune LightGBM:

Perform Randomized Search for hyperparameter tuning.

Evaluate Model:

Calculate metrics and plot confusion matrix.

Print Results:

Display evaluation metrics and confusion matrix.

Usage:

Run this script for hyperparameter tuning and evaluating the LightGBM model.

10. Predictions with LightGBM - m2_lg_test.py
    
Objective:

Generate predictions using a pre-trained LightGBM model on new test data.

Key Steps:

Load Trained Model:

Load model from best_lgb_model.pkl.

Load Test Dataset:

Load test5.csv.

Prepare Test Data:

Ensure the test data is formatted correctly.

Make Predictions:

Generate predictions using the model.

Evaluate Model (if labels are present):

Compare predictions with actual labels and compute metrics.

Output Predictions:

Print and save predictions to test_predictions.csv.

Usage:

Run this script to make and save predictions with the LightGBM model.

Overall Workflow:

Clean and prepare datasets (eda1.py, eda2.py, eda3.py).

Balance classes (sampling.py).

Assess feature importance (features.py).

Normalize data (normalization.py).

Train and evaluate models (m1_xgb.py, m2_lg.py).

Generate predictions (m1_xgb_test.py, m2_lg_test.py).

Ensure you follow the order of the scripts for seamless data processing and model evaluation.

DATA ANALYSIS:

EDA (Exploratory Data Analysis):

eda1: Initial data overview showing columns and their data types. There were missing values in IncidentGrade.

eda2: Added new time-based features (Year, Month, Day, etc.) and saved the DataFrame with new features.

eda3: Showed unique values in categorical columns (Category, IncidentGrade, EntityType, EvidenceRole) and encoding mappings.

IncidentGrade Encoding Mapping:
0 -> BenignPositive
1 -> FalsePositive
2 -> TruePositive

Model Training and Evaluation

LightGBM Model (m2):

Train Results:

Accuracy: 0.9465541803935535

Macro F1 Score: 0.9465900257107025

Precision: 0.9467997563226564

Recall: 0.9465541803935537


![image](https://github.com/user-attachments/assets/2e71f3d0-34ec-4ae2-af67-fb8de67d0131)

Test Results:

Accuracy: 0.9040005139909205

Macro F1 Score: 0.904019332303855

Precision: 0.9040486335261567

Recall: 0.9040005139909205

![image](https://github.com/user-attachments/assets/6ae4e50a-a430-4486-8f56-32b2eb98d8b6)

Decision Tree Model (m3):

Train Results:

Accuracy: 0.9595464933368949

Macro F1 Score: 0.9595478308056423

Precision: 0.9595500125690446

Recall: 0.9595464933368949

![image](https://github.com/user-attachments/assets/7c110da8-1819-4d1d-bd84-fbf1f44d9358)

Test Results:

Accuracy: 0.8764494324926954

Macro F1 Score: 0.8763264324926954

Precision: 0.8695500125687446

Recall: 0.8795464933539949

![image](https://github.com/user-attachments/assets/d4f54a7e-ab0d-4f10-99c7-3defbcda63f3)


XGBoost Model (m1):

Train Results:

Accuracy: 0.9336286468999495

Macro F1 Score: 0.9336665901496163

Precision: 0.9339990601068985

Recall: 0.9336286468999496

![image](https://github.com/user-attachments/assets/e14c5ef8-9d97-4719-84ee-3a3ebfbb8b7c)

Test Results:

Accuracy: 0.8861495509150352

Macro F1 Score: 0.8859237182744844

Precision: 0.8865937934362286

Recall: 0.8861495509150353

![image](https://github.com/user-attachments/assets/550206cd-8618-46c2-807b-4e74f8c578f5)

Feature Importance

LightGBM Model Features: OrgId, IncidentId, DetectorId, and AlertId are among the top features.

XGBoost Model Features: Similar to LightGBM, with OrgId and IncidentId being important.

Normalization and Missing Values

eda1: Identified missing values in several columns. The MitreTechniques, ActionGrouped, and ActionGranular columns had significant missing values.

eda2: Reviewed unique values in categorical columns.

Observations and Recommendations

Model Performance: LightGBM and XGBoost performed well, with LightGBM showing slightly better accuracy on the test set. Decision Tree performed better on the training data but less effectively on the test set.

Feature Importance: Key features include OrgId, IncidentId, and DetectorId. Ensure these are correctly processed and validated.

Handling Missing Values: Address missing values in columns with significant gaps. Consider imputation or other methods to handle these.
