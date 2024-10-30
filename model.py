import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = './data/kidney_disease.csv'
df = pd.read_csv(file_path)

# Drop 'id' column if it exists
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Rename columns for consistency
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']

# Convert necessary columns to numerical type
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

# Handle inconsistencies in categorical data
df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no')
df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})
df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
df['class'] = pd.to_numeric(df['class'], errors='coerce')

# Fill missing values using appropriate methods
def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample
    
def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

# Fill numerical columns using random sampling
num_cols = [col for col in df.columns if df[col].dtype != 'object']
for col in num_cols:
    random_value_imputation(col)

# Fill categorical columns using mode
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
random_value_imputation('red_blood_cells')
random_value_imputation('pus_cell')
for col in cat_cols:
    impute_mode(col)

# Encode categorical variables
categorical_columns = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'hypertension', 
                       'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia']

# Initialize LabelEncoders for each categorical column
encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    df[col] = encoders[col].fit_transform(df[col])

# Save the encoders for later use during prediction
joblib.dump(encoders, './data/encoders.pkl')

# Separate features and target
X = df.drop(columns=['class'])
y = df['class']

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Print evaluation metrics
model_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, model.predict(X_train))}")
print(f"Test Accuracy of Random Forest Classifier is {model_acc}\n")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, model.predict(X_test))}\n")
print(f"Classification Report:\n{classification_report(y_test, model.predict(X_test))}")

# Save the trained model
joblib.dump(model, './data/kidney_disease_rf_model.pkl')

# Define preprocessing for prediction
def preprocess_input(input_data):
    """
    Preprocess user input by encoding categorical values and converting all data to numerical form.
    """
    # Load encoders
    encoders = joblib.load('./data/encoders.pkl')

    # Create DataFrame from input data with all expected columns
    expected_columns = [
        'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
        'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
        'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
        'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'
    ]
    input_df = pd.DataFrame([input_data], columns=expected_columns)

    # Encode categorical columns
    for col in categorical_columns:
        if col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform([input_df[col][0]])

    # Convert all numerical columns to float to match training data type
    input_df = input_df.astype(float)
    
    return input_df.values
