import pandas as pd
import numpy as np
import joblib
# Sklearn packages for machine learning in python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Load the dataset
file_path = './data/kidney_disease.csv'
ckd_dataset = pd.read_csv(file_path)

# Drop the id column
ckd_dataset.drop(columns='id', inplace=True)

# Rename the column names to more descriptive names
ckd_dataset.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'anaemia', 'class']

# Convert necessary columns to numerical type
ckd_dataset['packed_cell_volume'] = pd.to_numeric(ckd_dataset['packed_cell_volume'], errors='coerce')
ckd_dataset['white_blood_cell_count'] = pd.to_numeric(ckd_dataset['white_blood_cell_count'], errors='coerce')
ckd_dataset['red_blood_cell_count'] = pd.to_numeric(ckd_dataset['red_blood_cell_count'], errors='coerce')

# Remove leading/trailing whitespace in the categorical columns
ckd_dataset['diabetes_mellitus'] = ckd_dataset['diabetes_mellitus'].str.strip()
ckd_dataset['coronary_artery_disease'] = ckd_dataset['coronary_artery_disease'].str.strip()
ckd_dataset['class'] = ckd_dataset['class'].str.strip()

ckd_dataset['class'] = ckd_dataset['class'].map({'ckd': 1, 'notckd': 0})
ckd_dataset['class'] = pd.to_numeric(ckd_dataset['class'])

def handling_missing_values(df):
    """
    Comprehensive missing data handling pipeline for the CKD dataset.
    
    Parameters:
    df (pandas.DataFrame): Raw CKD dataset
    
    Returns:
    pandas.DataFrame: clean dataset
   
    """
    # Create a copy to avoid modifying the original data
    data = df.copy()
    
    # Step 1: Separate numerical and categorical columns
    numerical_cols = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
                     'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                     'potassium', 'haemoglobin', 'packed_cell_volume',
                     'white_blood_cell_count', 'red_blood_cell_count']
    
    categorical_cols = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                       'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
                       'appetite', 'peda_edema', 'anaemia']
    
    # Step 2: Handle categorical variables first
    for col in categorical_cols:
        # Fill with mode for categorical columns
        mode_value = data[col].mode()[0]
        data[col] = data[col].fillna(mode_value)
    
    # Step 3: Prepare numerical data for KNN imputation
    numerical_data = data[numerical_cols].copy()
    
    # Step 4: Scale the data before KNN imputation
    scaler = StandardScaler()
    numerical_data_scaled = pd.DataFrame(
        scaler.fit_transform(numerical_data),
        columns=numerical_data.columns)
    
    # Step 5: Perform KNN imputation on scaled numerical data
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    numerical_data_imputed = pd.DataFrame(
        imputer.fit_transform(numerical_data_scaled),
        columns=numerical_data.columns)
    
    # Step 6: Inverse transform the scaled data
    numerical_data_final = pd.DataFrame(
        scaler.inverse_transform(numerical_data_imputed),
        columns=numerical_data.columns)
    
    # Step 7: Replace the original numerical columns with imputed values
    for col in numerical_cols:
        data[col] = numerical_data_final[col]
        
    return data

# Apply the cleaning to the dataset
clean_ckd_data = handling_missing_values(ckd_dataset)

# Function to replace 'notpresent' with 'absent
def replace_value(values):
    return np.where(values == 'notpresent', 'absent', values)

# Get the categorical columns
categorical_cols = [col for col in clean_ckd_data.columns if clean_ckd_data[col].dtype == 'object']
# Apply the function to all columns
clean_ckd_data[categorical_cols] = clean_ckd_data[categorical_cols].apply(replace_value)

# Encode the categorical columns
# Initialize LabelEncoders for each categorical column
encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    clean_ckd_data[col] = encoders[col].fit_transform(clean_ckd_data[col])

import joblib   
# Save the encoders for later use during prediction
joblib.dump(encoders, './data/encoders.pkl')

# Separate features and target
X = clean_ckd_data.drop(columns=['class'])
y = clean_ckd_data['class']

# Cross Validation with Random Forest
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

# StratifiedKFold for maintaining class distribution across folds which is especially important for imbalanced datasets 
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')

# Top 15 features from feature selection techniques
top_features = ['hypertension','red_blood_cell_count','specific_gravity','appetite','blood_glucose_random','blood_urea',
               'diabetes_mellitus','haemoglobin','albumin','packed_cell_volume','sodium',
               'blood_pressure','peda_edema','serum_creatinine','sugar']

# Separate features and target
X = clean_ckd_data[top_features]
y = clean_ckd_data['class']

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Random Forest with best parameters
best_params = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
RF_tuned = RandomForestClassifier(**best_params, random_state=42)
RF_tuned.fit(X_train, y_train)
    
# Predictions and evaluation
y_pred = RF_tuned.predict(X_test)

# Save the trained model
joblib.dump(RF_tuned, './data/kidney_disease_rf_model.pkl')

# Define preprocessing for prediction
def preprocess_input(input_data):
    """
    Preprocess user input by encoding categorical values and converting all data to numerical form.
    """
    # Load encoders
    encoders = joblib.load('./data/encoders.pkl')

    # Create DataFrame from input data with all expected columns
    expected_columns = top_features
    
    input_df = pd.DataFrame([input_data], columns=expected_columns)

    # Encode categorical columns
    for col in categorical_cols:
        if col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform([input_df[col][0]])

    # Convert all numerical columns to float to match training data type
    input_df = input_df.astype(float)
    
    return input_df.values
