"""ML Model Pipeline for CKD Prediction"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer


# Top 15 features selected through feature engineering
TOP_FEATURES = [
    'hypertension', 'red_blood_cell_count', 'specific_gravity', 'appetite',
    'blood_glucose_random', 'blood_urea', 'diabetes_mellitus', 'haemoglobin',
    'albumin', 'packed_cell_volume', 'sodium', 'blood_pressure', 'peda_edema',
    'serum_creatinine', 'sugar'
]


class MLModelPipeline:
    """
    ML Pipeline for Chronic Kidney Disease prediction.
    Encapsulates model training, preprocessing, and prediction logic.
    """

    def __init__(self, data_path='./data/kidney_disease.csv', verbose=True):
        """
        Initialize ML pipeline.

        Args:
            data_path: Path to the kidney disease dataset
            verbose: Print progress information
        """
        self.data_path = data_path
        self.verbose = verbose
        self.model = None
        self.encoders = None
        self.top_features = TOP_FEATURES

    def load_and_prepare_dataset(self):
        """Load and perform initial preprocessing on the dataset."""
        ckd_dataset = pd.read_csv(self.data_path)

        # Drop the id column
        ckd_dataset.drop(columns='id', inplace=True)

        # Rename columns to descriptive names
        ckd_dataset.columns = [
            'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
            'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
            'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
            'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
            'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
            'coronary_artery_disease', 'appetite', 'peda_edema', 'anaemia', 'class'
        ]

        # Convert necessary columns to numerical type
        ckd_dataset['packed_cell_volume'] = pd.to_numeric(ckd_dataset['packed_cell_volume'], errors='coerce')
        ckd_dataset['white_blood_cell_count'] = pd.to_numeric(ckd_dataset['white_blood_cell_count'], errors='coerce')
        ckd_dataset['red_blood_cell_count'] = pd.to_numeric(ckd_dataset['red_blood_cell_count'], errors='coerce')

        # Remove leading/trailing whitespace in categorical columns
        ckd_dataset['diabetes_mellitus'] = ckd_dataset['diabetes_mellitus'].str.strip()
        ckd_dataset['coronary_artery_disease'] = ckd_dataset['coronary_artery_disease'].str.strip()
        ckd_dataset['class'] = ckd_dataset['class'].str.strip()

        # Map class labels
        ckd_dataset['class'] = ckd_dataset['class'].map({'ckd': 1, 'notckd': 0})
        ckd_dataset['class'] = pd.to_numeric(ckd_dataset['class'])

        return ckd_dataset

    def handling_missing_values(self, df):
        """
        Comprehensive missing data handling pipeline for the CKD dataset.

        Parameters:
            df (pandas.DataFrame): Raw CKD dataset

        Returns:
            pandas.DataFrame: clean dataset
        """
        data = df.copy()

        # Separate numerical and categorical columns
        numerical_cols = [
            'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
            'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
            'potassium', 'haemoglobin', 'packed_cell_volume',
            'white_blood_cell_count', 'red_blood_cell_count'
        ]

        categorical_cols = [
            'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
            'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
            'appetite', 'peda_edema', 'anaemia'
        ]

        # Handle categorical variables with mode
        for col in categorical_cols:
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)

        # Prepare numerical data for KNN imputation
        numerical_data = data[numerical_cols].copy()

        # Scale the data before KNN imputation
        scaler = StandardScaler()
        numerical_data_scaled = pd.DataFrame(
            scaler.fit_transform(numerical_data),
            columns=numerical_data.columns
        )

        # Perform KNN imputation on scaled numerical data
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        numerical_data_imputed = pd.DataFrame(
            imputer.fit_transform(numerical_data_scaled),
            columns=numerical_data.columns
        )

        # Inverse transform the scaled data
        numerical_data_final = pd.DataFrame(
            scaler.inverse_transform(numerical_data_imputed),
            columns=numerical_data.columns
        )

        # Replace the original numerical columns with imputed values
        for col in numerical_cols:
            data[col] = numerical_data_final[col]

        return data

    def train(self):
        """Train Random Forest model with preprocessing and save artifacts."""
        if self.verbose:
            print("Loading and preparing dataset...")

        # Load dataset
        ckd_dataset = self.load_and_prepare_dataset()

        # Apply missing value handling
        clean_ckd_data = self.handling_missing_values(ckd_dataset)

        # Replace 'notpresent' with 'absent'
        categorical_cols = [col for col in clean_ckd_data.columns if clean_ckd_data[col].dtype == 'object']

        def replace_value(values):
            return np.where(values == 'notpresent', 'absent', values)

        clean_ckd_data[categorical_cols] = clean_ckd_data[categorical_cols].apply(replace_value)

        # Encode categorical columns
        encoders = {col: LabelEncoder() for col in categorical_cols}
        for col in categorical_cols:
            clean_ckd_data[col] = encoders[col].fit_transform(clean_ckd_data[col])

        # Save encoders
        joblib.dump(encoders, './data/encoders.pkl')
        self.encoders = encoders

        # Separate features and target
        X = clean_ckd_data[self.top_features]
        y = clean_ckd_data['class']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train Random Forest with best parameters
        best_params = {
            'max_depth': None,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 50
        }

        if self.verbose:
            print("Training Random Forest model...")

        RF_tuned = RandomForestClassifier(**best_params, random_state=42)
        RF_tuned.fit(X_train, y_train)

        # Save model
        joblib.dump(RF_tuned, './data/kidney_disease_rf_model.pkl')
        self.model = RF_tuned

        # Predictions and evaluation
        y_pred = RF_tuned.predict(X_test)
        accuracy = (y_pred == y_test).mean()

        if self.verbose:
            print(f"Model trained successfully. Test accuracy: {accuracy:.3f}")

        return accuracy

    def load_model(self):
        """Load pre-trained model and encoders from disk."""
        if self.verbose:
            print("Loading ML model and encoders...")

        self.model = joblib.load('./data/kidney_disease_rf_model.pkl')
        self.encoders = joblib.load('./data/encoders.pkl')

        if self.verbose:
            print("ML model loaded successfully.")

    def predict(self, input_data):
        """
        Make prediction on input data.

        Args:
            input_data: Dictionary with health parameters

        Returns:
            int: Prediction (0 = no CKD, 1 = CKD)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        features = preprocess_input(input_data)
        prediction = self.model.predict(features)[0]
        return prediction


def preprocess_input(input_data):
    """
    Preprocess user input by encoding categorical values and converting to numerical form.

    Args:
        input_data: Dictionary with keys matching TOP_FEATURES

    Returns:
        numpy array ready for prediction
    """
    # Load encoders
    encoders = joblib.load('./data/encoders.pkl')

    # Create DataFrame from input data with expected columns
    input_df = pd.DataFrame([input_data], columns=TOP_FEATURES)

    # Get categorical columns from encoders
    categorical_cols = list(encoders.keys())

    # Encode categorical columns
    for col in categorical_cols:
        if col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform([input_df[col][0]])

    # Convert all to float
    input_df = input_df.astype(float)

    return input_df.values
