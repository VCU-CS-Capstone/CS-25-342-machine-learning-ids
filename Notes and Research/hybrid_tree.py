import os
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split

# Define constants
MODEL_PATH = "hybrid_model.pkl"
DATASET_PATH = "dataset.csv"
SELECTED_FEATURES = [
    'Dur', 'TotBytes', 'SrcBytes', 'DstBytes', 'sMeanPktSz', 'dMeanPktSz', 'Load', 'RunTime', 'Mean', 'Sum', 'Min', 'Max',
    'sTos', 'sTtl', 'dHops', 'Offset', 'SrcLoad', 'pLoss', 'Rate', 'SrcRate', 'TcpRtt', 'TotPkts', 'SrcPkts', 'DstPkts', 'SrcGap', 'DstGap'
]

def preprocess_data(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    data = data.copy()
    data.columns = data.columns.str.strip()
    data = data.loc[:, ~data.columns.duplicated()]

    print("Initial columns:", data.columns.tolist())
    print("Initial shape:", data.shape)

    for feature in SELECTED_FEATURES:
        if feature not in data.columns:
            print(f"Adding missing feature: {feature}")
            data[feature] = np.nan

    try:
        data_subset = data[SELECTED_FEATURES].copy()
    except KeyError as e:
        missing = [f for f in SELECTED_FEATURES if f not in data.columns]
        raise KeyError(f"Missing features in data: {missing}")

    data_subset = data_subset.apply(pd.to_numeric, errors='coerce')

    print("Data subset shape before imputation:", data_subset.shape)
    print("Data subset columns:", data_subset.columns.tolist())

    if data_subset.empty:
        raise ValueError("Data subset is empty after preprocessing")

    imputer = SimpleImputer(strategy='median')
    # Fit the imputer on the current data subset
    imputer.fit(data_subset)
    transformed = imputer.transform(data_subset)
    data_subset_imputed = pd.DataFrame(transformed, columns=SELECTED_FEATURES, index=data_subset.index)

    print("Data subset shape after imputation:", data_subset_imputed.shape)
    print("Data subset columns after imputation:", data_subset_imputed.columns.tolist())

    try:
        data[SELECTED_FEATURES] = data_subset_imputed
    except ValueError as e:
        print("Error assigning imputed values:", e)
        print("Data shape:", data.shape)
        print("Data subset shape:", data_subset_imputed.shape)
        print("Data columns:", data.columns.tolist())
        print("Data subset columns:", data_subset_imputed.columns.tolist())
        print("Data index:", data.index)
        print("Data subset index:", data_subset.index)
        raise

    return data

# Function to prepare features and target
def prepare_features_and_target(data):
    X = data[SELECTED_FEATURES]
    y = data['Label'].apply(lambda x: 1 if str(x).strip().upper() != 'BENIGN' else 0).astype(int)
    return X, y

# Function to train and save the model
def train_and_save_model(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please ensure it is available locally.")

    print("Loading training dataset...")
    train_data = pd.read_csv(dataset_path, low_memory=False)
    train_data = preprocess_data(train_data)  # Preprocess training data (includes fitting imputer)

    X_train = train_data[SELECTED_FEATURES]
    y_train = train_data['Label'].apply(lambda x: 1 if str(x).strip().upper() != 'BENIGN' else 0).astype(int)

    print("Training classifiers...")
    classifiers = {
        'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=15),
        'naive_bayes': GaussianNB()
    }

    probas = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        probas.append(clf.predict_proba(X_train)[:, 1])

    combined_probas = 0.8 * probas[0] + 0.2 * probas[1]

    # The imputer is now fitted within preprocess_data, so we don't need to fit it again here.
    # The preprocess_data function returns the processed DataFrame.

    print("Saving trained model...")
    model_data = {
        'classifiers': classifiers,
        'combined_weights': [0.8, 0.2],
        'selected_features': SELECTED_FEATURES,
        'imputer': SimpleImputer(strategy='median').fit(X_train) # Fit and save a *new* imputer here
    }
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}.")

# Function to load the saved model
def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)

# Function to classify new data using the saved model
def classify_data(new_data, model_path=MODEL_PATH):
    model_data = load_model(model_path)
    classifiers = model_data['classifiers']
    weights = model_data['combined_weights']
    selected_features = model_data['selected_features']
    imputer = model_data['imputer']  # Load the fitted imputer

    new_data = preprocess_data(new_data) # Preprocess the incoming batch

    try:
        X = new_data[selected_features].copy()
        X_imputed = imputer.transform(X) # Use the fitted imputer to transform
        X_processed = pd.DataFrame(X_imputed, columns=selected_features, index=X.index)
    except KeyError as e:
        missing = [f for f in selected_features if f not in new_data.columns]
        raise KeyError(f"Missing features in new data for classification: {missing}")
    except Exception as e:
        print(f"Error during imputation/feature processing: {e}")
        raise

    probas = []
    for name, clf in classifiers.items():
        probas.append(clf.predict_proba(X_processed)[:, 1])

    combined_probas = sum(w * p for w, p in zip(weights, probas))
    predictions = (combined_probas >= 0.5).astype(int)
    return predictions, combined_probas

if __name__ == "__main__":
    train_and_save_model()