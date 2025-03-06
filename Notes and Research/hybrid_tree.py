import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import gdown
import time

# Define the Google Drive ID where the dataset is hosted
drive_id = "1ARyLuQlDJxJ7Q42unIcCgSbOxKe0_6jF"
dataset_path = "dataset.csv"

# Define selected features
selected_features = [
    'Dur', 'TotBytes', 'SrcBytes', 'DstBytes', 'sMeanPktSz', 'dMeanPktSz', 'Load', 'RunTime', 'Mean', 'Sum', 'Min', 'Max',
    'sTos', 'sTtl', 'dHops', 'Offset', 'SrcLoad', 'pLoss', 'Rate', 'SrcRate', 'TcpRtt', 'TotPkts', 'SrcPkts', 'DstPkts', 'SrcGap', 'DstGap'
]

# Function to download the dataset from Google Drive and load it into a DataFrame
def download_and_load_dataset_from_drive(drive_id, dataset_path):
    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        url = f"https://drive.google.com/uc?export=download&id={drive_id}"
        gdown.download(url, dataset_path, quiet=False)
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists. Skipping download.")
    
    # Load the dataset into a DataFrame
    data = pd.read_csv(dataset_path, low_memory=False)
    return data

# Function to preprocess data
def preprocess_data(data):
    start_time = time.time()
    
    # Strip leading and trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Print column names for debugging
    print("Dataset columns:", data.columns.tolist())

    # Convert features to numeric, coerce errors to NaN
    print("Converting features to numeric...")
    data[selected_features] = data[selected_features].apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with the median of each feature using SimpleImputer
    # Median is less sensitive to outliers and preserves distribution of data
    print("Filling NaN values with median using SimpleImputer...")
    imputer = SimpleImputer(strategy='median')
    data[selected_features] = imputer.fit_transform(data[selected_features])

    # Check the shape of the data after preprocessing
    print(f"Data shape after preprocessing: {data.shape}")

    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")
    return data

# Function to prepare features and target
def prepare_features_and_target(data):
    start_time = time.time()
    
    # Verify all features are present and numeric
    for feature in selected_features:
        if feature not in data.columns:
            raise ValueError(f"Feature {feature} not found in dataset")
        if not np.issubdtype(data[feature].dtype, np.number):
            raise ValueError(f"Feature {feature} is not numeric. dtype: {data[feature].dtype}")

    X = data[selected_features]

    # Convert labels to binary and ensure they are integers
    y = data['Label'].apply(lambda x: 1 if str(x).strip().upper() != 'BENIGN' else 0).astype(int)
    
    print(f"Feature and target preparation completed in {time.time() - start_time:.2f} seconds.")
    return X, y

# Function to train individual classifiers and combine their probabilities
def train_and_combine_classifiers(X, y):
    start_time = time.time()
    
    classifiers = {
        'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=15),
        'naive_bayes': GaussianNB()
    }
    
    # Train classifiers and store their predicted probabilities
    probas = []
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X, y)
        probas.append(clf.predict_proba(X)[:, 1])
    
    # Combine probabilities by averaging
    combined_probas = np.mean(probas, axis=0)
    
    print(f"Training and combining classifiers completed in {time.time() - start_time:.2f} seconds.")
    return combined_probas

# Function to visualize model performance
def visualize_performance(y_true, y_pred_proba):
    start_time = time.time()
    
    # Make final predictions based on combined probabilities
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_pred_proba)[:2])
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()
    
    print(f"Visualization completed in {time.time() - start_time:.2f} seconds.")

# Main function to execute the workflow
def main():
    # Download and load dataset
    data = download_and_load_dataset_from_drive(drive_id, dataset_path)
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Prepare features and target
    X, y = prepare_features_and_target(data)
    
    # Train classifiers and combine their probabilities
    combined_probas = train_and_combine_classifiers(X, y)
    
    # Visualize performance
    visualize_performance(y, combined_probas)

if __name__ == "__main__":
    main()