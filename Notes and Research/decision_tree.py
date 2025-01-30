import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import gdown

# Define the Google Drive ID where the dataset is hosted
drive_id = "1hSm0xrYXRnhyHu-p-e5l5WVVFRxg30kk"

# Function to download the dataset from Google Drive and load it into a DataFrame
def download_and_load_dataset_from_drive(drive_id):
    print("Downloading dataset...")
    url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    output = "dataset.csv"
    gdown.download(url, output, quiet=False)
    print("Dataset downloaded successfully.")
    
    # Load the dataset into a DataFrame
    data = pd.read_csv(output, low_memory=False)
    return data

# Function to load and preprocess data
def load_and_preprocess_data(data):
    # Strip leading and trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Define features and target
    features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std'
    ]
    
    # Convert features to numeric, coerce errors to NaN
    X = data[features].apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values with the mean of the column
    X = X.fillna(X.mean())
    
    y = data['Label'].apply(lambda x: 1 if x != 'BENIGN' else 0)  # Binary target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, features

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
                xticklabels=["BENIGN", "ATTACK"], yticklabels=["BENIGN", "ATTACK"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.show()

# Function to visualize feature importance
def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Decision Tree)")
    plt.bar(range(len(features)), importances[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()

# Function to visualize the Decision Tree
def plot_decision_tree(model, features):
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=features, class_names=["BENIGN", "ATTACK"], max_depth=3)
    plt.title("Decision Tree Visualization")
    plt.show()

# Main function
def main():
    # Download and load the dataset from Google Drive
    data = download_and_load_dataset_from_drive(drive_id)

    # Load and preprocess data
    X, y, scaler, features = load_and_preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

    # Train Decision Tree model
    dt_classifier = DecisionTreeClassifier(random_state=200)
    dt_classifier.fit(X_train, y_train)

    # Evaluate the model
    print("Decision Tree Model Evaluation:")
    evaluate_model(dt_classifier, X_test, y_test)

    # Visualize feature importance
    plot_feature_importance(dt_classifier, features)

    # Visualize the Decision Tree
    plot_decision_tree(dt_classifier, features)

    # Perform cross-validation
    cv_scores = cross_val_score(dt_classifier, X, y, cv=5, scoring="accuracy")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Save the model and scaler
    joblib.dump(dt_classifier, "decision_tree_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved to disk.")

# Run the main function
if __name__ == "__main__":
    main()