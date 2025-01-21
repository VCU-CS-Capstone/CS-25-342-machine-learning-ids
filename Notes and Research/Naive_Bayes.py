import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings('ignore', category=DataConversionWarning)
pd.set_option('display.max_columns', None)


def load_and_preprocess_data(file_path):
    """Load and preprocess the network traffic data."""
    if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
        raise FileNotFoundError(f"File {file_path} does not exist or is not readable.")

    # Load the dataset with low_memory=False to prevent DtypeWarning
    data = pd.read_csv(file_path, low_memory=False)
    data.columns = data.columns.str.strip()

    # Display initial info about the dataset
    print("\nInitial data info:")
    print(data.info())

    # Convert features to numeric, handling errors
    features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std'
    ]

    for feature in features:
        try:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
        except Exception as e:
            print(f"Error converting {feature}: {str(e)}")
            print(f"Unique values in {feature}: ", data[feature].unique())

    # Handle missing values after conversion
    data = data.replace([np.inf, -np.inf], np.nan)

    # Print summary of missing values
    print("\nMissing values after conversion:")
    print(data[features].isnull().sum())

    # Fill missing values with median instead of mean (more robust to outliers)
    for feature in features:
        data[feature] = data[feature].fillna(data[feature].median())

    return data


def prepare_features_and_target(data):
    """Prepare feature matrix and target vector."""
    features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std'
    ]

    # Verify all features are present and numeric
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Feature {feature} not found in dataset")
        if not np.issubdtype(data[feature].dtype, np.number):
            raise ValueError(f"Feature {feature} is not numeric. dtype: {data[feature].dtype}")

    X = data[features]

    # Print feature statistics
    print("\nFeature statistics:")
    print(X.describe())

    # Convert labels to binary
    y = data['Label'].apply(lambda x: 1 if str(x).strip().upper() != 'BENIGN' else 0)

    # Print class distribution
    print("\nClass distribution:")
    print(y.value_counts())

    return X, y


def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def evaluate_model(model, X_test, y_test, print_results=True):
    """
    Evaluate the model and return detailed metrics.
    Added comprehensive metrics reporting with precision and recall for each class.
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=None),  # For each class
        'recall': recall_score(y_test, y_pred, average=None),  # For each class
        'precision_avg': precision_score(y_test, y_pred, average='weighted'),  # Weighted average
        'recall_avg': recall_score(y_test, y_pred, average='weighted'),  # Weighted average
        'f1_avg': f1_score(y_test, y_pred, average='weighted'),  # Weighted average
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, digits=4)
    }

    if print_results:
        print("\n" + "=" * 50)
        print("MODEL EVALUATION METRICS")
        print("=" * 50)

        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")

        print("\nPer-Class Metrics:")
        print("-" * 30)
        print("Class 0 (BENIGN):")
        print(f"Precision: {metrics['precision'][0]:.4f}")
        print(f"Recall: {metrics['recall'][0]:.4f}")

        print("\nClass 1 (Attack):")
        print(f"Precision: {metrics['precision'][1]:.4f}")
        print(f"Recall: {metrics['recall'][1]:.4f}")

        print("\nWeighted Averages:")
        print("-" * 30)
        print(f"Precision: {metrics['precision_avg']:.4f}")
        print(f"Recall: {metrics['recall_avg']:.4f}")
        print(f"F1-Score: {metrics['f1_avg']:.4f}")

        print("\nConfusion Matrix:")
        print("-" * 30)
        print(metrics['confusion_matrix'])

        print("\nDetailed Classification Report:")
        print("-" * 30)
        print(metrics['classification_report'])

        # Calculate and print additional insights
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        print("\nAdditional Insights:")
        print("-" * 30)
        print(f"True Negatives (Correctly identified benign): {tn}")
        print(f"False Positives (Incorrectly flagged as attack): {fp}")
        print(f"False Negatives (Missed attacks): {fn}")
        print(f"True Positives (Correctly identified attacks): {tp}")

        print(f"\nFalse Positive Rate: {fp / (fp + tn):.4f}")
        print(f"False Negative Rate: {fn / (fn + tp):.4f}")

    return metrics


def main():
    # Define the file path
    file_path = "C:/Users/domin/Downloads/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data = load_and_preprocess_data(file_path)

        print("\nPreparing features and target...")
        X, y = prepare_features_and_target(data)

        # Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=200, stratify=y
        )

        # Train and evaluate base model
        print("\nTraining and evaluating base Naive Bayes model...")
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)
        base_metrics = evaluate_model(nb_classifier, X_test, y_test)

        # Perform cross-validation with detailed scoring
        print("\nPerforming cross-validation...")
        cv_scores = {
            'accuracy': cross_val_score(nb_classifier, X_scaled, y, cv=5, scoring='accuracy'),
            'precision': cross_val_score(nb_classifier, X_scaled, y, cv=5, scoring='precision_weighted'),
            'recall': cross_val_score(nb_classifier, X_scaled, y, cv=5, scoring='recall_weighted')
        }

        print("\nCross-validation results:")
        print("-" * 30)
        for metric, scores in cv_scores.items():
            print(f"{metric.capitalize()}:")
            print(f"Scores: {scores}")
            print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        # Grid search with multiple scoring metrics
        print("\nPerforming grid search with multiple metrics...")
        param_grid = {
            'var_smoothing': np.logspace(-10, -8, 10)
        }

        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        }

        grid_search = GridSearchCV(
            GaussianNB(),
            param_grid,
            cv=5,
            scoring=scoring,
            refit='accuracy',  # Use accuracy to select the best model
            n_jobs=-1,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        print("\nGrid Search Results:")
        print("-" * 30)
        print(f"Best parameters: {grid_search.best_params_}")
        print("\nBest scores for each metric:")
        for metric in scoring.keys():
            print(
                f"{metric.capitalize()}: {grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]:.4f}")

        # Evaluate best model
        print("\nEvaluating best model...")
        best_model = grid_search.best_estimator_
        best_metrics = evaluate_model(best_model, X_test, y_test)

        # Save detailed results
        results_dict = {
            'base_model': base_metrics,
            'best_model': best_metrics,
            'cross_validation': cv_scores,
            'grid_search': grid_search.cv_results_
        }

        # Save results to CSV
        pd.DataFrame(grid_search.cv_results_).to_csv('grid_search_results.csv', index=False)

        # Create and save a summary report
        with open('model_evaluation_report.txt', 'w') as f:
            f.write("Network Intrusion Detection System - Model Evaluation Report\n")
            f.write("=" * 70 + "\n\n")
            f.write("Base Model Metrics:\n")
            f.write(f"Accuracy: {base_metrics['accuracy']:.4f}\n")
            f.write(f"Weighted Precision: {base_metrics['precision_avg']:.4f}\n")
            f.write(f"Weighted Recall: {base_metrics['recall_avg']:.4f}\n\n")
            f.write("Best Model Metrics:\n")
            f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
            f.write(f"Weighted Precision: {best_metrics['precision_avg']:.4f}\n")
            f.write(f"Weighted Recall: {best_metrics['recall_avg']:.4f}\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()