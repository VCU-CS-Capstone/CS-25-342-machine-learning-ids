import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Define the file path
file_path = "C:/Users/Quack/Downloads/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Check if the file exists and is readable
if os.path.exists(file_path) and os.access(file_path, os.R_OK):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Strip leading and trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Define features and target
    features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std'
    ]
    X = data[features]
    y = data['Label'].apply(lambda x: 1 if x != 'BENIGN' else 0)  # Binary target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

    # Train Decision Tree model
    dt_classifier = DecisionTreeClassifier(random_state=200)
    dt_classifier.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.10f}")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Precision: {precision:.10f}")
    print(f"Recall: {recall:.10f}")
    print(f"F1-Score: {f1:.10f}")

    # Create a DataFrame for the classification report
    report = classification_report(y_test, y_pred, digits=10, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:\n", report_df)

    # Optionally, save the classification report to a CSV file
    # report_df.to_csv("classification_report.csv", index=True)

    # Optional: Hyperparameter tuning
    param_grid = {
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=200), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)
else:
    print(f"File {file_path} does not exist or is not readable.")