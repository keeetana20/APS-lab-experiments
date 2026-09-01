import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_model(X_train, y_train):

    # Create the machine learning pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    # Train the model
    model.fit(X_train, y_train)

    return model


def get_probabilities(model, X_test):

    # Get probability predictions
    probabilities = model.predict_proba(X_test)

    return probabilities


def create_results(y_test, probabilities):

    # Since:
    # 0 = Benign
    # 1 = Malignant
    #
    # probability column 0 = P(Benign)
    # probability column 1 = P(Malignant)

    results = pd.DataFrame({
        "actual_class": y_test.values,
        "P_malignant": probabilities[:, 1],
        "P_benign": probabilities[:, 0]
    })

    # Convert numbers to labels
    results["Actual_label"] = results[
        "actual_class"
    ].map({
        0: "Benign",
        1: "Malignant"
    })

    return results


def add_predictions(results, threshold=0.60):

    # Predict malignant if probability >= threshold
    results["Predicted_malignant"] = (
        results["P_malignant"] >= threshold
    ).astype(int)

    # Convert prediction to label
    results["Predicted_label"] = results[
        "Predicted_malignant"
    ].map({
        1: "Malignant",
        0: "Benign"
    })

    return results


def show_threshold_predictions(
    results,
    thresholds=[0.60, 0.80, 0.90]
):

    for threshold in thresholds:

        predictions = (
            results["P_malignant"] >= threshold
        ).astype(int)

        print(
            f"Threshold = {threshold}: "
            f"Predicted malignant cases = "
            f"{predictions.sum()}"
        )


def show_confusion_matrices(
    y_test,
    probabilities,
    thresholds=[0.50, 0.60, 0.90]
):

    for threshold in thresholds:

        predicted_malignant = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_test,
            predicted_malignant,
            labels=[0, 1]
        )

        print(f"\nThreshold = {threshold}")
        print(cm)


def calculate_metrics(
    y_test,
    probabilities,
    threshold=0.60
):

    # Create predictions
    predictions = (
        probabilities[:, 1] >= threshold
    ).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nMetrics")
    print("Threshold:", threshold)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return predictions


def compare_threshold_metrics(
    y_test,
    probabilities,
    thresholds=[0.10, 0.30, 0.50, 0.70, 0.90]
):

    metrics_results = []

    for threshold in thresholds:

        y_pred_threshold = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred_threshold,
            labels=[0, 1]
        ).ravel()

        total = tp + tn + fp + fn

        accuracy = (tp + tn) / total

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics_results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return metrics_results