from data import load_data

from preprocessing import (
    split_data,
    show_split_information
)

from model import (
    train_model,
    get_probabilities,
    create_results,
    add_predictions,
    show_threshold_predictions,
    show_confusion_matrices,
    calculate_metrics,
    compare_threshold_metrics
)

from visualization import (
    plot_class_distribution,
    plot_confusion_matrix
)

from sklearn.metrics import confusion_matrix


def main():

    # ==========================================
    # 1. LOAD DATA
    # ==========================================

    print("========== DATASET ==========")

    X, y, data = load_data()

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Class names:", data.target_names)

    print("\nTarget value counts:")
    print(y.value_counts())


    # ==========================================
    # 2. CLASS DISTRIBUTION
    # ==========================================

    print("\n========== CLASS DISTRIBUTION ==========")

    class_distribution = plot_class_distribution(y)


    # ==========================================
    # 3. TRAIN TEST SPLIT
    # ==========================================

    print("\n========== TRAIN TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    show_split_information(
        y_train,
        y_test
    )


    # ==========================================
    # 4. TRAIN MODEL
    # ==========================================

    print("\n========== MODEL TRAINING ==========")

    model = train_model(
        X_train,
        y_train
    )

    print("Logistic Regression model trained successfully.")


    # ==========================================
    # 5. PREDICT PROBABILITIES
    # ==========================================

    print("\n========== PROBABILITIES ==========")

    probabilities = get_probabilities(
        model,
        X_test
    )

    print("\nFirst 5 probability rows:")
    print(probabilities[:5])

    print("\nSum of probabilities:")
    print(
        probabilities[:5].sum(axis=1)
    )


    # ==========================================
    # 6. CREATE RESULTS TABLE
    # ==========================================

    print("\n========== RESULTS ==========")

    results = create_results(
        y_test,
        probabilities
    )

    print(
        results.head(10)
    )


    # ==========================================
    # 7. ADD PREDICTIONS
    # ==========================================

    print("\n========== PREDICTIONS ==========")

    threshold = 0.60

    results = add_predictions(
        results,
        threshold
    )

    print(
        results[
            [
                "Actual_label",
                "P_malignant",
                "P_benign",
                "Predicted_label"
            ]
        ].head(10)
    )


    # ==========================================
    # 8. THRESHOLD COMPARISON
    # ==========================================

    print("\n========== THRESHOLD COMPARISON ==========")

    show_threshold_predictions(
        results
    )


    # ==========================================
    # 9. CONFUSION MATRICES
    # ==========================================

    print("\n========== CONFUSION MATRICES ==========")

    show_confusion_matrices(
        y_test,
        probabilities
    )


    # ==========================================
    # 10. METRICS
    # ==========================================

    print("\n========== MODEL METRICS ==========")

    predictions = calculate_metrics(
        y_test,
        probabilities,
        threshold=0.60
    )


    # ==========================================
    # 11. CONFUSION MATRIX GRAPH
    # ==========================================

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    plot_confusion_matrix(
        cm,
        threshold=0.60
    )


    # ==========================================
    # 12. METRICS FOR DIFFERENT THRESHOLDS
    # ==========================================

    print(
        "\n========== METRICS FOR DIFFERENT THRESHOLDS =========="
    )

    metrics_results = compare_threshold_metrics(
        y_test,
        probabilities
    )

    for result in metrics_results:

        print(
            f"\nThreshold: {result['threshold']}"
        )

        print(
            f"Accuracy: {result['accuracy']:.4f}"
        )

        print(
            f"Precision: {result['precision']:.4f}"
        )

        print(
            f"Recall: {result['recall']:.4f}"
        )

        print(
            f"F1 Score: {result['f1']:.4f}"
        )


# ==========================================
# RUN MAIN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()