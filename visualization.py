import matplotlib.pyplot as plt
import pandas as pd


def plot_class_distribution(y):

    # Count each class
    class_counts = y.value_counts().sort_index()

    # Create class distribution table
    class_distribution = pd.DataFrame({
        "Class": [
            "Benign",
            "Malignant"
        ],
        "Count": [
            class_counts.get(0, 0),
            class_counts.get(1, 0)
        ],
        "Probability": [
            class_counts.get(0, 0) / len(y),
            class_counts.get(1, 0) / len(y)
        ]
    })

    print(class_distribution)

    # Plot
    class_distribution.plot(
        x="Class",
        y="Count",
        kind="bar",
        legend=False
    )

    plt.ylabel("Number of observations")
    plt.title("Class Distribution")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()

    return class_distribution


def plot_confusion_matrix(cm, threshold):

    plt.figure()

    plt.imshow(cm)

    plt.title(
        f"Confusion Matrix - Threshold {threshold}"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(
        [0, 1],
        ["Benign", "Malignant"]
    )

    plt.yticks(
        [0, 1],
        ["Benign", "Malignant"]
    )

    # Display numbers inside the matrix
    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.show()