from sklearn.model_selection import train_test_split


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def show_split_information(y_train, y_test):

    print("Training size", len(y_train))
    print("Testing size", len(y_test))

    print("\nTraining proportions")
    print(
        y_train.value_counts(
            normalize=True
        ).sort_index()
    )

    print("\nTesting proportions")
    print(
        y_test.value_counts(
            normalize=True
        ).sort_index()
    )