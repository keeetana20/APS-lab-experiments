from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data)

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():

    # Load the breast cancer dataset
    data = load_breast_cancer()

    # Create feature matrix
    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # Original sklearn:
    # 0 = malignant
    # 1 = benign
    #
    # We convert it to:
    # 0 = benign
    # 1 = malignant

    y = pd.Series(
        (data.target == 0).astype(int),
        name="malignant"
    )

    return X, y, data


if __name__ == "__main__":

    X, y, data = load_data()

    print(y.value_counts())
    print("Feature matrix shape", X.shape)
    print("Target shape", y.shape)
    print("Class names", data.target_names)