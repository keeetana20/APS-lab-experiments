# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data)

# %%
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns = data.feature_names
)

y = pd.Series(
    (data.target == 0).astype(int),
    name = "malignant"   
)

print(y.value_counts())
print("Feature matrix shape",X.shape)
print("target shape", y.shape)
print("class names",data.target_names)

# %%
class_counts = y.value_counts().sort_index()
class_distribution = pd.DataFrame({"Class" : data.target_names,
    "Count" : class_counts.values,
    "Probablity" : class_counts.values/len(y)}
)

print(class_distribution)

# %%
class_distribution.plot(
    x="Class",
    y="Count",
    kind="bar",
    legend=False,
    color=["tomato","steelblue"]
)

plt.ylabel("no of observations")
plt.title("class distribution")
plt.xticks(rotation=0)
plt.show()

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state = 42, stratify = y)

print("training size", len(y_train))
print("testing size",len(y_test))

print("\ntraining proportions")
print(y_train.value_counts(normalize=True).sort_index())

print("\n testing proportions")
print(y_test.value_counts(normalize=True).sort_index())

# %%
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train,y_train)

# %%
probablities = model.predict_proba(X_test)

print(probablities[:5])

# %%
print(probablities[:5].sum(axis=1))

# %%
results = pd.DataFrame({"actual_class": y_test.values,
    "P_malignant": probablities[:,0],
    "P_benign": probablities[:,1]}
    
)
print(results.head(10))

# %%
results["Actual_label"] = results["actual_class"].map({
    0: "Malignant",
    1: "Benign"
})

print(
    results[
        ["Actual_label","P_malignant","P_benign"]
    ].head(10)
)

# %%
threshold = 0.60

results["Predicted_malignant"]=(
    results["P_malignant"] >= threshold
).astype(int)

results["Predicted_label"]= results[
    "Predicted_malignant"
].map({
    1 : "Malignant",
    0 : "Benign"
})

print(
    results[
        [
            "Actual_label",
            "P_malignant",
            "Predicted_label"
        ]
    ].head(10)
)


# %%
for threshold in [0.60, 0.80, 0.90]:
    predictions = (
        results["P_malignant"] >= threshold
    ).astype(int)
    
    print(
        f"Threshold = {threshold}:"
        f"Predicted malignant cases = {predictions.sum()}"
    )

# %%
# Construct a confusion matrix

actual_malignant = (y_test.values == 0).astype(int)
for threshold in[0.50, 0.60, 0.90]:
    predicted_malignant = (
        probablities[:,0] >= threshold
    ).astype(int)
    
    cm = confusion_matrix(
        actual_malignant,
        predicted_malignant
    )
    print(f"\n Threshold = {threshold}")
    print(cm)

# %%
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
print("sklearn accuracy:", accuracy_score(y_test, predictions))

print("sklearn precision:", precision_score(y_test,predictions))

print("sklearn recall",recall_score(y_test,predictions))

print("sklearn f1 score", f1_score(y_test,predictions))

# %%
metrics_results=[]

for threshold in [0.10, 0.30, 0.50,0.70,0.90]:
    y_pred_threshold = (
        probablities[:,1] >= threshold
    ).astype(int)
    
    tn, fp ,fn, tp = confusion_matrix(y_test,y_pred_threshold,labels=[0,1]).ravel()
    
    accuracy = (tp+tn)


