
# Practical: Ensemble Learning Techniques
# Stacking, AdaBoost and Voting Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import (
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Classes:", data.target_names)

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 3. Define Base Classifiers
base_classifiers = [
    ('lr', make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )),

    ('dt', DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )),

    ('knn', make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )),

    ('nb', make_pipeline(
        StandardScaler(),
        GaussianNB()
    ))
]

# 4. Define Individual Models
models = {
    'Logistic Regression': base_classifiers[0][1],
    'Decision Tree': base_classifiers[1][1],
    'KNN': base_classifiers[2][1],
    'Naive Bayes': base_classifiers[3][1]
}

results = []

# 5. Evaluation Function
def evaluate_model(name, model):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted'
    )
    recall = recall_score(
        y_test, y_pred, average='weighted'
    )
    f1 = f1_score(
        y_test, y_pred, average='weighted'
    )

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    print("\n", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# 6. Evaluate Individual Models
print("\n--- BASE CLASSIFIERS ---")

for name, model in models.items():
    evaluate_model(name, model)

# 7. Stacking Ensemble
stacking_model = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5
)

print("\n--- STACKING ENSEMBLE ---")
evaluate_model("Stacking", stacking_model)

# 8. AdaBoost Ensemble
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

print("\n--- ADABOOST ENSEMBLE ---")
evaluate_model("AdaBoost", ada_model)

# 9. Voting Ensemble
voting_model = VotingClassifier(
    estimators=base_classifiers,
    voting='hard'
)

print("\n--- VOTING ENSEMBLE ---")
evaluate_model("Voting", voting_model)

# 10. Performance Comparison
results_df = pd.DataFrame(results)

print("\n--- PERFORMANCE COMPARISON ---")
print(results_df.round(4).to_string(index=False))

# 11. Identify Best Model
best_model = results_df.loc[
    results_df['Accuracy'].idxmax()
]

print("\nBest Performing Model:")
print(best_model)

# 12. Performance Comparison Graph
metrics = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1 Score'
]

results_df.set_index('Model')[metrics].plot(
    kind='bar',
    figsize=(12, 6),
    ylim=(0, 1.05)
)

plt.title("Ensemble Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Models")
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("performance_comparison.png")
plt.show()

# 13. Confusion Matrices
ensemble_models = {
    'Stacking': stacking_model,
    'AdaBoost': ada_model,
    'Voting': voting_model
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, model) in zip(
    axes, ensemble_models.items()
):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=data.target_names,
        yticklabels=data.target_names
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

print("\nExperiment Completed Successfully!")
