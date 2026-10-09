
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Split data into training and testing sets
Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Train Random Forest model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
).fit(Xtr, ytr)

# Train AdaBoost model
ab = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
).fit(Xtr, ytr)

# Calculate and print accuracy
print("RF accuracy:", round(accuracy_score(yte, rf.predict(Xte)), 4))
print("AdaBoost accuracy:", round(accuracy_score(yte, ab.predict(Xte)), 4))
