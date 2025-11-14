from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("Accuracy:", acc)

# Try a custom input
sample = [[5.0, 3.6, 1.4, 0.2]]
print("Prediction for sample:", iris.target_names[model.predict(sample)[0]])
