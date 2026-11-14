
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Starting model training...")

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded:", len(X), "samples")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Model training completed!")

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy)

# Save model and flower names
joblib.dump(model, "iris_model.pkl")
joblib.dump(iris.target_names, "flower_names.pkl")

print("Model saved successfully!")
print("Training completed!")
