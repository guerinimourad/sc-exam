import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("https://github.com/guerinimourad/sc-exam/edit/master/cancer%20issue.csv/", sep=";")

# Drop unnecessary columns
df = df.drop(columns=["PatientID", "FamilyHistory"])

# Convert target variable to binary (Yes = 1, No = 0)
df["Recurrence"] = df["Recurrence"].map({"No": 0, "Yes": 1})

# Features and target
X = df.drop(columns=["Recurrence"])
y = df["Recurrence"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
model_path = "https://github.com/guerinimourad/sc-exam/model.pkl"
joblib.dump(model, model_path)
