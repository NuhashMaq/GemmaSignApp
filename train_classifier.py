import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Folder where CSV files are stored
DATA_DIR = "data"

# Load all CSVs into one DataFrame
all_data = []
all_labels = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")
        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path, header=None)
        all_data.append(df)
        all_labels += [label] * len(df)

X = pd.concat(all_data).values
y = np.array(all_labels)

print(f"Total samples: {len(y)}")
print(f"Labels: {set(y)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN Classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "gesture_model.pkl")
print("✅ Model saved as gesture_model.pkl")
