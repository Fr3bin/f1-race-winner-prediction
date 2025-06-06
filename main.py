import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Import joblib to save encoders

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load CSVs
# Make sure your 'data' directory is set up correctly relative to your script
races = pd.read_csv("data/races.csv", na_values='\\N')
results = pd.read_csv("data/results.csv", na_values='\\N')
drivers = pd.read_csv("data/drivers.csv", na_values='\\N')
constructors = pd.read_csv("data/constructors.csv", na_values='\\N')
# qualifying = pd.read_csv("data/qualifying.csv", na_values='\\N') # Not used in your current code

# Merge data
data = results.merge(races, on='raceId') \
              .merge(drivers, on='driverId') \
              .merge(constructors, on='constructorId')

# Keep only necessary columns
data = data[[
    'raceId', 'year', 'round', 'name_x',  # race info
    'forename', 'surname',                # driver info
    'name_y',                             # constructor/team
    'grid', 'positionOrder', 'points', 'laps', 'milliseconds'
]]

# Rename for clarity
data.rename(columns={
    'name_x': 'race_name',
    'name_y': 'constructor',
}, inplace=True)

# Create full driver name
data['driver'] = data['forename'] + ' ' + data['surname']

# Predict the WINNER (positionOrder = 1)
data['winner'] = data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)

# Encode categorical features
le_driver = LabelEncoder()
le_constructor = LabelEncoder()
le_race = LabelEncoder()

data['driver_enc'] = le_driver.fit_transform(data['driver'])
data['constructor_enc'] = le_constructor.fit_transform(data['constructor'])
data['race_enc'] = le_race.fit_transform(data['race_name'])

# --- NEW: Save the LabelEncoders ---
joblib.dump(le_driver, "driver_encoder.pkl")
joblib.dump(le_constructor, "constructor_encoder.pkl")
joblib.dump(le_race, "race_encoder.pkl")
# --- END NEW ---

# Select features and label
# Ensure these features match what you'll provide for prediction
X = data[['grid', 'laps', 'points', 'milliseconds', 'driver_enc', 'constructor_enc', 'race_enc']]
y = data['winner']

# Fill missing values (if any)
# It's better to inspect missing values and handle them appropriately,
# but for now, filling with 0 if it's acceptable for your data.
X.fillna(0, inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save model
joblib.dump(model, "f1_winner_predictor.pkl")
print("Model and encoders saved successfully!")