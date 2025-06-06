import joblib
import numpy as np
import pandas as pd # Needed to read CSVs for looking up names

# Load encoders and model
try:
    driver_enc = joblib.load("driver_encoder.pkl")
    constructor_enc = joblib.load("constructor_encoder.pkl")
    race_enc = joblib.load("race_encoder.pkl")
    model = joblib.load("f1_winner_predictor.pkl")
    print("Model and encoders loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure main.py has been run to save them.")
    exit()

# --- Input for Prediction ---
# You need to provide the EXACT string names that were in your training data.
# Let's assume these are the correct names. If you encounter a ValueError
# during encoding, it means the name wasn't in the training data, and you'd
# need to check your original CSVs and how you processed them.

driver_name_to_predict = "Oscar Piastri"
constructor_name_to_predict = "Red Bull" # Or "Red Bull Racing" if that's what's in your constructors.csv
race_name_to_predict = "Spanish Grand Prix"

# For grid, laps, points, milliseconds:
# These are race-specific. For a hypothetical prediction, you'd make educated guesses.
# 'grid': Max Verstappen often starts from pole position. Let's assume P1.
# 'laps': Monaco Grand Prix has 78 laps.
# 'points': Points are awarded *after* the race, so this should be 0 before the race.
# 'milliseconds': This is the race duration, unknown before the race. Use 0.

grid_position = 1 # Max often qualifies P1 or very high
laps_completed = 78 # Typical laps for Monaco Grand Prix
points_at_start = 0 # Points are for the race being predicted, so 0 at start
milliseconds_duration = 0 # Unknown before the race

print(f"\nAttempting to predict for: Driver='{driver_name_to_predict}', Constructor='{constructor_name_to_predict}', Race='{race_name_to_predict}'")
print(f"Assumed race conditions: Grid={grid_position}, Laps={laps_completed}, Points={points_at_start}, Milliseconds={milliseconds_duration}")

# Encode the actual string names
try:
    driver_encoded = driver_enc.transform([driver_name_to_predict])[0]
    constructor_encoded = constructor_enc.transform([constructor_name_to_predict])[0]
    race_encoded = race_enc.transform([race_name_to_predict])[0]
    print(f"Encoded values: Driver={driver_encoded}, Constructor={constructor_encoded}, Race={race_encoded}")

except ValueError as ve:
    print(f"\nEncoding error: {ve}. This usually means the name '{ve.args[0].split()[-1].strip('.')}' was not seen during training.")
    print("Please ensure the driver name, constructor name, and race name exist in the training data (from your CSVs).")
    print("Double-check the exact spelling and capitalization of the names.")
    exit()
except Exception as e:
    print(f"An unexpected encoding error occurred: {e}")
    exit()

# Create the feature vector, matching the order of X in main.py:
# X = data[['grid', 'laps', 'points', 'milliseconds', 'driver_enc', 'constructor_enc', 'race_enc']]
data_point = np.array([[
    grid_position,
    laps_completed,
    points_at_start,
    milliseconds_duration,
    driver_encoded,
    constructor_encoded,
    race_encoded
]])

# Make prediction
prediction = model.predict(data_point)

print("\n--- Prediction Result ---")
if prediction[0] == 1:
    print(f"Predicted Winner: {driver_name_to_predict} 🏆")
else:
    print(f"Predicted No Win for: {driver_name_to_predict} ❌")