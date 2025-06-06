import pandas as pd
import numpy as np
import joblib

# --- Step 1: Setup and Loading ---
print("Step 1: Loading model, encoders, and data...")

try:
    # Load your trained model and encoders
    model = joblib.load("f1_winner_predictor.pkl")
    driver_enc = joblib.load("driver_encoder.pkl")
    constructor_enc = joblib.load("constructor_encoder.pkl")
    race_enc = joblib.load("race_encoder.pkl")
    print("Model and encoders loaded successfully.")

    # Load raw data CSVs to get lists of names for iteration
    # These are only used for validating the names you manually provide.
    drivers_df = pd.read_csv("data/drivers.csv", na_values='\\N')
    constructors_df = pd.read_csv("data/constructors.csv", na_values='\\N')
    races_df = pd.read_csv("data/races.csv", na_values='\\N')
    print("Raw data CSVs loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading required files: {e}.")
    print("Please ensure 'f1_winner_predictor.pkl', 'driver_encoder.pkl', 'constructor_encoder.pkl', 'race_encoder.pkl' are in the same directory as this script.")
    print("Also ensure 'data/drivers.csv', 'data/constructors.csv', 'data/races.csv' are in the 'data' subfolder.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    exit()


# --- Step 2: Define the Target Race and Candidate Pool (MODIFIED) ---
print("\nStep 2: Defining target race and preparing candidate list...")

# Define the specific race you want to predict for
# IMPORTANT: Use the EXACT name as it appears in your 'races.csv'
TARGET_RACE_NAME = "Australian Grand Prix" # <--- MODIFY THIS FOR DIFFERENT RACES

# Define a LIMITED list of candidate drivers and constructors for prediction.
# This prevents the script from trying to predict for every historical driver/team,
# which can be millions of combinations.
# IMPORTANT: Use the EXACT names as they appear in your 'drivers.csv' and 'constructors.csv'
# If you get an encoding error, double-check spelling and capitalization.
CANDIDATE_DRIVERS = [
    "Max Verstappen",
    "Sergio Pérez",
    "Lewis Hamilton",
    "George Russell",
    "Charles Leclerc",
    "Carlos Sainz",
    "Lando Norris",
    "Oscar Piastri",
    "Fernando Alonso",
    "Lance Stroll",
    "Pierre Gasly",
    "Esteban Ocon",
    "Alexander Albon",
    "Logan Sargeant",
    "Valtteri Bottas",
    "Guanyu Zhou",
    "Kevin Magnussen",
    "Nico Hülkenberg",
    "Daniel Ricciardo",
    "Yuki Tsunoda"
]

CANDIDATE_CONSTRUCTORS = [
    "Red Bull",
    "Mercedes",
    "Ferrari",
    "McLaren",
    "Aston Martin",
    "Alpine",
    "Williams",
    "Sauber", # For Kick Sauber
    "Haas F1 Team",
    "AlphaTauri", # For RB F1 Team / AlphaTauri
    "RB F1 Team" # Include both if present in your data, to cover all cases
]

# --- Input Validation for Candidates (New) ---
# It's a good idea to check if the candidate names actually exist in your encoders' classes
for name_list, encoder, entity_type in [(CANDIDATE_DRIVERS, driver_enc, "driver"),
                                         (CANDIDATE_CONSTRUCTORS, constructor_enc, "constructor")]:
    for name in name_list:
        if name not in encoder.classes_:
            print(f"Warning: The {entity_type} '{name}' is not in the training data's known {entity_type}s. It will be skipped.")

# Check if the target race name exists in the encoder's known labels
if TARGET_RACE_NAME not in race_enc.classes_:
    print(f"Error: The race '{TARGET_RACE_NAME}' was not found in the training data.")
    print(f"Available races in training data: {race_enc.classes_}")
    exit()

encoded_race = race_enc.transform([TARGET_RACE_NAME])[0]

print(f"Preparing to predict for the '{TARGET_RACE_NAME}'.")
print(f"Predicting for {len(CANDIDATE_DRIVERS)} specific drivers and {len(CANDIDATE_CONSTRUCTORS)} specific constructors.")

# Initialize a list to store predictions
predictions_list = []

# --- Step 3: Prediction Loop (MODIFIED) ---
print("\nStep 3: Generating predictions for selected drivers...")

# Set hypothetical common values for the race conditions
# These are crucial and will influence the prediction.
# You might want to experiment with these or refine them.
# Consult a source like Wikipedia for actual lap counts for GPs
HYPOTHETICAL_LAPS = 66 # Example for Spanish GP (typically 66 laps) - adjust for other races
HYPOTHETICAL_POINTS_AT_START = 0 # Points before the race starts are 0 for the race being predicted
HYPOTHETICAL_MILLISECONDS_DURATION = 0 # Race duration is unknown before the race

# Loop through each chosen driver and each chosen constructor
for driver_name in CANDIDATE_DRIVERS:
    for constructor_name in CANDIDATE_CONSTRUCTORS:
        try:
            # Encode current driver and constructor
            # Only encode if the name was seen during training
            if driver_name in driver_enc.classes_ and constructor_name in constructor_enc.classes_:
                encoded_driver = driver_enc.transform([driver_name])[0]
                encoded_constructor = constructor_enc.transform([constructor_name])[0]

                # We'll make predictions for different hypothetical grid positions
                # Let's consider the top 3 starting positions as strong indicators
                for grid_position in [1, 2, 3]: # Assuming strong drivers start high on the grid
                    # Create the feature vector, matching the order from main.py's X:
                    # X = data[['grid', 'laps', 'points', 'milliseconds', 'driver_enc', 'constructor_enc', 'race_enc']]
                    data_point = np.array([[
                        grid_position,
                        HYPOTHETICAL_LAPS,
                        HYPOTHETICAL_POINTS_AT_START,
                        HYPOTHETICAL_MILLISECONDS_DURATION,
                        encoded_driver,
                        encoded_constructor,
                        encoded_race
                    ]])

                    # Get probability of winning (class 1)
                    # predict_proba returns [[prob_class_0, prob_class_1]]
                    win_probability = model.predict_proba(data_point)[0][1]

                    predictions_list.append({
                        'driver': driver_name,
                        'constructor': constructor_name,
                        'race': TARGET_RACE_NAME,
                        'grid_position': grid_position,
                        'win_probability': win_probability
                    })
            # else:
            #     print(f"Skipping prediction for '{driver_name}' with '{constructor_name}' as one is not in training data.")

        except Exception as e:
            # Catch other unexpected errors during the loop
            print(f"An unexpected error occurred during prediction for {driver_name} with {constructor_name}: {e}")

print(f"Generated {len(predictions_list)} individual predictions.")

# --- Step 4: Determining and Displaying the Winner ---
print("\nStep 4: Analyzing results and predicting the winner...")

if not predictions_list:
    print("No predictions were generated. This might be because the candidate drivers/constructors were not in the training data.")
    print("Please verify the names in CANDIDATE_DRIVERS and CANDIDATE_CONSTRUCTORS against your drivers.csv and constructors.csv.")
else:
    # Convert predictions to a DataFrame for easier sorting and analysis
    predictions_df = pd.DataFrame(predictions_list)

    # Sort by win probability in descending order
    predictions_df = predictions_df.sort_values(by='win_probability', ascending=False)

    # Filter out predictions with very low probability (e.g., less than 0.01)
    # This helps focus on plausible candidates
    relevant_predictions = predictions_df[predictions_df['win_probability'] > 0.01]

    if relevant_predictions.empty:
        print(f"No driver is predicted to win the '{TARGET_RACE_NAME}' with a probability greater than 1%.")
        print("This is common due to the rarity of wins and potential class imbalance in your training data.")
        print("Consider inspecting the top predictions regardless:")
        relevant_predictions = predictions_df.head(5) # Show top 5 even if low probability

    print(f"\n--- Top Predictions for {TARGET_RACE_NAME} ---")
    # Display the top 5 predictions
    print(relevant_predictions.head(5).to_string(index=False))

    # Identify the top predicted winner based on the highest probability
    # If multiple drivers have the same highest probability, it will pick the first one
    top_winner = predictions_df.iloc[0]

    print(f"\n--- Model's Most Likely Winner for {TARGET_RACE_NAME} ---")
    if top_winner['win_probability'] > 0.5: # A threshold for a "confident" win prediction
        print(f"Predicted Winner: {top_winner['driver']} ({top_winner['constructor']})")
        print(f"  Starting Grid Position: {top_winner['grid_position']}")
        print(f"  Winning Probability: {top_winner['win_probability']:.2f} 🏆")
    else:
        print(f"No single driver is predicted to win with high confidence (> 50%).")
        print(f"The most likely candidate is {top_winner['driver']} ({top_winner['constructor']})")
        print(f"  Starting Grid Position: {top_winner['grid_position']}")
        print(f"  Winning Probability: {top_winner['win_probability']:.2f} ❌")
        print("\nNote: Predicting a winner is challenging due to the class imbalance (very few wins per race).")
        print("The model might output low probabilities for 'win' even for the most likely candidate.")

print("\nPrediction process complete.")