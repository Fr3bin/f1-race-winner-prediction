# F1 Race Winner Prediction using Machine Learning

---

### Societal or Industrial Impact (for Cambodia)

This project aims to enhance F1 fan engagement in Cambodia through data-driven predictions, potentially inspiring broader motorsports interest. This could lead to:

- Government support for **F1 academies/development programs**.
- Cambodia eventually **hosting an F1 Grand Prix**, bringing significant **economic growth (tourism, jobs, infrastructure)** and **international recognition**.

---

### Problem Statement

The project aims to accurately predict F1 race winners from historical data. The main challenge is the **severe class imbalance** (one winner among many drivers), which affects prediction confidence.

---

### Research Questions

- **What:** Which F1 historical data points are most influential in predicting a race winner?
- **Why:** Why use ML for F1 prediction? (Due to abundant data, for fan engagement, and to lay groundwork for a Cambodian F1 ecosystem).
- **How:** How can a robust ML model be built and evaluated to effectively predict F1 winners despite data challenges?

---

### Contributions

- **Comprehensive ML Pipeline:** Implemented end-to-end data integration, preprocessing, model training, evaluation, and inference.
- **Class Imbalance Handling:** Addressed and analyzed challenges of predicting a rare outcome.
- **Reusable Prediction Tools:** Developed scripts for both single and multi-candidate race winner predictions.
- **Foundation for Future Work:** Created a base for more advanced F1 analytics and skill development.

---

### Dataset (Primary / Secondary)

**Secondary Dataset** (sourced from Kaggle)

### Dataset

Historical F1 race data from Kaggle, including:

- `races.csv`
- `results.csv`
- `drivers.csv`
- `constructors.csv`
  These files were merged for a comprehensive dataset.

---

### Methodology

A standard ML workflow was followed:

1.  **Data Collection & Integration:** Merging F1 CSVs.
2.  **Data Preprocessing:** Handling missing values (e.g., `\N` -> `0`), feature engineering, `LabelEncoder` for categorical data, `StandardScaler` (Z-score) for numerical features.
3.  **Data Splitting:** 80% training, 20% testing.
4.  **Model Selection:** `RandomForestClassifier`.
5.  **Model Training:** Training on scaled data.
6.  **Model Evaluation:** Assessing performance on test data.
7.  **Model Persistence:** Saving the trained model and preprocessing tools.
8.  **Inference:** Using saved model for new predictions.

---

### ML Model Selection

**RandomForestClassifier**

### Why ?

Chosen for its:

- **Suitability for Abundant Data:** Handles large datasets effectively.
- **Robustness:** Less prone to overfitting, good with noise.
- **Mixed Data Handling:** Works well with both numerical and encoded categorical features.
- **Non-Linearity:** Captures complex relationships.
- **Classification Task:** Ideal for predicting a binary outcome (win/not win).

---

### Evaluation Technique

- **Accuracy:** Overall correctness.
- **Classification Report:** Detailed Precision, Recall, F1-score (crucial for the imbalanced "win" class).
- **Confusion Matrix:** Visual breakdown of correct/incorrect predictions.

### Hyperparameter

- `n_estimators=100`: Number of trees in the forest.
- `random_state=42`: For reproducibility.
  _(Note: Further tuning of other hyperparameters could optimize performance.)_

---

### Code Implementation & Submission

- **Key Python Scripts:** `main.py` (training/evaluation), `predict.py` (single prediction), `predict_race_winner.py` (multi-candidate prediction).
- **Submission:** Project code will be submitted via a **GitHub repository link**.

---

### Reflection and Argument

The project demonstrated effective ML application despite challenges like severe class imbalance. While direct "win" probabilities remained low (due to rarity of wins), the model successfully identified the most likely candidates. The project also highlighted the importance of robust preprocessing and serves as a foundation for applying data science skills to contribute to Cambodia's development, as outlined in the impact section.

---

### Report Submission

The final project report, based on this outline, will be submitted along with the GitHub repository link.
