import pickle
from collections import Counter
from preprocess import load_and_clean_data

# =====================
# LOAD DATA
# =====================

df = load_and_clean_data("../data/kidney_disease.csv")

selected_features = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr"
]

X = df[selected_features]

# =====================
# LOAD MODEL
# =====================

saved = pickle.load(open("../model/ckd_model.pkl", "rb"))
model = saved["model"]

# sample patient
input_data = X.iloc[[0]]

print("\n--- EACH TREE PREDICTION ---\n")

tree_preds = []

for i, tree in enumerate(model.estimators_):
    pred = tree.predict(input_data)[0]
    tree_preds.append(pred)
    print(f"Tree {i+1}: {pred}")

# =====================
# VOTING
# =====================

votes = Counter(tree_preds)

print("\nVoting Result:", votes)

final_pred = model.predict(input_data)[0]

print("\nFinal Ensemble Prediction:", final_pred)

print("\nPrediction Probability:")
print(model.predict_proba(input_data))
