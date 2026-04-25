import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import load_and_clean_data


# =====================================
# PROJECT ROOT PATH (WORKS EVERYWHERE)
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================
# DATASET PATH
# =====================================

data_path = os.path.join(BASE_DIR, "data", "kidney_disease.csv")


# =====================================
# MODEL SAVE PATH
# =====================================

model_path = os.path.join(BASE_DIR, "model", "ckd_model.pkl")


# =====================================
# LOAD DATASET
# =====================================

df = load_and_clean_data(data_path)


# =====================================
# SELECT ONLY FRONTEND FEATURES
# (Do NOT change — matches Streamlit UI)
# =====================================

selected_features = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr"
]


X = df[selected_features]
y = df["classification"]


# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================
# RANDOM FOREST MODEL
# =====================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(X_train, y_train)


# =====================================
# MODEL EVALUATION
# =====================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)
print("Classes:", model.classes_)


# =====================================
# ENSURE MODEL DIRECTORY EXISTS
# =====================================

os.makedirs(os.path.dirname(model_path), exist_ok=True)


# =====================================
# SAVE MODEL OBJECT (USED BY STREAMLIT)
# =====================================

saved_object = {
    "model": model,
    "features": selected_features,
    "classes": model.classes_
}


pickle.dump(saved_object, open(model_path, "wb"))


# =====================================
# FEATURE IMPORTANCE OUTPUT
# =====================================

print("\nModel saved successfully at:", model_path)

print("\nFeature Importance:")

for feature, importance in zip(selected_features, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")