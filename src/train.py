import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import load_and_clean_data

# ========================
# LOAD DATA
# ========================

df = load_and_clean_data("../data/kidney_disease.csv")

# ========================
# FRONTEND FEATURES ONLY
# ========================

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

# ========================
# TRAIN TEST SPLIT
# ========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# MODEL
# ========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Classes:", model.classes_)  # VERY IMPORTANT

# ========================
# SAVE MODEL
# ========================

saved_object = {
    "model": model,
    "features": selected_features,
    "classes": model.classes_
}

pickle.dump(saved_object,
            open("../model/ckd_model.pkl", "wb"))

print("Model saved successfully")
print("Feature Importance:")
print(model.feature_importances_)

