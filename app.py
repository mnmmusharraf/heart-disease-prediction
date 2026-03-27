from flask import Flask, render_template, request
import joblib
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load columns + booster model
COLUMNS = joblib.load("xgb_columns.pkl")

booster = xgb.Booster()
booster.load_model("heart_xgb.json")



def make_features_single(raw: dict) -> pd.DataFrame:
    # EXACT training column names
    X = pd.DataFrame([{
        "Age": raw["Age"],
        "Sex": raw["Sex"],
        "Chest pain type": raw["ChestPain"],
        "BP": raw["BP"],
        "Cholesterol": raw["Cholesterol"],
        "FBS over 120": raw["FBS"],
        "EKG results": raw["EKG"],
        "Max HR": raw["MaxHR"],
        "Exercise angina": raw["ExerciseAngina"],
        "ST depression": raw["STDepression"],
        "Slope of ST": raw["Slope"],
        "Number of vessels fluro": raw["Vessels"],
        "Thallium": raw["Thallium"],
    }])

    # Feature engineering (same logic)
    X["Age_group"] = pd.cut(X["Age"], bins=[0, 40, 55, 100], labels=["young", "mid", "old"])
    X["Chol_cat"] = pd.cut(X["Cholesterol"], bins=[0, 200, 240, 1000], labels=["low", "mid", "high"])

    X["Age_BP"] = X["Age"] * X["BP"]
    X["Age_Chol"] = X["Age"] * X["Cholesterol"]
    X["BP_Chol"] = X["BP"] * X["Cholesterol"]

    X["Chol_per_Age"] = X["Cholesterol"] / (X["Age"] + 1e-6)
    X["MaxHR_per_Age"] = X["Max HR"] / (X["Age"] + 1e-6)

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    # Align columns to training
    X = X.reindex(columns=COLUMNS, fill_value=0)

    return X



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        raw = {
            "Age": float(request.form["Age"]),
            "Sex": float(request.form["Sex"]),
            "ChestPain": float(request.form["ChestPain"]),
            "BP": float(request.form["BP"]),
            "Cholesterol": float(request.form["Cholesterol"]),
            "FBS": float(request.form["FBS"]),
            "EKG": float(request.form["EKG"]),
            "MaxHR": float(request.form["MaxHR"]),
            "ExerciseAngina": float(request.form["ExerciseAngina"]),
            "STDepression": float(request.form["STDepression"]),
            "Slope": float(request.form["Slope"]),
            "Vessels": float(request.form["Vessels"]),
            "Thallium": float(request.form["Thallium"]),
        }

        X = make_features_single(raw)
        dm = xgb.DMatrix(X, feature_names=COLUMNS)
        prob = float(booster.predict(dm)[0])

        prediction = f"Heart Disease Probability: {prob:.4f}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
