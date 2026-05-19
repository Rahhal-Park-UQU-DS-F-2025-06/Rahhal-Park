import joblib
import numpy as np
import json
import os
from datetime import datetime


crowd_model = joblib.load("models/API/crowd_classifier.pkl")
wait_model = joblib.load("models/API/wait_time_regressor.pkl")
label_encoder = joblib.load("models/API/label_encoder.pkl")
day_encoder = joblib.load("models/API/day_encoder.pkl")

HISTORY_PATH = "models/API/user_history.json"


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r") as f:
        try:
            return json.load(f)
        except:
            return []


def save_history(data):
    with open(HISTORY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def compute_adaptive_features(history, month, is_weekend):

    if month in [6, 7, 8]:  # الصيف 
        base = 60000
    elif month in [12, 1]:  # إجازات
        base = 50000
    elif month in [3, 4, 5]:  # ربيع
        base = 35000
    else:  # باقي الأيام
        base = 25000

    if is_weekend:
        base *= 1.5

    if history:
        waits = [h["wait"] for h in history if "wait" in h]

        if waits:
            avg_wait = np.mean(waits)

            learned = avg_wait * 1200

            base = max(base, learned)

    lag1 = base
    lag7 = base * 1.05
    roll7 = base * 1.02
    diff1 = base * 0.05

    return lag1, lag7, roll7, diff1


def predict_dashboard(date_str):

    history = load_history()

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    month = date_obj.month
    day_name = date_obj.strftime("%A")
    is_weekend = int(day_name in ["Saturday", "Sunday"])

    # encoding
    day_encoded = day_encoder.transform([day_name])[0]

    # 🔥 الذكاء هنا
    lag1, lag7, roll7, diff1 = compute_adaptive_features(
        history, month, is_weekend
    )

   
    X = np.array([[
        month,
        is_weekend,
        day_encoded,
        lag1,
        lag7,
        roll7,
        diff1
    ]])

    print("FEATURES:", X)

    # prediction
    crowd_pred = crowd_model.predict(X)
    wait_pred = wait_model.predict(X)

    crowd = label_encoder.inverse_transform(crowd_pred)[0]
    wait = int(wait_pred[0])

    print("PREDICTED:", crowd, wait)


    history.append({
        "date": date_str,
        "crowd": crowd,
        "wait": wait
    })

    save_history(history)

    return crowd, wait


