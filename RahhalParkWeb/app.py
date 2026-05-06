from flask import Flask, render_template, request, jsonify
from models.sentiment import analyze_sentiment
from models.dashboard_model import predict_dashboard
from models.Emergency import detect_lang_auto, keyword_category, format_handoff, ask_menu

# (RECOMMENDATION)
import numpy as np
import pickle
from tensorflow import keras

app = Flask(__name__)

reviews = []

# (INDEX)
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    msg = None

    if request.method == 'POST':
        text = request.form['text']
        result, msg = analyze_sentiment(text)

    return render_template('index.html', result=result, msg=msg)


# home
@app.route('/home')
def main_home():
    return render_template('home.html')


#(SENTIMENT)
@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

all_reviews = []

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    result, msg = analyze_sentiment(text)

    all_reviews.append(result)

    total = len(all_reviews)
    positive = all_reviews.count('positive')
    negative = all_reviews.count('negative')

    pos_percent = round((positive / total) * 100, 2) if total > 0 else 0
    neg_percent = round((negative / total) * 100, 2) if total > 0 else 0

    return jsonify({
        'result': result,
        'msg': msg,
        'total': total,
        'pos_percent': pos_percent,
        'neg_percent': neg_percent
    })

#(DASHBOARD)
from models.dashboard_model import predict_dashboard
@app.route('/predict_dashboard', methods=['POST'])
def predict_dashboard_api():

    data = request.get_json()
    date = data['date']

    crowd, wait = predict_dashboard(date)

    return jsonify({
        "crowd": crowd,
        "wait": wait   
    })
@app.route('/waiting')
def waiting():
    return render_template('waiting.html')

#(RECOMMENDATION)
rec_model = keras.models.load_model("models/API/recommendation_model.h5")

with open("models/API/encoders_X.pkl", "rb") as f:
    encoders_X = pickle.load(f)

with open("models/API/label_encoder_y1.pkl", "rb") as f:
    le_y1 = pickle.load(f)

with open("models/API/label_encoder_y2.pkl", "rb") as f:
    le_y2 = pickle.load(f)

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    input_data = {
        "Age_Group": data['age'].lower().strip(),
        "weight_kg": float(data['weight']),
        "diseases": data['disease'].lower().strip(),
        "accompanied_with": data['group'].lower().strip(),
        "preference": data['preference'].lower().strip(),
        "Height_cm": float(data['height']),
        "Fear_of_Heights": data['fear'].lower().strip()
    }

    order = [
        "Age_Group",
        "weight_kg",
        "diseases",
        "accompanied_with",
        "preference",
        "Height_cm",
        "Fear_of_Heights"
    ]

    values = []
    for col in order:
        val = input_data[col]

        if col in encoders_X:
            try:
                val = encoders_X[col].transform([val])[0]
            except:
                val = 0

        values.append(float(val))

    values = np.array([values])

    pred1, pred2 = rec_model.predict(values)

    return jsonify({
        "best_fit": le_y1.inverse_transform([np.argmax(pred1)])[0],
        "example": le_y2.inverse_transform([np.argmax(pred2)])[0]
    })

# (EMERGENCY)
@app.route('/emergency')
def emergency():
    return render_template('emargency.html')


@app.route('/emergency_action', methods=['POST'])
def emergency_action():
    data = request.get_json()
    situation = data['situation']

    lang = detect_lang_auto(situation)
    cat, confidence = keyword_category(situation, lang)

    if confidence == 0:
        response = ask_menu(lang)
    else:
        response = format_handoff(lang, cat)

    return jsonify({
        "response": response
    })

if __name__ == "__main__":
    app.run(debug=True, port=5010)

