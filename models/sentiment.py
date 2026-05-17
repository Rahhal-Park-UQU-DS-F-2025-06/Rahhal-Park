import re
import string
import random
import numpy as np
import joblib

model = joblib.load("models/API/sentiment_model.pkl")
vectorizer = joblib.load("models/API/vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

positive_responses = [
    "We're thrilled you had an amazing experience!",
    "Your positive feedback truly motivates us!"
]

negative_responses = [
    "We're sorry your experience wasn't ideal.",
    "We are committed to doing better next time."
]

def analyze_sentiment(text):

    cleaned = clean_text(text)

    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]

    if prediction == "positive":
        msg = random.choice(positive_responses)
    else:
        msg = random.choice(negative_responses)

    return prediction, msg