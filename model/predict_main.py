"""Spam detection inference helpers."""

from pathlib import Path
import joblib
import re
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



# rule engine ( kept it very lightweight)

spam_keywords = [
    "free", "money", "win", "prize",
    "offer", "click", "urgent"
]

def rule_engine(text):
    score = sum(word in text for word in spam_keywords)

    if "http" in text or "www" in text:
        score += 2

    return min(score / 5, 1.0)


BASE_DIR = Path(__file__).resolve().parent


def load_artifact(filename: str):
    return joblib.load(BASE_DIR / filename)


#load models
tfidf_model = load_artifact("spam_model.pkl")
vectorizer = load_artifact("vectorizer.pkl")

lstm_model = load_model(BASE_DIR / "lstm_model.keras")
tokenizer = load_artifact("tokenizer.pkl")

max_len = 100


#predict function
def predict(text):
    raw = text.lower()
    cleaned = clean_text(raw)

    # rule
    rule_conf = rule_engine(raw)

    # tfidf
    vec = vectorizer.transform([cleaned])
    tfidf_prob = tfidf_model.predict_proba(vec)[0][1]

    # lstm
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    lstm_prob = lstm_model.predict(padded, verbose=0)[0][0]

    # final scores
    final_score = (
        0.25 * rule_conf +
        0.35 * tfidf_prob +
        0.40 * lstm_prob
    )

    # decision
    if final_score > 0.55:
        label = "SPAM"
    else:
        label = "HAM"

    return {
        "label": label,
        "score": round(float(final_score), 3),
        "rule_conf": round(float(rule_conf), 3),
        "tfidf_prob": round(float(tfidf_prob), 3),
        "lstm_prob": round(float(lstm_prob), 3)
    }



# cli
if __name__ == "__main__":
    user_input = input("\nEnter email text: ")

    result = predict(user_input)

    print("\nFINAL RESULT:")
    print(result)