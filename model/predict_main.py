# spam detection — tf-idf + lstm + rule engine
import joblib
import re
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " suspiciouslink ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# rule engine 
spam_keywords = [
    "free", "money", "win", "prize", "offer", "click", "urgent",
    "suspend", "verify", "password", "bank details", "claim", "login"
]

def rule_engine(text):
    score = 0
    
    # Heavy penalty--direct phishing phrases
    phishing_phrases = ["account will be suspended", "verify your identity", "bank details", "claim reward"]
    if any(phrase in text for phrase in phishing_phrases):
        score += 2 
        
    # Standard penalty--individual spam keywords
    score += sum(word in text for word in spam_keywords)

    # Heavy penalty--external links
    if "suspiciouslink" in text or "http" in text or "www" in text:
        score += 2

    return min(score / 5, 1.0)


# load models
tfidf_model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

lstm_model = load_model("lstm_model.keras")
tokenizer = joblib.load("tokenizer.pkl")

max_len = 100


# predict function
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
    lstm_prob = lstm_model.predict(padded)[0][0]

    # final scores
    final_score = (
        0.25 * rule_conf +
        0.35 * tfidf_prob +
        0.40 * lstm_prob
    )
    
    return {
        "verdict": "SPAM" if final_score > 0.5 else "HAM",
        "confidence": round(final_score, 4),
        "rule_score": round(rule_conf, 4),
        "tfidf_score": round(tfidf_prob, 4),
        "lstm_score": round(lstm_prob, 4)
    }


# Quick test
if __name__ == "__main__":
    test_email = 'Urgent! Your Account Will Be Suspended. Please click the link below to verify your identity: http://fake-login.com'
    
    print("\n--- INCOMING EMAIL ---")
    print(f"Text: {test_email}\n")
    
    result = predict(test_email)
    
    print(f"Verdict: {result['verdict']}")
    print(f"Overall Confidence: {result['confidence'] * 100}% SPAM")
    print("-" * 25)
    print("Under the Hood (Model Breakdown):")
    print(f"Rule Engine Output: {result['rule_score']}")
    print(f"TF-IDF Model Output: {result['tfidf_score']}")
    print(f"LSTM Model Output: {result['lstm_score']}\n")