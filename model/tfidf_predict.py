import joblib
import re
import string


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# load trained data
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return "SPAM" if pred == 1 else "HAM"


# test
print(predict("You won a lottery claim your prize now"))