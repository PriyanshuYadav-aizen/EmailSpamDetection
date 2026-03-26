import joblib
import re
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text):
    # normalize and remove noise
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# load saved model and tokenizer
model = load_model("lstm_model.keras")
tokenizer = joblib.load("tokenizer.pkl")

max_len = 100


def predict(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    return model.predict(padded)[0][0]


if __name__ == "__main__":
    user_input = input("\nEnter email text: ")
    score = predict(user_input)
    print("\nPrediction score:", score)
    print("Meaning:", "SPAM" if score > 0.5 else "HAM")