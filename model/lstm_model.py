import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional


def clean_text(text):
    # normalize and remove noise
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# load and prepare the dataset
df = pd.read_csv("enron.csv")

df["text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")
df["label"] = df["Spam/Ham"]

df = df[["text", "label"]].dropna()
df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
df = df.dropna()

df["text"] = df["text"].astype(str).apply(clean_text)


# augment with spam samples — watch the ratio if dataset is small
extra_texts = [
    # direct spam
    "free money now",
    "win cash instantly",
    "claim your prize now",

    # conversational spam 
    "my friend told me about a way to make money online",
    "someone showed me a trick to earn money easily",
    "you should try this method to get rich quickly",
    "i heard people are making money from this",
    "my grandma told me about free money hacks",
    "i found a way to earn money without doing much",
    "this might help you earn money fast",
    "have you seen this easy money trick",

    # mixed tone
    "this is not a scam you can earn money online",
    "just sharing a way to make money from home",
    "thought you might like this earning opportunity",

    # real-like sentences
    "dont you want to make money easily",
    "you can earn money from home without effort",
    "people are earning money from this simple trick"
]
extra_texts = extra_texts * 600
extra_df = pd.DataFrame({
    "text": extra_texts,
    "label": [1] * len(extra_texts)
})

df = pd.concat([df, extra_df], ignore_index=True)


# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)


# tokenize on training vocab only
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# pad to uniform length
max_len = 100

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)


# embedding → bidirectional lstm → dropout → binary output
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


model.fit(X_train_pad, y_train, epochs=5, batch_size=64)


loss, acc = model.evaluate(X_test_pad, y_test)
print("\nAccuracy:", acc)


# save for inference
model.save("lstm_model.keras")
joblib.dump(tokenizer, "tokenizer.pkl")


# quick sanity check
test = ["You won a lottery claim your prize now"]
test_clean = [clean_text(t) for t in test]
test_seq = tokenizer.texts_to_sequences(test_clean)
test_pad = pad_sequences(test_seq, maxlen=max_len)

prediction = model.predict(test_pad)

print("\nPrediction score:", prediction[0][0])
print("Meaning:", "SPAM" if prediction[0][0] > 0.5 else "HAM")