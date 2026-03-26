import pandas as pd
import re
import string
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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
    # conversational spam — harder for the model to catch
    "someone told me about free money hacks",
    "i heard you can earn money fast online",
    "have you seen this trick to make money",
    "my friend showed me a way to get rich quick",
    "you should try this easy money method",
    "people are making money easily from this",
    "i found a way to earn money without effort",
    # mixed patterns
    "click here to learn how to make money",
    "this is not a scam earn money fast",
    "earn money from home no experience needed",
    # soft urgency
    "you might want to check this offer",
    "just sharing this opportunity with you",
    "thought you might be interested in this offer"
] * 400

extra_df = pd.DataFrame({
    "text": extra_texts,
    "label": [1] * len(extra_texts)
})

df = pd.concat([df, extra_df], ignore_index=True)


# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)


# fit on train only
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=2.0
)

model.fit(X_train_vec, y_train)


preds = model.predict(X_test_vec)

print("\n===== TF-IDF MODEL =====\n")
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n")
print(classification_report(y_test, preds))


# save for inference
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


# quick sanity check
test = ["free money hacks"]
test_clean = [clean_text(t) for t in test]
test_vec = vectorizer.transform(test_clean)

prediction = model.predict(test_vec)

print("\nTest Input:", test[0])
print("Prediction:", prediction[0])
print("Meaning:", "SPAM" if prediction[0] == 1 else "HAM")