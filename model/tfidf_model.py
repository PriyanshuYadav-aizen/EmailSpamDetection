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
    # Replace with a heavy keyword instead of deleting
    text = re.sub(r"http\S+|www\S+", " suspiciouslink ", text)
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


# augmenting
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
    "people are earning money from this simple trick",
    
    # links
    "urgent your account will be suspended please click the link http://fake-login.com",
    "please click the link below to verify your identity http://secure-update.com",
    "send your bank details to claim reward http://prize-claim.com",
    "verify your account immediately by clicking http://verify-account.com",
    "unauthorized login attempt click here to secure account http://alert-login.com",
    "your password expires in 24 hours reset it here http://reset-password.com",
    "urgent action required click the link to prevent suspension http://login.com"
]

extra_texts = extra_texts * 600

extra_df = pd.DataFrame({
    "text": extra_texts,
    "label": [1] * len(extra_texts)
})


extra_df["text"] = extra_df["text"].apply(clean_text)

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


# quicksanity check
test = ["urgent your account will be suspended please click the link http://fake-login.com"]
test_clean = [clean_text(t) for t in test]
test_vec = vectorizer.transform(test_clean)

prediction = model.predict(test_vec)

print("\nTest Input:", test[0])
print("Prediction:", prediction[0])
print("Meaning:", "SPAM" if prediction[0] == 1 else "HAM")