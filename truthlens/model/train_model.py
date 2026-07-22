"""
Trains a real TF-IDF + Logistic Regression fake-news classifier
and saves it to disk with joblib so the Flask backend can load it.

USAGE
-----
Quick start (uses the small bundled sample_dataset.csv, ~50 rows,
good enough to prove the pipeline works end-to-end):

    python model/train_model.py

For a production-quality model, download the well-known ISOT
"Fake and Real News" dataset (Fake.csv + True.csv) from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
and place Fake.csv and True.csv inside the model/ folder, then run:

    python model/train_model.py --full

This will combine both files (fake=0/real=1), train on ~44,000
articles, and produce a much more accurate model.
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT = os.path.join(BASE_DIR, "fake_news_model.pkl")
VEC_OUT = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")


def load_sample_dataset():
    path = os.path.join(BASE_DIR, "sample_dataset.csv")
    df = pd.read_csv(path)
    df["label_bin"] = (df["label"].str.upper() == "REAL").astype(int)
    return df[["text", "label_bin"]].rename(columns={"label_bin": "label"})


def load_full_dataset():
    fake_path = os.path.join(BASE_DIR, "Fake.csv")
    real_path = os.path.join(BASE_DIR, "True.csv")
    if not (os.path.exists(fake_path) and os.path.exists(real_path)):
        raise FileNotFoundError(
            "Fake.csv / True.csv not found in model/. Download the ISOT "
            "dataset from Kaggle and place both files in the model/ folder."
        )
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake["label"] = 0
    real["label"] = 1
    df = pd.concat([fake, real], ignore_index=True)
    # ISOT dataset has "title" and "text" columns — combine them
    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna(""))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df[["text", "label"]]


def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    joblib.dump(model, MODEL_OUT)
    joblib.dump(vectorizer, VEC_OUT)
    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved vectorizer -> {VEC_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="Train on the full ISOT Fake.csv/True.csv dataset instead of the small sample."
    )
    args = parser.parse_args()

    dataset = load_full_dataset() if args.full else load_sample_dataset()
    print(f"Training on {len(dataset)} labeled articles...")
    train(dataset)
