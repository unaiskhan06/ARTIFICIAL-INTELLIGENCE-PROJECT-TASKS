"""
Email Classifier - Spam vs Ham (Not Spam)
=========================================
A clean, efficient email classification tool using TF-IDF features
and a Multinomial Naive Bayes classifier.

Features:
  1. Spam / Not-Spam classification with confidence score.
  2. Built-in labeled training dataset (runs out of the box).
  3. Optional custom CSV training data (columns: text, label).
  4. Unique bonus features:
     - URL / currency / uppercase-ratio heuristics blended into features.
     - Batch classification of multiple emails at once.
     - Per-email explanation of why it was flagged.

Usage:
  python email_classifier.py                 # demo on sample emails
  python email_classifier.py --train data.csv --email "Win free money!"
  python email_classifier.py --batch emails.txt
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Built-in labeled dataset so the script runs without external files.
# --------------------------------------------------------------------------- #
SAMPLE_DATA = [
    ("Win a FREE iPhone now! Click http://win-prize.biz", "spam"),
    ("URGENT: Your account is suspended, verify immediately", "spam"),
    ("Congratulations! You've won $1,000,000 lottery", "spam"),
    ("Limited time offer, buy one get one free, click here", "spam"),
    ("Cheap meds online, no prescription needed, 50% off", "spam"),
    ("Hi team, please review the attached report before Monday", "ham"),
    ("Are we still on for lunch tomorrow at 1pm?", "ham"),
    ("Please find the invoice for last month attached.", "ham"),
    ("Don't forget the standup at 9am.", "ham"),
    ("Thanks for sending the documents, I'll review them tonight.", "ham"),
    ("Earn money from home, no experience required, sign up now!!!", "spam"),
    ("Your Amazon order has shipped and will arrive Tuesday.", "ham"),
    ("Claim your exclusive reward now before it expires tonight", "spam"),
    ("Can you forward me the meeting notes from yesterday?", "ham"),
    ("Double your income in 7 days, guaranteed results, click link", "spam"),
    ("Reminder: submit your timesheet by end of day Friday.", "ham"),
    ("FREE vacation package, you are our lucky winner today!", "spam"),
    ("I'll be working from home on Thursday.", "ham"),
    ("Act now! Limited stock, 90% discount, free shipping worldwide", "spam"),
    ("Great presentation today, well done on the project.", "ham"),
]


# --------------------------------------------------------------------------- #
# Heuristic feature engineering (the "unique" twist).
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
CURRENCY_RE = re.compile(r"[$€£₹]")
EXCLAIM_RE = re.compile(r"!")


def heuristic_features(text: str) -> dict:
    """Return lightweight spam-signal heuristics for a single email."""
    words = text.split()
    n = max(len(words), 1)
    upper_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    return {
        "has_url": int(bool(URL_RE.search(text))),
        "has_currency": int(bool(CURRENCY_RE.search(text))),
        "uppercase_ratio": round(upper_words / n, 3),
        "exclaim_count": len(EXCLAIM_RE.findall(text)),
    }


def heuristics_matrix(texts) -> np.ndarray:
    """Stack heuristic features for a list of texts into an array."""
    rows = [list(heuristic_features(t).values()) for t in texts]
    return np.array(rows, dtype=float)


# --------------------------------------------------------------------------- #
# Classifier wrapper combining TF-IDF + heuristics + Naive Bayes.
# --------------------------------------------------------------------------- #
class EmailClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", lowercase=True, ngram_range=(1, 2)
        )
        self.model = MultinomialNB()
        self._fitted = False

    def fit(self, texts, labels):
        tfidf = self.vectorizer.fit_transform(texts)
        heur = heuristics_matrix(texts)
        # Scale heuristics so they are comparable to TF-IDF weights.
        heur_scaled = heur * 3.0
        X = np.hstack([tfidf.toarray(), heur_scaled])
        self.model.fit(X, labels)
        self._fitted = True
        return self

    def _vectorize(self, texts):
        tfidf = self.vectorizer.transform(texts)
        heur = heuristics_matrix(texts) * 3.0
        return np.hstack([tfidf.toarray(), heur])

    def predict(self, texts):
        if not self._fitted:
            raise RuntimeError("Classifier is not trained yet.")
        X = self._vectorize(texts)
        labels = self.model.predict(X)
        probs = self.model.predict_proba(X)
        results = []
        classes = self.model.classes_
        for label, prob in zip(labels, probs):
            idx = list(classes).index(label)
            results.append((label, round(float(prob[idx]), 4)))
        return results

    def explain(self, text: str) -> str:
        """Short human-readable explanation of why an email was flagged."""
        feats = heuristic_features(text)
        label, conf = self.predict([text])[0]
        reasons = []
        if feats["has_url"]:
            reasons.append("contains a URL")
        if feats["has_currency"]:
            reasons.append("mentions currency/money")
        if feats["uppercase_ratio"] > 0.2:
            reasons.append("heavy use of UPPERCASE words")
        if feats["exclaim_count"] >= 2:
            reasons.append("multiple exclamation marks")
        if not reasons:
            reasons.append("no strong spam signals; looks like normal text")
        signals = "; ".join(reasons)
        return f"[{label.upper()}] confidence={conf:.2%} -> {signals}"


# --------------------------------------------------------------------------- #
# Data loading helpers.
# --------------------------------------------------------------------------- #
def load_training_data(csv_path: str | None):
    if csv_path:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if not {"text", "label"}.issubset(df.columns):
            sys.exit("CSV must have 'text' and 'label' columns.")
        return df["text"].tolist(), df["label"].tolist()
    texts = [t for t, _ in SAMPLE_DATA]
    labels = [l for _, l in SAMPLE_DATA]
    return texts, labels


# --------------------------------------------------------------------------- #
# CLI entry point.
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Email Spam Classifier")
    parser.add_argument("--train", help="Path to a CSV with 'text,label' columns.")
    parser.add_argument("--email", help="A single email text to classify.")
    parser.add_argument("--batch", help="Path to a text file with one email per line.")
    args = parser.parse_args()

    texts, labels = load_training_data(args.train)
    clf = EmailClassifier().fit(texts, labels)

    if args.email:
        label, conf = clf.predict([args.email])[0]
        print(f"Result: {label.upper()} (confidence: {conf:.2%})")
        print(clf.explain(args.email))
        return

    if args.batch:
        batch_texts = [
            line.strip()
            for line in Path(args.batch).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for text, (label, conf) in zip(batch_texts, clf.predict(batch_texts)):
            print(f"[{label.upper():4} {conf:.2%}] {text[:70]}")
        return

    # Default demo run.
    print("=" * 60)
    print(" Email Spam Classifier - Demo")
    print("=" * 60)
    demo_emails = [
        "Congratulations! You have won a $500 gift card, click now",
        "Hey, can we move our meeting to 3pm today?",
        "URGENT: verify your bank account at http://secure-bank-login.xyz",
        "Please send the quarterly report when you get a chance.",
    ]
    for email in demo_emails:
        print(f"\nEmail: {email}")
        print(clf.explain(email))


if __name__ == "__main__":
    main()
