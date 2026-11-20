"""Train and compare spam classifiers on the SMS Spam Collection dataset.

Download spam.csv from Kaggle ("SMS Spam Collection Dataset") and put it in data/spam.csv
Run:  python train.py
"""
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["clean"] = df["text"].apply(clean)
print(df["label"].value_counts(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced")),
}

best_name, best_f1, best_pipe = None, -1, None
for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    f1 = f1_score(y_test, preds)
    print(f"=== {name} (spam F1 = {f1:.4f}) ===")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    if f1 > best_f1:
        best_name, best_f1, best_pipe = name, f1, pipe

print(f"Best model: {best_name} (F1 = {best_f1:.4f})")

cm = confusion_matrix(y_test, best_pipe.predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

joblib.dump(best_pipe, "model.pkl")
print("Saved model.pkl and confusion_matrix.png")
