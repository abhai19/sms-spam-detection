"""
train_model.py
Trains an SMS spam classifier and saves it, plus evaluation results.

Run from the project folder:  python train_model.py
"""

import csv
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ---------- Paths (relative to THIS file, so they work on Mac and in the cloud) ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_PATH = BASE_DIR / "model" / "spam_pipeline.joblib"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_SEED = 42
TEST_SIZE = 0.20


def load_data():
    """Read the UCI file: one message per line, 'label<TAB>text', no header."""
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {DATA_PATH}\n"
            "Follow the download steps in README.md first."
        )
    # quoting=csv.QUOTE_NONE because SMS texts contain quote characters
    try:
        df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"],
                         quoting=csv.QUOTE_NONE, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"],
                         quoting=csv.QUOTE_NONE, encoding="latin-1")
    df = df.dropna()
    df["target"] = (df["label"] == "spam").astype(int)  # spam = 1, ham = 0
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} messages "
          f"({int(df['target'].sum())} spam, {int((1 - df['target']).sum())} ham)")

    # ---------- Stratified 80/20 split with a fixed seed ----------
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"],
        test_size=TEST_SIZE, stratify=df["target"], random_state=RANDOM_SEED,
    )

    # ---------- Pipeline: TF-IDF turns text into numbers, Naive Bayes classifies ----------
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("nb", MultinomialNB()),
    ])
    pipeline.fit(X_train, y_train)  # the WHOLE pipeline sees training data only

    # ---------- Evaluate on the held-out test set ----------
    y_pred = pipeline.predict(X_test)
    metrics = {
        "dataset_total": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "spam_precision": round(float(precision_score(y_test, y_pred, pos_label=1)), 4),
        "spam_recall": round(float(recall_score(y_test, y_pred, pos_label=1)), 4),
        "spam_f1": round(float(f1_score(y_test, y_pred, pos_label=1)), 4),
        "random_seed": RANDOM_SEED,
        "scikit_learn_version": sklearn.__version__,
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "true_legit_pred_legit": int(cm[0, 0]),
        "true_legit_pred_spam": int(cm[0, 1]),
        "true_spam_pred_legit": int(cm[1, 0]),
        "true_spam_pred_spam": int(cm[1, 1]),
    }

    # ---------- Save model, metrics, confusion-matrix picture ----------
    MODEL_PATH.parent.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Spam"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix (20% held-out test set)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # ---------- One misclassified example (if any) ----------
    wrong = X_test[y_test.values != y_pred]
    example_file = RESULTS_DIR / "misclassified_example.txt"
    if len(wrong) > 0:
        idx = wrong.index[0]
        actual = "Spam" if y_test.loc[idx] == 1 else "Legitimate"
        predicted = "Spam" if y_pred[list(X_test.index).index(idx)] == 1 else "Legitimate"
        example_file.write_text(
            f"Message  : {wrong.iloc[0]}\nActual   : {actual}\nPredicted: {predicted}\n")
        print(f"\nMisclassified example:\n  {wrong.iloc[0]}\n  actual={actual}, predicted={predicted}")
    else:
        example_file.write_text("No misclassified messages in this run.\n")
        print("\nNo misclassified messages in this run.")

    # ---------- Print summary ----------
    print("\n===== Test-set results =====")
    print(f"Accuracy       : {metrics['accuracy']}")
    print(f"Spam precision : {metrics['spam_precision']}")
    print(f"Spam recall    : {metrics['spam_recall']}")
    print(f"Spam F1-score  : {metrics['spam_f1']}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")
    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved results -> {RESULTS_DIR}")
    print(f"scikit-learn version used: {sklearn.__version__}")


if __name__ == "__main__":
    main()
