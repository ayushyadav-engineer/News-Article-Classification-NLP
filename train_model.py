"""
train_model.py
---------------
Trains the News Article Classification model.

Workflow:
    1. Load the AG-News-style dataset (dataset/train.csv)
    2. Preprocess every article using preprocess.clean_text()
    3. Convert text to TF-IDF feature vectors
    4. Split into train/test sets
    5. Train Logistic Regression and Multinomial Naive Bayes
    6. Compare both models on accuracy / precision / recall / F1
    7. Automatically select and save the best performing model
    8. Save the TF-IDF vectorizer and label encoder
    9. Save a confusion matrix + model comparison chart image

Run:
    python train_model.py
"""

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe for servers/CI)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import config
from preprocess import clean_text


def load_dataset() -> pd.DataFrame:
    """Load the CSV dataset and combine title + description into one text field."""
    print(f"[1/8] Loading dataset from: {config.DATASET_PATH}")
    df = pd.read_csv(config.DATASET_PATH)

    # Combine title and description into a single "text" column (AG-News style)
    if "title" in df.columns and "description" in df.columns:
        df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    elif "text" not in df.columns:
        raise ValueError("Dataset must contain either a 'text' column, "
                          "or both 'title' and 'description' columns.")

    df = df.dropna(subset=["category", "text"])
    df = df[df["text"].str.strip() != ""]
    print(f"      Loaded {len(df)} rows across categories: "
          f"{sorted(df['category'].unique())}")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the NLP cleaning pipeline to every article."""
    print("[2/8] Preprocessing text (lowercase, punctuation/number removal, "
          "tokenization, stopword removal, lemmatization)...")
    start = time.time()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]
    print(f"      Preprocessing completed in {time.time() - start:.2f}s")
    return df


def vectorize_text(df: pd.DataFrame):
    """Fit a TF-IDF vectorizer on the cleaned text."""
    print("[3/8] Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    X = vectorizer.fit_transform(df["clean_text"])
    print(f"      TF-IDF matrix shape: {X.shape}")
    return X, vectorizer


def encode_labels(df: pd.DataFrame):
    """Encode category labels into numeric form."""
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["category"])
    return y, encoder


def split_data(X, y):
    """Split features/labels into train and test sets."""
    print("[4/8] Splitting into train/test sets "
          f"({int((1 - config.TEST_SIZE) * 100)}/{int(config.TEST_SIZE * 100)})...")
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def evaluate_model(name, model, X_test, y_test, label_encoder):
    """Compute accuracy, precision, recall, F1 and print a report."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    print(f"\n      --- {name} ---")
    print(f"      Accuracy : {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall   : {metrics['recall']:.4f}")
    print(f"      F1 Score : {metrics['f1']:.4f}")
    print("\n" + classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    return metrics, y_pred


def plot_results(lr_metrics, nb_metrics, y_test, lr_pred, best_name, label_encoder):
    """Save a combined figure: model comparison bar chart + confusion matrix."""
    print("[7/8] Generating evaluation charts...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Model comparison bar chart ---
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    lr_values = [lr_metrics["accuracy"], lr_metrics["precision"],
                 lr_metrics["recall"], lr_metrics["f1"]]
    nb_values = [nb_metrics["accuracy"], nb_metrics["precision"],
                 nb_metrics["recall"], nb_metrics["f1"]]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, lr_values, width, label="Logistic Regression", color="#4361ee")
    axes[0].bar(x + width / 2, nb_values, width, label="Naive Bayes", color="#f72585")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Model Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # --- Confusion matrix for the best model ---
    cm = confusion_matrix(y_test, lr_pred)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title(f"Confusion Matrix ({best_name})")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticks(range(len(label_encoder.classes_)))
    axes[1].set_yticks(range(len(label_encoder.classes_)))
    axes[1].set_xticklabels(label_encoder.classes_, rotation=45, ha="right")
    axes[1].set_yticklabels(label_encoder.classes_)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_IMAGE, dpi=150)
    plt.close(fig)
    print(f"      Saved chart to: {config.CONFUSION_MATRIX_IMAGE}")


def main():
    print("=" * 70)
    print("NEWS ARTICLE CLASSIFICATION - MODEL TRAINING")
    print("=" * 70)

    df = load_dataset()
    df = preprocess_dataset(df)

    X, vectorizer = vectorize_text(df)
    y, label_encoder = encode_labels(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---------------- Train Logistic Regression ----------------
    print("[5/8] Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, C=5.0, random_state=config.RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    lr_metrics, lr_pred = evaluate_model(
        "Logistic Regression", lr_model, X_test, y_test, label_encoder)

    # ---------------- Train Multinomial Naive Bayes ----------------
    print("[6/8] Training Multinomial Naive Bayes model...")
    nb_model = MultinomialNB(alpha=0.3)
    nb_model.fit(X_train, y_train)
    nb_metrics, nb_pred = evaluate_model(
        "Multinomial Naive Bayes", nb_model, X_test, y_test, label_encoder)

    # ---------------- Compare & select best model ----------------
    if lr_metrics["accuracy"] >= nb_metrics["accuracy"]:
        best_model, best_name, best_metrics = lr_model, "Logistic Regression", lr_metrics
    else:
        best_model, best_name, best_metrics = nb_model, "Multinomial Naive Bayes", nb_metrics

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}  (Accuracy: {best_metrics['accuracy']:.4f})")
    print("=" * 70)

    plot_results(lr_metrics, nb_metrics, y_test, lr_pred, best_name, label_encoder)

    # ---------------- Save artifacts ----------------
    print("[8/8] Saving model artifacts with Joblib...")
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    joblib.dump({
        "best_model_name": best_name,
        "logistic_regression": lr_metrics,
        "naive_bayes": nb_metrics,
        "categories": list(label_encoder.classes_),
        "training_samples": int(X_train.shape[0]),
        "testing_samples": int(X_test.shape[0]),
    }, config.METRICS_PATH)

    print(f"      Model saved      -> {config.MODEL_PATH}")
    print(f"      Vectorizer saved -> {config.VECTORIZER_PATH}")
    print(f"      Label encoder    -> {config.LABEL_ENCODER_PATH}")
    print(f"      Metrics saved    -> {config.METRICS_PATH}")
    print("\nTraining complete! You can now run: python app.py")


if __name__ == "__main__":
    main()
