"""
predict.py
----------
Loads the trained model, TF-IDF vectorizer, and label encoder, and
exposes a simple function to classify new/unseen news article text.

The text passed in is cleaned using the EXACT SAME preprocessing
pipeline (preprocess.clean_text) that was used during training, which
is essential for the model to work correctly.
"""

import os
import joblib
import numpy as np

import config
from preprocess import clean_text


class NewsClassifier:
    """Wraps the trained model + vectorizer + label encoder for easy reuse."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model artifacts from disk. Raises a clear error if missing."""
        missing = [p for p in
                   [config.MODEL_PATH, config.VECTORIZER_PATH, config.LABEL_ENCODER_PATH]
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Model artifacts not found: "
                f"{missing}. Please run 'python train_model.py' first."
            )

        self.model = joblib.load(config.MODEL_PATH)
        self.vectorizer = joblib.load(config.VECTORIZER_PATH)
        self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)

    def predict(self, text: str) -> dict:
        """
        Predict the news category for a given raw text string.

        Args:
            text (str): Raw news article text (title + body, or either).

        Returns:
            dict: {
                "category": str,             # predicted category label
                "confidence": float,         # confidence of top prediction (0-100)
                "cleaned_text": str,         # text after NLP preprocessing
                "probabilities": {           # confidence for every category
                    "World": float, "Sports": float,
                    "Business": float, "Sci/Tech": float
                }
            }

        Raises:
            ValueError: if the text is empty or becomes empty after cleaning
                        (e.g. it contained only stopwords/punctuation).
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty. Please enter a news article.")

        cleaned = clean_text(text)

        if not cleaned:
            raise ValueError(
                "The input text did not contain enough meaningful words to "
                "classify. Please enter a more descriptive news article."
            )

        features = self.vectorizer.transform([cleaned])

        # Not every scikit-learn estimator exposes predict_proba (though both
        # LogisticRegression and MultinomialNB do), so we guard just in case.
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
        else:
            decision = self.model.decision_function(features)[0]
            exp_scores = np.exp(decision - np.max(decision))
            probabilities = exp_scores / exp_scores.sum()

        predicted_index = int(np.argmax(probabilities))
        predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(probabilities[predicted_index]) * 100

        prob_dict = {
            self.label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 2)
            for i, p in enumerate(probabilities)
        }

        return {
            "category": predicted_label,
            "confidence": round(confidence, 2),
            "cleaned_text": cleaned,
            "probabilities": prob_dict,
        }


# ------------------------------------------------------------------
# Module-level singleton so app.py doesn't reload the model on every
# request. Instantiated lazily via get_classifier().
# ------------------------------------------------------------------
_classifier_instance = None


def get_classifier() -> NewsClassifier:
    """Return a cached NewsClassifier instance (loads artifacts only once)."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = NewsClassifier()
    return _classifier_instance


if __name__ == "__main__":
    clf = get_classifier()
    sample = ("Scientists at a leading university have developed a new "
              "artificial intelligence model that can predict protein "
              "structures with remarkable accuracy, opening new doors "
              "for drug discovery.")
    result = clf.predict(sample)
    print("Input     :", sample)
    print("Category  :", result["category"])
    print("Confidence:", result["confidence"])
    print("Probabilities:", result["probabilities"])
