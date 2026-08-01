"""
preprocess.py
-------------
Text preprocessing pipeline for the News Article Classification project.

The SAME function (clean_text) is used both while training the model
(train_model.py) and while predicting on new/unseen text (predict.py),
which guarantees consistent behaviour between training and inference.

Pipeline steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove special characters
    5. Tokenization
    6. Stopword removal
    7. Lemmatization
    8. Whitespace cleanup
"""

import re
import string
import nltk

# ------------------------------------------------------------------
# Ensure required NLTK corpora/models are available.
# This block tries to use already-downloaded data first, and only
# reaches out to the network if the data is missing.
# ------------------------------------------------------------------
_REQUIRED_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

for _resource_path, _package_name in _REQUIRED_NLTK_PACKAGES:
    try:
        nltk.data.find(_resource_path)
    except (LookupError, OSError):
        # LookupError = package genuinely missing.
        # OSError = a broken/partial previous download (corrupted folder).
        # In both cases, re-downloading resolves it.
        nltk.download(_package_name, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ------------------------------------------------------------------
# Reusable NLP objects (loaded once at import time for performance)
# ------------------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Precompiled regex patterns for speed
_PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")
_NUMBER_PATTERN = re.compile(r"\d+")
_SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def to_lowercase(text: str) -> str:
    """Step 1: Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Step 2: Remove punctuation characters."""
    return _PUNCTUATION_PATTERN.sub(" ", text)


def remove_numbers(text: str) -> str:
    """Step 3: Remove numeric digits."""
    return _NUMBER_PATTERN.sub(" ", text)


def remove_special_characters(text: str) -> str:
    """Step 4: Remove any character that is not a letter or whitespace."""
    return _SPECIAL_CHAR_PATTERN.sub(" ", text)


def tokenize_text(text: str):
    """Step 5: Split text into individual word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Step 6: Remove common English stopwords."""
    return [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 1]


def lemmatize_tokens(tokens):
    """Step 7: Reduce words to their base/dictionary form."""
    return [LEMMATIZER.lemmatize(tok) for tok in tokens]


def clean_whitespace(text: str) -> str:
    """Step 8: Collapse multiple spaces into a single space and strip ends."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Applies, in order:
        lowercase -> remove punctuation -> remove numbers ->
        remove special characters -> tokenize -> remove stopwords ->
        lemmatize -> whitespace cleanup

    Args:
        text (str): Raw input news article text.

    Returns:
        str: Cleaned, lemmatized text ready for TF-IDF vectorization.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)

    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    cleaned = " ".join(tokens)
    cleaned = clean_whitespace(cleaned)
    return cleaned


if __name__ == "__main__":
    # Quick manual test
    sample = "The Stock Market CRASHED by 12% today!! Investors are worried about 2024's economy."
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
