"""
config.py
---------
Central configuration file for the News Article Classification project.
Stores file paths, category labels, and model hyperparameters so that
train_model.py, predict.py, and app.py all stay in sync.
"""

import os

# ----------------------------------------------------------------------
# Base directory of the project
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Dataset paths
# ----------------------------------------------------------------------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "train.csv")

# ----------------------------------------------------------------------
# Model artifact paths
# ----------------------------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

# ----------------------------------------------------------------------
# Static image output paths (generated during training)
# ----------------------------------------------------------------------
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
CONFUSION_MATRIX_IMAGE = os.path.join(STATIC_IMAGES_DIR, "accuracy.png")

# ----------------------------------------------------------------------
# News categories (AG News style)
# ----------------------------------------------------------------------
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

# ----------------------------------------------------------------------
# TF-IDF Vectorizer settings
# ----------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# ----------------------------------------------------------------------
# Train/Test split settings
# ----------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Flask settings
# ----------------------------------------------------------------------
DEBUG_MODE = True
HOST = "0.0.0.0"
PORT = 5000
