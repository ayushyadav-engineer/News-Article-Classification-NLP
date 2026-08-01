"""
app.py
------
Flask web application for the News Article Classification project.

Routes:
    GET  /            -> Home page
    GET  /predict      -> Predict page (form)
    POST /api/predict  -> JSON API used by predict.html's JavaScript (AJAX)
    GET  /about        -> About page

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
import joblib

import config
from predict import get_classifier

app = Flask(__name__)

# ----------------------------------------------------------------------
# Load model metrics once at startup (used on the About page).
# Training must be run first: `python train_model.py`
# ----------------------------------------------------------------------
try:
    MODEL_METRICS = joblib.load(config.METRICS_PATH)
except FileNotFoundError:
    MODEL_METRICS = None

# A few example articles for the "Try an Example" button on the Predict page.
EXAMPLE_ARTICLES = [
    {
        "category": "Sports",
        "text": ("The national football team secured a dramatic last-minute "
                  "victory over their fiercest rivals in front of a sold-out "
                  "stadium, sending them straight to the top of the league "
                  "standings ahead of next week's crucial fixture.")
    },
    {
        "category": "Business",
        "text": ("The central bank raised interest rates for the third time "
                  "this year in an effort to control rising inflation, a move "
                  "that sent stock markets tumbling and left investors "
                  "worried about a potential economic slowdown.")
    },
    {
        "category": "Sci/Tech",
        "text": ("Researchers have unveiled a new artificial intelligence "
                  "model capable of diagnosing diseases from medical scans "
                  "with greater accuracy than experienced radiologists, a "
                  "breakthrough that could transform early diagnosis.")
    },
    {
        "category": "World",
        "text": ("The United Nations Security Council held an emergency "
                  "session today after neighboring nations called for "
                  "immediate ceasefire talks amid escalating border tensions "
                  "and growing concern from the international community.")
    },
]


@app.route("/")
def index():
    """Home page with project overview, NLP intro, workflow and features."""
    return render_template("index.html", active_page="home")


@app.route("/predict")
def predict_page():
    """Predict page containing the text area and prediction UI."""
    return render_template(
        "predict.html",
        active_page="predict",
        categories=config.CATEGORIES,
        examples=EXAMPLE_ARTICLES,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint used by the JavaScript on the Predict page.

    Expects JSON body: { "text": "<news article text>" }
    Returns JSON: { "success": bool, "category": ..., "confidence": ...,
                    "probabilities": {...} }  or  { "success": false, "error": ... }
    """
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        classifier = get_classifier()
        result = classifier.predict(text)

        return jsonify({
            "success": True,
            "category": result["category"],
            "confidence": result["confidence"],
            "cleaned_text": result["cleaned_text"],
            "probabilities": result["probabilities"],
        })

    except ValueError as ve:
        # Expected/handled errors (empty input, no meaningful words, etc.)
        return jsonify({"success": False, "error": str(ve)}), 400

    except FileNotFoundError as fe:
        # Model artifacts missing -> tell the developer clearly.
        return jsonify({
            "success": False,
            "error": "Model is not trained yet. Please run 'python train_model.py' "
                     "and restart the Flask server."
        }), 500

    except Exception as exc:  # pragma: no cover - safety net
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {exc}"
        }), 500


@app.route("/about")
def about():
    """About page with dataset info, model details, accuracy, and tech stack."""
    return render_template(
        "about.html",
        active_page="about",
        categories=config.CATEGORIES,
        metrics=MODEL_METRICS,
    )


@app.errorhandler(404)
def page_not_found(_e):
    return render_template("index.html", active_page="home"), 404


if __name__ == "__main__":
    app.run(debug=config.DEBUG_MODE, host=config.HOST, port=config.PORT)
