"""Flask web application for Watson NLP emotion detection."""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

HOST = "0.0.0.0"
PORT = 5000
APP = Flask(__name__)


@APP.route("/")
def home():
    """Render the emotion detection home page."""
    return render_template("index.html")


@APP.route("/emotionDetector", methods=["GET"])
def detect_emotion():
    """Analyze submitted text and return a human-readable result."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    result = emotion_detector(text_to_analyze)

    if result["dominant_emotion"] is None:
        return "Invalid text! Please try again!"

    scores = ", ".join(
        f"'{emotion}': {result[emotion]}"
        for emotion in ("anger", "disgust", "fear", "joy", "sadness")
    )
    return (
        f"For the given statement, the system response is {scores}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


if __name__ == "__main__":
    APP.run(host=HOST, port=PORT, debug=False)
