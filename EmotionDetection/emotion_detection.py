"""Call the Skills Network Watson NLP emotion prediction service."""

import json

import requests

EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
EMOTION_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTION_KEYS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion for the supplied text.

    A blank input is passed to the service so its HTTP 400 response can be
    handled consistently with the Skills Network project requirements.
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    try:
        response = requests.post(
            EMOTION_URL,
            headers=EMOTION_HEADERS,
            json=payload,
            timeout=30,
        )
    except requests.RequestException:
        return {key: None for key in (*EMOTION_KEYS, "dominant_emotion")}

    if response.status_code == 400:
        return {key: None for key in (*EMOTION_KEYS, "dominant_emotion")}

    response.raise_for_status()
    response_data = json.loads(response.text)
    emotion_scores = response_data["emotionPredictions"][0]["emotion"]
    result = {key: emotion_scores[key] for key in EMOTION_KEYS}
    result["dominant_emotion"] = max(
        EMOTION_KEYS, key=lambda emotion: result[emotion]
    )
    return result
