"""Unit tests for the Watson NLP emotion detection application.

Run these tests in the Skills Network Theia Lab, where the Watson NLP
endpoint is available.
"""

import unittest

from EmotionDetection import emotion_detector


class EmotionDetectionTests(unittest.TestCase):
    """Verify the dominant emotion for the five required sample statements."""

    def test_joy(self):
        """A glad statement should be classified as joy."""
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger(self):
        """An angry statement should be classified as anger."""
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust(self):
        """A disgusted statement should be classified as disgust."""
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness(self):
        """A sad statement should be classified as sadness."""
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear(self):
        """A fearful statement should be classified as fear."""
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
