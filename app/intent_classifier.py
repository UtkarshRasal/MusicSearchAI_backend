from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from typing import Dict

# Emotion → mood filter mapping
EMOTION_TO_FILTER = {
    "sadness": {"max_valence": 0.4, "max_energy": 0.6},
    "joy": {"min_valence": 0.6, "min_energy": 0.6},
    "love": {"min_valence": 0.6},
    "anger": {"min_energy": 0.7, "max_valence": 0.5},
    "fear": {"max_valence": 0.4, "max_energy": 0.5},
    "surprise": {},  # Optional: no mood filter
}

class EmotionClassifier:
    _instance = None

    def __init__(self):
        model_name = "nateraw/bert-base-uncased-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = list(EMOTION_TO_FILTER.keys())

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EmotionClassifier()
        return cls._instance

    def classify(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            top_idx = torch.argmax(probs, dim=1).item()
            return self.labels[top_idx]

def get_emotion_filter(query: str) -> Dict:
    """
        Predicts the emotion of query and returns mood filter
    """

    try:
        classifier = EmotionClassifier.get_instance()
        emotion = classifier.classify(query)
        print(f"[Intent] Emotion detected: {emotion}")
        return EMOTION_TO_FILTER.get(emotion, {})

    except Exception as e:
        raise Exception(f"Failed to classify intent - {e}")
