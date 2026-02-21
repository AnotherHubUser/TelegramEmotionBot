# import numpy as np
import gigaam
from logger import get_logger

logger = get_logger(__name__)

class EmotionClassifier:
    def __init__(self):
        logger.info("start loading model")
        self.model = gigaam.load_model('emo')
        logger.info("model loaded")
    
    def predict(self, audio_path):
        logger.info(f"try to predict best emotion by path {audio_path}")
        emotion2prob = self.model.get_probs(audio_path)
        top_emotion = max(emotion2prob, key=emotion2prob.get)
        return top_emotion
    
    def predict_with_scores(self, audio_path):
        logger.info(f"try to predict all emotions by path {audio_path}")
        return self.model.get_probs(audio_path)
    

# def predict(filepath: str):
#     emotions = ["fury", "sad", "happy", "excited"]
#     return np.random.choice(emotions)