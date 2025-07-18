"""
Enhanced Emotion Classifier for EmoAI
Advanced emotion detection with context awareness and confidence scoring.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionClassifier:
    """Enhanced emotion classifier with multiple models and context awareness."""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion classifier.
        
        Args:
            model_name (str): Hugging Face model name
        """
        self.model_name = model_name
        self.classifier = None
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the emotion classification model."""
        try:
            # Load pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load tokenizer and model for advanced features
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            logger.info(f"Emotion classifier loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load emotion classifier: {e}")
            raise
    
    def classify_emotion(self, text: str, context: Optional[Dict] = None, 
                        confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Classify emotion in text with context awareness.
        
        Args:
            text (str): Input text
            context (Dict, optional): Previous emotional context
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            Dict: Emotion classification results
        """
        if not text or not text.strip():
            return self._get_default_result()
        
        try:
            # Get raw predictions
            scores = self.classifier(text)[0]
            
            # Process results
            emotions = {s['label']: round(s['score'], 4) for s in scores}
            top_emotion = max(scores, key=lambda x: x['score'])
            
            # Apply context if available
            if context:
                top_emotion, emotions = self._apply_context(top_emotion, emotions, context)
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(emotions)
            
            # Prepare result
            result = {
                "label": top_emotion['label'],
                "score": round(top_emotion['score'], 4),
                "emotions": emotions,
                "confidence_metrics": confidence_metrics,
                "text_length": len(text),
                "word_count": len(text.split()),
                "context_applied": context is not None,
                "high_confidence": top_emotion['score'] > confidence_threshold
            }
            
            # Add text analysis
            result.update(self._analyze_text_features(text))
            
            return result
            
        except Exception as e:
            logger.error(f"Emotion classification failed: {e}")
            return self._get_default_result(error=str(e))
    
    def _apply_context(self, top_emotion: Dict, emotions: Dict, 
                      context: Dict) -> tuple:
        """Apply emotional context to adjust predictions."""
        try:
            if 'mood' in context:
                prev_mood = context['mood']
                prev_label = prev_mood.get('label', '').lower()
                
                # Context-based adjustment
                if prev_label in emotions:
                    # Boost previous emotion slightly for consistency
                    emotions[prev_label] = min(1.0, emotions[prev_label] * 1.1)
                    
                    # Recalculate top emotion
                    top_label = max(emotions, key=emotions.get)
                    top_emotion = {
                        'label': top_label,
                        'score': emotions[top_label]
                    }
            
            return top_emotion, emotions
            
        except Exception as e:
            logger.error(f"Context application failed: {e}")
            return top_emotion, emotions
    
    def _calculate_confidence_metrics(self, emotions: Dict) -> Dict[str, float]:
        """Calculate confidence metrics for the prediction."""
        scores = list(emotions.values())
        
        return {
            "max_score": max(scores),
            "min_score": min(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "entropy": self._calculate_entropy(scores),
            "confidence_gap": max(scores) - sorted(scores)[-2] if len(scores) > 1 else 0
        }
    
    def _calculate_entropy(self, scores: List[float]) -> float:
        """Calculate entropy of emotion scores."""
        # Normalize scores to probabilities
        scores = np.array(scores)
        scores = scores / np.sum(scores)
        
        # Calculate entropy
        entropy = -np.sum(scores * np.log2(scores + 1e-10))
        return float(entropy)
    
    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """Analyze text features for additional insights."""
        words = text.split()
        
        # Basic text analysis
        features = {
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "sentence_count": len([s for s in text.split('.') if s.strip()]),
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "punctuation_density": sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        }
        
        # Emotional indicators
        emotional_words = {
            'positive': ['good', 'great', 'amazing', 'wonderful', 'fantastic', 'love', 'happy', 'joy'],
            'negative': ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'disappointed'],
            'intense': ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely']
        }
        
        text_lower = text.lower()
        for category, words_list in emotional_words.items():
            count = sum(1 for word in words_list if word in text_lower)
            features[f'{category}_word_count'] = count
        
        return {"text_analysis": features}
    
    def _get_default_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Get default result for error cases."""
        return {
            "label": "neutral",
            "score": 0.0,
            "emotions": {"neutral": 1.0},
            "confidence_metrics": {
                "max_score": 0.0,
                "min_score": 0.0,
                "mean_score": 0.0,
                "std_score": 0.0,
                "entropy": 0.0,
                "confidence_gap": 0.0
            },
            "text_length": 0,
            "word_count": 0,
            "context_applied": False,
            "high_confidence": False,
            "text_analysis": {},
            "error": error
        }
    
    def batch_classify(self, texts: List[str], context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Classify emotions for multiple texts.
        
        Args:
            texts (List[str]): List of texts to classify
            context (Dict, optional): Emotional context
            
        Returns:
            List[Dict]: List of emotion classification results
        """
        return [self.classify_emotion(text, context) for text in texts]
    
    def get_emotion_trends(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze emotion trends across multiple texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            Dict: Trend analysis results
        """
        results = self.batch_classify(texts)
        
        # Extract emotion labels
        emotions = [r['label'] for r in results]
        
        # Calculate trends
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate trajectory
        trajectory = self._calculate_trajectory(emotions)
        
        return {
            "emotion_distribution": emotion_counts,
            "dominant_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral",
            "emotion_variety": len(set(emotions)),
            "trajectory": trajectory,
            "total_texts": len(texts),
            "average_confidence": np.mean([r['score'] for r in results])
        }
    
    def _calculate_trajectory(self, emotions: List[str]) -> str:
        """Calculate emotional trajectory."""
        if len(emotions) < 2:
            return "stable"
        
        positive_emotions = {'joy', 'love', 'surprise'}
        negative_emotions = {'sadness', 'anger', 'fear', 'disgust'}
        
        # Calculate positive/negative trend
        first_half = emotions[:len(emotions)//2]
        second_half = emotions[len(emotions)//2:]
        
        first_positive = sum(1 for e in first_half if e in positive_emotions)
        second_positive = sum(1 for e in second_half if e in positive_emotions)
        
        if second_positive > first_positive:
            return "improving"
        elif second_positive < first_positive:
            return "declining"
        else:
            return "stable"

# Global instance
_classifier_instance = None

def get_classifier_instance() -> EmotionClassifier:
    """Get or create global classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EmotionClassifier()
    return _classifier_instance

# Backward compatible function
def classify_emotion(text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Classify emotion in text (backward compatible).
    
    Args:
        text (str): Input text
        context (Dict, optional): Previous emotional context
        
    Returns:
        Dict: Emotion classification results
    """
    classifier = get_classifier_instance()
    return classifier.classify_emotion(text, context)
