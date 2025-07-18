"""
Audio-Text Fusion Module for EmoAI
Combines speech recognition with text processing for enhanced emotion detection.
"""

import whisper
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTextFusion:
    """Enhanced audio-text fusion with advanced speech analysis."""
    
    def __init__(self, model_size="base"):
        """
        Initialize the AudioTextFusion module.
        
        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model with error handling."""
        try:
            self.model = whisper.load_model(self.model_size)
            logger.info(f"Whisper model '{self.model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file to text with enhanced features.
        
        Args:
            file_path (str): Path to audio file
            language (str, optional): Language code for transcription
            
        Returns:
            Dict containing transcription results and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            logger.info(f"Transcribing audio file: {file_path}")
            result = self.model.transcribe(
                file_path,
                language=language,
                word_timestamps=True,
                temperature=0.2
            )
            
            # Extract additional metadata
            transcription_data = {
                'text': result['text'].strip(),
                'language': result['language'],
                'segments': result.get('segments', []),
                'confidence': self._calculate_confidence(result),
                'duration': self._get_audio_duration(result),
                'word_count': len(result['text'].split()),
                'speaking_rate': self._calculate_speaking_rate(result)
            }
            
            logger.info(f"Transcription completed. Language: {transcription_data['language']}")
            return transcription_data
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': [],
                'confidence': 0.0,
                'duration': 0.0,
                'word_count': 0,
                'speaking_rate': 0.0,
                'error': str(e)
            }
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate average confidence score from segments."""
        if 'segments' not in result or not result['segments']:
            return 0.0
        
        confidences = []
        for segment in result['segments']:
            if 'words' in segment:
                for word in segment['words']:
                    if 'probability' in word:
                        confidences.append(word['probability'])
        
        return np.mean(confidences) if confidences else 0.0
    
    def _get_audio_duration(self, result: Dict) -> float:
        """Get audio duration from transcription result."""
        if 'segments' not in result or not result['segments']:
            return 0.0
        
        last_segment = result['segments'][-1]
        return last_segment.get('end', 0.0)
    
    def _calculate_speaking_rate(self, result: Dict) -> float:
        """Calculate words per minute speaking rate."""
        duration = self._get_audio_duration(result)
        if duration == 0:
            return 0.0
        
        word_count = len(result['text'].split())
        return (word_count / duration) * 60  # words per minute
    
    def analyze_audio_features(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze audio features for enhanced emotion detection.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            Dict containing audio analysis features
        """
        try:
            # Load audio using whisper's audio loading
            audio = whisper.load_audio(file_path)
            
            # Basic audio analysis
            features = {
                'sample_rate': 16000,  # Whisper uses 16kHz
                'duration': len(audio) / 16000,
                'amplitude_mean': float(np.mean(np.abs(audio))),
                'amplitude_std': float(np.std(np.abs(audio))),
                'energy': float(np.sum(audio ** 2)),
                'zero_crossing_rate': self._calculate_zcr(audio),
                'spectral_centroid': self._calculate_spectral_centroid(audio)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Audio feature analysis failed: {e}")
            return {}
    
    def _calculate_zcr(self, audio: np.ndarray) -> float:
        """Calculate zero crossing rate."""
        signs = np.sign(audio)
        zero_crossings = np.diff(signs)
        return float(np.sum(zero_crossings != 0) / len(audio))
    
    def _calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid (brightness measure)."""
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/16000)
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        return float(np.sum(freqs * magnitude) / np.sum(magnitude))
    
    def fuse_modalities(self, text: str, audio_path: str, 
                       fusion_strategy: str = "concatenate") -> Dict[str, Any]:
        """
        Fuse text and audio modalities for enhanced emotion detection.
        
        Args:
            text (str): Input text
            audio_path (str): Path to audio file
            fusion_strategy (str): Strategy for fusion ('concatenate', 'weighted', 'contextual')
            
        Returns:
            Dict containing fused data and metadata
        """
        transcription_data = self.transcribe_audio(audio_path)
        audio_features = self.analyze_audio_features(audio_path)
        
        if fusion_strategy == "concatenate":
            fused_text = self._concatenate_fusion(text, transcription_data['text'])
        elif fusion_strategy == "weighted":
            fused_text = self._weighted_fusion(text, transcription_data, audio_features)
        elif fusion_strategy == "contextual":
            fused_text = self._contextual_fusion(text, transcription_data, audio_features)
        else:
            fused_text = self._concatenate_fusion(text, transcription_data['text'])
        
        return {
            'fused_text': fused_text,
            'original_text': text,
            'transcription': transcription_data,
            'audio_features': audio_features,
            'fusion_strategy': fusion_strategy,
            'confidence_score': transcription_data.get('confidence', 0.0)
        }
    
    def _concatenate_fusion(self, text: str, transcription: str) -> str:
        """Simple concatenation fusion strategy."""
        return f"{text.strip()} {transcription.strip()}".strip()
    
    def _weighted_fusion(self, text: str, transcription_data: Dict, 
                        audio_features: Dict) -> str:
        """Weighted fusion based on confidence and audio quality."""
        confidence = transcription_data.get('confidence', 0.0)
        
        # Weight based on transcription confidence
        if confidence > 0.8:
            weight = 0.7  # High confidence, favor transcription
        elif confidence > 0.5:
            weight = 0.5  # Medium confidence, equal weight
        else:
            weight = 0.3  # Low confidence, favor original text
        
        # Create weighted representation
        if weight > 0.5:
            return f"{text.strip()} [AUDIO_EMPHASIS] {transcription_data['text'].strip()}"
        else:
            return f"{text.strip()} [AUDIO_CONTEXT] {transcription_data['text'].strip()}"
    
    def _contextual_fusion(self, text: str, transcription_data: Dict, 
                          audio_features: Dict) -> str:
        """Contextual fusion with audio feature integration."""
        transcription = transcription_data['text']
        
        # Add context based on audio features
        context_markers = []
        
        # Speaking rate context
        speaking_rate = transcription_data.get('speaking_rate', 0)
        if speaking_rate > 180:
            context_markers.append("[FAST_SPEECH]")
        elif speaking_rate < 120:
            context_markers.append("[SLOW_SPEECH]")
        
        # Energy context
        energy = audio_features.get('energy', 0)
        if energy > 1000:
            context_markers.append("[HIGH_ENERGY]")
        elif energy < 100:
            context_markers.append("[LOW_ENERGY]")
        
        # Combine with context
        context_str = " ".join(context_markers)
        return f"{text.strip()} {context_str} {transcription.strip()}".strip()

# Global instance for backward compatibility
_fusion_instance = None

def get_fusion_instance(model_size="base"):
    """Get or create global fusion instance."""
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = AudioTextFusion(model_size)
    return _fusion_instance

# Backward compatible functions
def transcribe_audio(file_path: str, language: Optional[str] = None) -> str:
    """
    Transcribe audio file to text (backward compatible).
    
    Args:
        file_path (str): Path to audio file
        language (str, optional): Language code
        
    Returns:
        str: Transcribed text
    """
    fusion = get_fusion_instance()
    result = fusion.transcribe_audio(file_path, language)
    return result.get('text', '')

def fuse_modalities(text: str, audio_path: str, 
                   fusion_strategy: str = "concatenate") -> str:
    """
    Fuse text and audio modalities (backward compatible).
    
    Args:
        text (str): Input text
        audio_path (str): Path to audio file
        fusion_strategy (str): Fusion strategy
        
    Returns:
        str: Fused text
    """
    fusion = get_fusion_instance()
    result = fusion.fuse_modalities(text, audio_path, fusion_strategy)
    return result.get('fused_text', f"{text.strip()} {transcribe_audio(audio_path)}")
