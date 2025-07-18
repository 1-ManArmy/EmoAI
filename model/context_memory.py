"""
Context Memory Module for EmoAI
Manages emotional context and memory across sessions with advanced analytics.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEMORY_FILE = "output/memory_log.json"
ANALYTICS_FILE = "output/emotion_analytics.json"

class ContextMemory:
    """Advanced context memory with analytics and pattern recognition."""
    
    def __init__(self, memory_file: str = MEMORY_FILE, max_memory: int = 100):
        """
        Initialize context memory.
        
        Args:
            memory_file (str): Path to memory file
            max_memory (int): Maximum number of memories to keep
        """
        self.memory_file = memory_file
        self.analytics_file = ANALYTICS_FILE
        self.max_memory = max_memory
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
    
    def load_memory(self) -> List[Dict]:
        """Load memory from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
                return []
        return []
    
    def save_memory(self, memory: List[Dict]):
        """Save memory to file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def update_mood_memory(self, current_mood: Dict[str, Any], 
                          context: Optional[Dict] = None):
        """
        Update mood memory with enhanced context.
        
        Args:
            current_mood (Dict): Current mood data
            context (Dict, optional): Additional context information
        """
        memory = self.load_memory()
        
        # Create enhanced memory entry
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'mood': current_mood,
            'context': context or {},
            'session_id': self._get_session_id(),
            'sequence_number': len(memory) + 1
        }
        
        memory.append(memory_entry)
        
        # Keep only recent memories
        if len(memory) > self.max_memory:
            memory = memory[-self.max_memory:]
        
        self.save_memory(memory)
        self._update_analytics(memory_entry)
        
        logger.info(f"Updated mood memory: {current_mood.get('label', 'unknown')}")
    
    def recall_last_mood(self, n: int = 1) -> Optional[Dict]:
        """
        Recall the last N mood entries.
        
        Args:
            n (int): Number of recent moods to recall
            
        Returns:
            Dict or List: Last mood(s) or None if no memory
        """
        memory = self.load_memory()
        if not memory:
            return None
        
        if n == 1:
            return memory[-1]
        else:
            return memory[-n:] if len(memory) >= n else memory
    
    def get_mood_pattern(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze mood patterns over the specified time period.
        
        Args:
            hours (int): Number of hours to analyze
            
        Returns:
            Dict: Pattern analysis results
        """
        memory = self.load_memory()
        if not memory:
            return {'pattern': 'insufficient_data', 'moods': []}
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_memory = []
        
        for entry in memory:
            try:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_time:
                    recent_memory.append(entry)
            except:
                continue
        
        if not recent_memory:
            return {'pattern': 'no_recent_data', 'moods': []}
        
        # Analyze patterns
        moods = [entry['mood']['label'] for entry in recent_memory]
        mood_counts = {}
        for mood in moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        # Detect patterns
        pattern = self._detect_pattern(moods)
        
        return {
            'pattern': pattern,
            'moods': moods,
            'mood_distribution': mood_counts,
            'dominant_mood': max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else None,
            'mood_changes': len(set(moods)),
            'stability_score': self._calculate_stability(moods),
            'time_period_hours': hours,
            'total_entries': len(recent_memory)
        }
    
    def _detect_pattern(self, moods: List[str]) -> str:
        """Detect mood pattern from sequence."""
        if len(moods) < 2:
            return 'insufficient_data'
        
        # Check for stability
        if len(set(moods)) == 1:
            return 'stable'
        
        # Check for improvement/decline
        positive_moods = {'joy', 'love', 'surprise'}
        negative_moods = {'sadness', 'anger', 'fear', 'disgust'}
        
        recent_positive = sum(1 for mood in moods[-3:] if mood.lower() in positive_moods)
        recent_negative = sum(1 for mood in moods[-3:] if mood.lower() in negative_moods)
        
        if recent_positive > recent_negative:
            return 'improving'
        elif recent_negative > recent_positive:
            return 'declining'
        else:
            return 'fluctuating'
    
    def _calculate_stability(self, moods: List[str]) -> float:
        """Calculate mood stability score (0-1)."""
        if len(moods) < 2:
            return 1.0
        
        # Calculate how often mood changes
        changes = sum(1 for i in range(1, len(moods)) if moods[i] != moods[i-1])
        max_changes = len(moods) - 1
        
        return 1.0 - (changes / max_changes) if max_changes > 0 else 1.0
    
    def _get_session_id(self) -> str:
        """Generate or retrieve session ID."""
        # Simple session ID based on hour
        return datetime.now().strftime("%Y%m%d_%H")
    
    def _update_analytics(self, memory_entry: Dict):
        """Update emotion analytics data."""
        try:
            analytics = self._load_analytics()
            
            mood_label = memory_entry['mood'].get('label', 'unknown')
            
            # Update analytics
            analytics['total_entries'] += 1
            analytics['mood_counts'][mood_label] = analytics['mood_counts'].get(mood_label, 0) + 1
            analytics['last_updated'] = datetime.now().isoformat()
            
            # Daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in analytics['daily_stats']:
                analytics['daily_stats'][today] = {'count': 0, 'moods': {}}
            
            analytics['daily_stats'][today]['count'] += 1
            analytics['daily_stats'][today]['moods'][mood_label] = \
                analytics['daily_stats'][today]['moods'].get(mood_label, 0) + 1
            
            # Keep only last 30 days
            self._cleanup_daily_stats(analytics['daily_stats'])
            
            self._save_analytics(analytics)
            
        except Exception as e:
            logger.error(f"Failed to update analytics: {e}")
    
    def _load_analytics(self) -> Dict:
        """Load analytics data."""
        if os.path.exists(self.analytics_file):
            try:
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'total_entries': 0,
            'mood_counts': {},
            'daily_stats': {},
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_analytics(self, analytics: Dict):
        """Save analytics data."""
        try:
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save analytics: {e}")
    
    def _cleanup_daily_stats(self, daily_stats: Dict):
        """Keep only last 30 days of daily stats."""
        cutoff_date = datetime.now() - timedelta(days=30)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        keys_to_remove = []
        for date_str in daily_stats.keys():
            if date_str < cutoff_str:
                keys_to_remove.append(date_str)
        
        for key in keys_to_remove:
            del daily_stats[key]
    
    def get_analytics(self) -> Dict:
        """Get emotion analytics data."""
        return self._load_analytics()
    
    def clear_memory(self):
        """Clear all memory data."""
        try:
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
            if os.path.exists(self.analytics_file):
                os.remove(self.analytics_file)
            logger.info("Memory cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

# Global instance
_memory_instance = None

def get_memory_instance() -> ContextMemory:
    """Get or create global memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ContextMemory()
    return _memory_instance

# Backward compatible functions
def update_mood_memory(current_mood: Dict[str, Any], context: Optional[Dict] = None):
    """Update mood memory (backward compatible)."""
    memory = get_memory_instance()
    memory.update_mood_memory(current_mood, context)

def recall_last_mood(n: int = 1) -> Optional[Dict]:
    """Recall last mood (backward compatible)."""
    memory = get_memory_instance()
    return memory.recall_last_mood(n)
