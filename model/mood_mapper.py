def map_mood(emotion_data):
    label = emotion_data['label']
    mood = {
        "joy": {"label": "Uplifted", "emoji": "😊", "context": "Positive vibes detected"},
        "sadness": {"label": "Melancholy", "emoji": "😢", "context": "A tender heaviness in tone"},
        "anger": {"label": "Agitated", "emoji": "😠", "context": "Irritation or conflict brewing"},
        "surprise": {"label": "Alert", "emoji": "😲", "context": "Unexpected emotional spike"},
        "fear": {"label": "Tense", "emoji": "😨", "context": "Signs of unease or concern"},
        "neutral": {"label": "Stable", "emoji": "😐", "context": "Emotionally consistent"},
        "disgust": {"label": "Displeased", "emoji": "🤢", "context": "Aversion or repulsion noticed"},
        "love": {"label": "Connected", "emoji": "💖", "context": "Affection in expression"}
    }
    return mood.get(label.lower(), {"label": "Undefined", "emoji": "🤖", "context": "Emotion could not be parsed"})
