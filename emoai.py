import argparse
from model.emotion_classifier import classify_emotion
from model.mood_mapper import map_mood
from model.context_memory import update_mood_memory
from utils.audio_text_fusion import transcribe_audio, fuse_modalities
import json

def run_text_pipeline(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        content = file.read()
    emotion = classify_emotion(content)
    mood = map_mood(emotion)
    emoji = mood.get("emoji", "🙂")
    update_mood_memory(mood)
    print(f"[Emotion] {emotion} | [Mood] {mood['label']} {emoji}")
    _save_output(mood, emoji)

def run_audio_pipeline(audio_file):
    text = transcribe_audio(audio_file)
    fused_input = fuse_modalities(text, audio_file)
    emotion = classify_emotion(fused_input)
    mood = map_mood(emotion)
    emoji = mood.get("emoji", "🎵")
    update_mood_memory(mood)
    print(f"[Fused Emotion] {emotion} | [Mood] {mood['label']} {emoji}")
    _save_output(mood, emoji)

def _save_output(mood, emoji):
    with open('output/mood_scores.json', 'w') as mfile:
        json.dump(mood, mfile, indent=4)
    with open('output/emoji_suggestions.txt', 'w') as efile:
        efile.write(f"{emoji} — {mood['context']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Path to text file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    args = parser.parse_args()

    if args.text:
        run_text_pipeline(args.text)
    elif args.audio:
        run_audio_pipeline(args.audio)
    else:
        print("⚠️ Provide either --text or --audio input.")
