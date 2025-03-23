import streamlit as st
from audio_recorder_streamlit import audio_recorder
import scipy.io.wavfile as wav
import speech_recognition as sr
from transformers import pipeline
import torch

# Page settings
st.set_page_config(page_title="🎤 Voice Sentiment Analysis", layout="centered")
st.title("🎤 Voice Sentiment Analysis")
st.markdown("Welcome! Click the microphone icon below to record your speech and wait for the analysis to complete.")

# Record audio from browser
audio_bytes = audio_recorder(text="🎙 Click to record", icon_size="2x")

if audio_bytes:
    filename = "recorded_audio.wav"
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    st.success("✅ Audio successfully recorded!")

    # Transcribe audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language="en")
        st.markdown("**Transcription:**")
        st.code(text)
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        text = ""

    # Sentiment analysis
    if text:
        sentiment_pipeline = pipeline("sentiment-analysis")
        sentiment = sentiment_pipeline(text)

        label = sentiment[0]['label'].lower()
        score = sentiment[0]['score']
        score_percent = f"{score * 100:.2f}%"

        if label == "positive":
            color = "#d4edda"
            emoji = "😊"
            label_text = "Positive"
        elif label == "negative":
            color = "#f8d7da"
            emoji = "😠"
            label_text = "Negative"
        else:
            color = "#d1ecf1"
            emoji = "😐"
            label_text = "Neutral"

        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; border: 1px solid #ccc;">
                <h4 style="margin-bottom: 10px;">🔍 Sentiment Analysis Result</h4>
                <p style="font-size: 18px;"><strong>Sentiment:</strong> {label_text} {emoji}</p>
                <p style="font-size: 18px;"><strong>Confidence Score:</strong> {score_percent}</p>
            </div>
            """,
            unsafe_allow_html=True
        )






