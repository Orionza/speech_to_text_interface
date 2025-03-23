import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import tempfile

# Sayfa ayarları
st.set_page_config(page_title="🎤 Voice Sentiment Analysis", layout="centered")
st.title("🎤 Voice Sentiment Analysis")
st.markdown("Welcome! Click the microphone icon below to record your speech and wait for the analysis to complete.")

# Ses kaydı
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=44100)

if audio_bytes:
    st.success("✅ Audio received!")

    # Ses dosyasını geçici olarak kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Transkripsiyon
    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        st.markdown("### 📄 Transcription")
        st.code(text)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        text = ""

    # Duygu analizi (CardiffNLP modeli)
    if text:
        with st.spinner("Analyzing sentiment..."):
            model_name = "cardiffnlp/twitter-roberta-base-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

            result = sentiment_pipeline(text)[0]
            label = result["label"].lower()
            score = result["score"] * 100

            # Renkler ve emojiler
            if label == "positive":
                bg_color = "#0b3c2e"  # koyu yeşil
                emoji = "😊"
            elif label == "negative":
                bg_color = "#4b1c1c"  # koyu kırmızı
                emoji = "😠"
            else:  # neutral
                bg_color = "#1a2f4f"  # koyu mavi
                emoji = "😐"

            # Sonucu göster
            st.markdown("### 💬 Sentiment Analysis")
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; color: white; padding: 20px; border-radius: 10px;">
                    <p style="font-size: 18px;"><strong>Sentiment:</strong> {label.upper()} {emoji}</p>
                    <p><strong>Confidence:</strong> {score:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )












