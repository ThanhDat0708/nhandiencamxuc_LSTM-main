import streamlit as st
import tensorflow as tf
import numpy as np
import os
import tempfile
from pathlib import Path

# Import đúng từ utils
from utils.feature_extraction import extract_features
from utils.preprocessing import validate_input_shape, get_audio_duration

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Nhận Diện Cảm Xúc Qua Giọng Nói",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Hệ Thống Nhận Diện Cảm Xúc Qua Giọng Nói")
st.markdown("**Model**: CNN + Bidirectional LSTM | 4 cảm xúc")

EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]
EMOTIONS_VI = ["Tức giận", "Vui vẻ", "Buồn", "Trung lập"]

# ========================= LOAD MODEL =========================
@st.cache_resource(show_spinner="Đang tải model...")
def load_model():
    model_path = "model/speech_emotion_lstm_improved.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model tại: {model_path}")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        st.stop()

model = load_model()

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    use_preprocessing = st.checkbox("Sử dụng tiền xử lý âm thanh", value=True)
    use_vietnamese = st.checkbox("Hiển thị bằng tiếng Việt", value=True)

# ========================= MAIN =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Tải file âm thanh lên")
    uploaded_file = st.file_uploader("Chọn file ghi âm (.wav, .mp3, .ogg, .m4a)", 
                                   type=["wav", "mp3", "ogg", "m4a"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        st.audio(uploaded_file)

        if st.button("🔍 Phân tích cảm xúc", type="primary", use_container_width=True):
            with st.spinner("Đang trích xuất đặc trưng và dự đoán..."):
                try:
                    # Sử dụng extract_features mới (đã cải tiến)
                    features = extract_features(temp_path)

                    if features is not None:
                        is_valid, msg = validate_input_shape(features)
                        if not is_valid:
                            st.error(msg)
                        else:
                            prediction = model.predict(features, verbose=0)
                            predicted_idx = np.argmax(prediction, axis=1)[0]
                            confidence = float(np.max(prediction)) * 100

                            emotion = EMOTIONS_VI[predicted_idx] if use_vietnamese else EMOTIONS[predicted_idx]

                            st.success(f"**Cảm xúc dự đoán: {emotion}**")
                            st.metric("Độ tin cậy", f"{confidence:.1f}%")

                            # Biểu đồ
                            st.subheader("📊 Xác suất các cảm xúc")
                            prob_dict = {(EMOTIONS_VI[i] if use_vietnamese else EMOTIONS[i]): float(prediction[0][i]*100) 
                                        for i in range(4)}
                            st.bar_chart(prob_dict)

                            # Debug (xem console)
                            print("Raw probabilities:", [f"{p:.4f}" for p in prediction[0]])

                except Exception as e:
                    st.error(f"Lỗi dự đoán: {e}")
                finally:
                    # Dọn dẹp
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

with col2:
    st.subheader("📋 Hướng dẫn")
    st.markdown("""
    • File âm thanh nên rõ ràng, có giọng nói  
    • Độ dài tốt nhất: 2 - 5 giây  
    • Môi trường yên tĩnh → kết quả chính xác hơn
    """)

st.caption("Speech Emotion Recognition | CNN + BiLSTM | Keras 3")