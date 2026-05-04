import streamlit as st
import numpy as np
import os
import tempfile
from pathlib import Path

try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error(
        "Không tìm thấy TensorFlow trong Python hiện tại. "
        "Hãy dùng Python 3.13 và cài dependencies bằng `py -3.13 -m pip install -r requirements.txt`."
    )
    st.stop()

# Import đúng từ utils
from utils.feature_extraction import extract_features
from utils.preprocessing import validate_input_shape, get_audio_duration, extract_mfcc_only
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue

# Queue lưu audio frames từ mic để xử lý
_audio_q = queue.Queue()


def audio_frame_callback(frame: av.AudioFrame) -> None:
    # Chuyển frame thành numpy array mono
    pcm = frame.to_ndarray().astype('float32')
    # Nếu stereo, lấy kênh đầu
    if pcm.ndim > 1:
        pcm = pcm[0]
    _audio_q.put(pcm)

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
    st.markdown("---")
    feature_mode = st.radio("Chế độ trích xuất đặc trưng", ("improved", "mfcc-only"), index=0)
    # Debug: show model info and a test prediction
    if st.checkbox("Hiện thông tin model (debug)"):
        import io, sys
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        st.text_area("Model summary", buf.getvalue(), height=300)
        last = model.layers[-1]
        st.write('Last layer name:', getattr(last, 'name', None))
        try:
            st.write('Last layer config:', last.get_config())
        except Exception as e:
            st.write('Could not get last layer config:', e)
        # random prediction
        import numpy as _np
        x = _np.random.randn(1,220,40).astype('float32')
        pred = model.predict(x)
        st.write('Random pred (probs):', [float(p) for p in pred[0]])

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
                    # Chọn extractor theo cài đặt
                    if feature_mode == "improved":
                        features = extract_features(temp_path)
                    else:
                        features = extract_mfcc_only(temp_path)

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

                            # Biểu đồ & debug
                            st.subheader("📊 Xác suất các cảm xúc")
                            prob_dict = {(EMOTIONS_VI[i] if use_vietnamese else EMOTIONS[i]): float(prediction[0][i]*100)
                                        for i in range(4)}
                            st.bar_chart(prob_dict)
                            st.markdown("**Raw probabilities:**")
                            st.write([float(p) for p in prediction[0]])

                except Exception as e:
                    st.error(f"Lỗi dự đoán: {e}")
                finally:
                    # Dọn dẹp
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

with col2:
    st.subheader("📋 Hướng dẫn")
    st.markdown("---")
    st.subheader("🎙️ Hoặc thu âm trực tiếp từ mic")

    webrtc_ctx = webrtc_streamer(
        key="speech-emotion-mic",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_frame_callback=audio_frame_callback,
    )

    if st.button("🔴 Ghi 3s và dự đoán từ mic"):
        st.info("Đang ghi 3 giây từ mic...")
        import time
        frames = []
        start = time.time()
        while time.time() - start < 3.0:
            try:
                pcm = _audio_q.get(timeout=1.0)
                frames.append(pcm)
            except queue.Empty:
                break

        if len(frames) == 0:
            st.error("Không ghi được âm thanh từ mic. Kiểm tra quyền truy cập microphone.")
        else:
            audio = np.concatenate(frames)
            # Lưu tạm file wav
            import soundfile as sf, tempfile
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(tmpf.name, audio, 22050)

            with st.spinner("Đang trích xuất đặc trưng và dự đoán..."):
                features = extract_features(tmpf.name)
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

            os.unlink(tmpf.name)
    st.markdown("""
    • File âm thanh nên rõ ràng, có giọng nói  
    • Độ dài tốt nhất: 2 - 5 giây  
    • Môi trường yên tĩnh → kết quả chính xác hơn
    """)

st.caption("Speech Emotion Recognition | CNN + BiLSTM | Keras 3")