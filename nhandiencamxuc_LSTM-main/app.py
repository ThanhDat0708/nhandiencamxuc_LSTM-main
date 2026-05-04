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
from utils.preprocessing import validate_input_shape, extract_mfcc_only
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

# Mô hình này chỉ dùng 4 lớp cảm xúc
EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]
EMOTIONS_VI = ["Tức giận", "Vui vẻ", "Buồn", "Trung lập"]
st.markdown(
    "**Model**: CNN + Bidirectional LSTM | 4 lớp: Angry, Happy, Sad, Neutral"
)


def predict_audio_file(audio_path: str, use_trimmed_features: bool = True):
    """Dự đoán từ một file âm thanh bằng 1 hoặc 2 biến thể feature để giảm sai lệch."""
    candidates = []

    feature_extractors = []
    if use_trimmed_features:
        feature_extractors.append(("trimmed", extract_features))
        feature_extractors.append(("raw", extract_mfcc_only))
    else:
        feature_extractors.append(("raw", extract_mfcc_only))

    for name, extractor in feature_extractors:
        features = extractor(audio_path)
        if features is None:
            continue

        is_valid, msg = validate_input_shape(features)
        if not is_valid:
            continue

        prediction = model.predict(features, verbose=0)[0]
        confidence = float(np.max(prediction))
        predicted_idx = int(np.argmax(prediction))
        candidates.append(
            {
                "name": name,
                "prediction": prediction,
                "predicted_idx": predicted_idx,
                "confidence": confidence,
            }
        )

    if not candidates:
        return None, None, None, None

    # Chọn biến thể có độ tự tin cao nhất; nếu 2 biến thể gần nhau thì lấy trung bình.
    if len(candidates) >= 2:
        sorted_candidates = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
        top1 = sorted_candidates[0]
        top2 = sorted_candidates[1]
        if abs(top1["confidence"] - top2["confidence"]) < 0.08:
            avg_prediction = np.mean([item["prediction"] for item in candidates], axis=0)
            predicted_idx = int(np.argmax(avg_prediction))
            confidence = float(np.max(avg_prediction))
            return avg_prediction, predicted_idx, confidence, "ensemble"

        return top1["prediction"], top1["predicted_idx"], top1["confidence"], top1["name"]

    only = candidates[0]
    return only["prediction"], only["predicted_idx"], only["confidence"], only["name"]

# ========================= LOAD MODEL =========================
@st.cache_resource(show_spinner="Đang tải model...")
def load_model():
    model_candidates = [
        "model/speech_emotion_lstm_4classes.keras",
        "model/speech_emotion_lstm_improved.keras",
    ]

    for model_path in model_candidates:
        if not os.path.exists(model_path):
            continue

        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            st.success(f"✅ Model loaded successfully: {model_path}")
            return model
        except Exception as e:
            st.warning(f"Không load được {model_path}: {e}")

    st.error("❌ Không tìm thấy model hợp lệ.")
    st.stop()

model = load_model()

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    use_vietnamese = st.checkbox("Hiển thị bằng tiếng Việt", value=True)
    st.markdown("---")
    feature_mode = st.radio("Chế độ trích xuất đặc trưng", ("raw", "trimmed", "ensemble"), index=0)
    st.caption("Phân loại 4 lớp: Angry / Happy / Sad / Neutral")
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
                    use_trimmed = feature_mode in ("trimmed", "ensemble")

                    prediction, predicted_idx, confidence, chosen_mode = predict_audio_file(
                        temp_path,
                        use_trimmed_features=use_trimmed,
                    )

                    if prediction is None:
                        st.error("Không trích xuất được đặc trưng hợp lệ từ file âm thanh.")
                    else:
                        emotion = EMOTIONS_VI[predicted_idx] if use_vietnamese else EMOTIONS[predicted_idx]

                        st.success(f"**Cảm xúc dự đoán: {emotion}**")
                        st.metric("Độ tin cậy", f"{confidence * 100:.1f}%")
                        st.caption(f"Chế độ dùng: {chosen_mode}")

                        # Biểu đồ & debug
                        st.subheader("📊 Xác suất các cảm xúc")
                        prob_dict = {(EMOTIONS_VI[i] if use_vietnamese else EMOTIONS[i]): float(prediction[i] * 100)
                                    for i in range(4)}
                        st.bar_chart(prob_dict)
                        st.markdown("**Raw probabilities:**")
                        st.write([float(p) for p in prediction])

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
                use_trimmed = feature_mode in ("trimmed", "ensemble")

                prediction, predicted_idx, confidence, chosen_mode = predict_audio_file(
                    tmpf.name,
                    use_trimmed_features=use_trimmed,
                )
                if prediction is None:
                    st.error("Không trích xuất được đặc trưng hợp lệ từ mic.")
                else:
                    emotion = EMOTIONS_VI[predicted_idx] if use_vietnamese else EMOTIONS[predicted_idx]

                    st.success(f"**Cảm xúc dự đoán: {emotion}**")
                    st.metric("Độ tin cậy", f"{confidence * 100:.1f}%")
                    st.caption(f"Chế độ dùng: {chosen_mode}")

            os.unlink(tmpf.name)
    st.markdown("""
    • File âm thanh nên rõ ràng, có giọng nói  
    • Độ dài tốt nhất: 2 - 5 giây  
    • Môi trường yên tĩnh → kết quả chính xác hơn
    """)

st.caption("Speech Emotion Recognition | CNN + BiLSTM | Keras 3")