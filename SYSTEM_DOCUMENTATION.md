# 📘 Tài Liệu Chi Tiết Hệ Thống Nhận Diện Cảm Xúc Từ Giọng Nói

## 📋 Mục Lục
1. [Tổng Quan Hệ Thống](#tổng-quan-hệ-thống)
2. [Kiến Trúc Tổng Quát](#kiến-trúc-tổng-quát)
3. [Các Thành Phần Chính](#các-thành-phần-chính)
4. [Luồng Xử Lý Dữ Liệu](#luồng-xử-lý-dữ-liệu)
5. [Thư Viện & Phụ Thuộc](#thư-viện--phụ-thuộc)
6. [Mô Hình Deep Learning](#mô-hình-deep-learning)
7. [Hướng Dẫn Cài Đặt & Chạy](#hướng-dẫn-cài-đặt--chạy)
8. [Quá Trình Huấn Luyện](#quá-trình-huấn-luyện)
9. [Sử Dụng Ứng Dụng](#sử-dụng-ứng-dụng)
10. [Xử Lý Lỗi & Gỡ Rối](#xử-lý-lỗi--gỡ-rối)

---

## 🎯 Tổng Quan Hệ Thống

### Định Nghĩa
**Hệ Thống Nhận Diện Cảm Xúc Từ Giọng Nói (Speech Emotion Recognition - SER)** là một ứng dụng sử dụng **Trí Tuệ Nhân Tạo (AI)** để phân loại cảm xúc của người nói từ các file ghi âm.

### Mục Đích
- **Phân loại 8 loại cảm xúc**: Trung lập, Bình tĩnh, Vui vẻ, Buồn, Tức giận, Sợ hãi, Ghê tởm, Ngạc nhiên
- **Xử lý file âm thanh**: Chuyển đổi tín hiệu âm thanh thành đặc trưng (features) để mô hình có thể học
- **Dự đoán xác suất**: Cung cấp mức độ tin cậy (confidence) cho mỗi dự đoán

### Ứng Dụng Thực Tế
- 📞 **Call Center**: Phát hiện khách hàng tức giận để chuyển sang nhân viên giàu kinh nghiệm
- 🎮 **Game & VR**: Thích ứng nội dung dựa trên cảm xúc người chơi
- 🏥 **Y Tế Tâm Lý**: Hỗ trợ chẩn đoán rối loạn tâm lý
- 🎬 **Phân Tích Nội Dung**: Đánh giá cảm xúc trong những bài phát biểu, phỏng vấn

---

## 🏗️ Kiến Trúc Tổng Quát

```
┌─────────────────────────────────────────────────────────────┐
│                  NGƯỜI DÙNG (Streamlit App)                 │
│                     app.py                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    ┌────────┐   ┌──────────┐   ┌──────────┐
    │ Upload │   │ Preview  │   │ Settings │
    │ File   │   │ Audio    │   │ (Mode)   │
    └───┬────┘   └──────────┘   └──────────┘
        │
        ▼
    ┌────────────────────────────────┐
    │ Xử Lý Tệp (utils/)             │
    │ • feature_extraction.py        │
    │ • preprocessing.py             │
    └───────────┬────────────────────┘
        ┌───────┴────────┐
        ▼                ▼
    ┌──────────┐   ┌──────────────┐
    │ Extract  │   │ Validate     │
    │ MFCC     │   │ Shape        │
    │ Features │   │ (1,220,40)   │
    └────┬─────┘   └──────────────┘
        │
        ▼
    ┌────────────────────────────────┐
    │ Mô Hình Deep Learning          │
    │ (CNN + Bidirectional LSTM)     │
    │ model/                         │
    │ speech_emotion_lstm_8classes   │
    │ .keras                         │
    └───────────┬────────────────────┘
        │
        ▼
    ┌────────────────────────────────┐
    │ Kết Quả Dự Đoán (Raw Probs)    │
    │ [p0, p1, p2, ..., p7]          │
    │ → Nhãn: "Vui vẻ"               │
    │ → Độ tin cậy: 92%              │
    └────────────────────────────────┘
```

---

## 🔧 Các Thành Phần Chính

### 1️⃣ **Giao Diện Web** (`app.py`)

**Mục đích**: Cung cấp giao diện người dùng qua Streamlit để upload file và xem dự đoán.

**Các tính năng chính**:

```python
# 1. Tải model từ file .keras
load_model()
  ├─ Tìm model ở: model/speech_emotion_lstm_8classes.keras
  ├─ Load model_labels từ file: speech_emotion_lstm_8classes.labels.json
  └─ Trả về: (model, CLASS_ORDER, CLASS_ORDER_VI, MODEL_PATH)

# 2. Upload file âm thanh
uploaded_file = st.file_uploader()
  ├─ Hỗ trợ: .wav, .mp3, .ogg, .m4a
  └─ Lưu tạm vào: tempfile

# 3. Trích xuất đặc trưng (3 chế độ)
predict_audio_file()
  ├─ "raw": Chỉ dùng extract_mfcc_only (không cắt khoảng lặng)
  ├─ "trimmed": extract_features (có cắt khoảng lặng)
  └─ "ensemble": Lấy trung bình từ cả 2 phương pháp

# 4. Dự đoán và hiển thị kết quả
model.predict(features)
  ├─ Input: (1, 220, 40) - 1 file, 220 frames, 40 MFCC coefficients
  └─ Output: [p0, p1, ..., p7] - xác suất cho 8 lớp
```

**Luồng chính trong app.py**:

```
Người dùng upload file
    ↓
Lưu file tạm
    ↓
Trích xuất MFCC features (220, 40)
    ↓
Kiểm tra shape hợp lệ
    ↓
Model dự đoán: argmax(prediction) → class_index
    ↓
Hiển thị: CLASS_ORDER[class_index] + confidence %
```

**Lớp Cảm Xúc (8 lớp RAVDESS)**:
```
Index  Tiếng Anh    Tiếng Việt      RAVDESS Code
  0    Neutral      Trung lập       01
  1    Calm         Bình tĩnh        02
  2    Happy        Vui vẻ           03
  3    Sad          Buồn             04
  4    Angry        Tức giận         05
  5    Fearful      Sợ hãi           06
  6    Disgust      Ghê tởm          07
  7    Surprised    Ngạc nhiên       08
```

---

### 2️⃣ **Trích Xuất Đặc Trưng** (`utils/feature_extraction.py`)

**Mục đích**: Chuyển đổi tín hiệu âm thanh thô thành các đặc trưng (features) mà mô hình có thể hiểu.

**Khái Niệm: MFCC (Mel Frequency Cepstral Coefficients)**

MFCC là một kỹ thuật xử lý âm thanh mô phỏng cách tai người cảm nhận tần số:

```
Âm thanh thô (Waveform)
    ↓
Chia thành các khung (frames) ~50ms mỗi frame
    ↓
Chuyển từ miền thời gian → miền tần số (Fourier)
    ↓
Áp dụng filter mel (mô phỏng tai người)
    ↓
Lấy logarithm của năng lượng
    ↓
Chuyển ngược lại (DCT - Discrete Cosine Transform)
    ↓
Kết quả: 40 hệ số cepstral (MFCC coefficients)
```

**Ví dụ cụ thể**:
```python
# File âm thanh: 3 giây @ 22050 Hz
# Số samples = 3 * 22050 = 66150 samples

# Chia thành frames: hop_length = 512
# Số frames = (66150 - 2048) / 512 + 1 ≈ 125 frames

# Mỗi frame → 40 MFCC coefficients
# Kết quả: (125, 40) → Chuẩn hóa thành (220, 40)
#   - Nếu < 220: padding với 0 ở cuối
#   - Nếu > 220: cắt bỏ phần cuối
```

**Hai phương pháp trong hệ thống**:

```python
1. extract_features() [Được khuyến nghị]
   ├─ Cắt khoảng lặng: librosa.effects.trim(top_db=25)
   │  (Xóa các phần yên tĩnh ở đầu/cuối)
   ├─ Lợi thế: Tập trung vào phần có giọng nói
   └─ Nhược điểm: Có thể mất thông tin nếu người nói tạm dừng

2. extract_mfcc_only() [Fallback]
   ├─ Không cắt khoảng lặng
   ├─ Lợi thế: Giữ toàn bộ tín hiệu
   └─ Nhược điểm: Khoảng lặng có thể gây nhiễu
```

**Hàm chính trong feature_extraction.py**:

```python
def _build_mfcc_features(audio_path, max_len=220, n_mfcc=40, sr=22050, trim_silence=True):
    """
    Tạo MFCC features từ file âm thanh
    
    Tham số:
    - audio_path: đường dẫn file .wav/.mp3 v.v.
    - max_len: độ dài chuẩn hóa = 220 frames
    - n_mfcc: số MFCC coefficients = 40
    - sr: sample rate (tần số lấy mẫu) = 22050 Hz
    - trim_silence: có cắt khoảng lặng hay không
    
    Trả về:
    - (220, 40) array, chuyển về float32
    """
    # 1. Load file với sample rate 22050 Hz, duration tối đa 6s
    y, _ = librosa.load(audio_path, sr=sr, duration=6.0)
    
    # 2. (Tùy chọn) Cắt khoảng lặng
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=25)
    
    # 3. Đảm bảo có đủ samples
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')
    
    # 4. Tính MFCC
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=2048,           # kích thước FFT
        hop_length=512,       # độ dịch giữa các frames
        window='hann',        # hàm cửa
    )
    
    # 5. Chuẩn hóa độ dài → (220, 40)
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]  # cắt bỏ nếu dài quá
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), 
                      mode='constant', constant_values=0)  # thêm 0 nếu quá ngắn
    
    # 6. Chuyển thành shape (220, 40) float32
    return mfcc.T.astype(np.float32)


def extract_features(audio_path, max_len=220, n_mfcc=40, sr=22050):
    """Wrapper với trim_silence=True"""
    try:
        features = _build_mfcc_features(audio_path, ..., trim_silence=True)
        return np.expand_dims(features, axis=0)  # (1, 220, 40)
    except Exception as e:
        print(f"Lỗi: {e}")
        return None
```

---

### 3️⃣ **Xác Thực Đầu Vào** (`utils/preprocessing.py`)

**Mục đích**: Kiểm tra tính hợp lệ của đặc trưng trước khi đưa vào mô hình.

```python
def validate_input_shape(features):
    """
    Kiểm tra shape của features có đúng với mô hình không
    
    Mô hình mong đợi: (1, 220, 40)
    - 1: batch size (1 file)
    - 220: số frames
    - 40: số MFCC coefficients
    """
    expected_shape = (1, 220, 40)
    if features is None:
        return False, "Features is None"
    if features.shape != expected_shape:
        return False, f"Shape không đúng. Expected: {expected_shape}, Got: {features.shape}"
    return True, "Input shape hợp lệ"


def get_audio_duration(audio_path):
    """Lấy thời lượng file âm thanh (giây) để debug"""
    y, sr = librosa.load(audio_path, sr=None, duration=1)
    return librosa.get_duration(y=y, sr=sr)
```

---

### 4️⃣ **Huấn Luyện Mô Hình** (`scripts/train_ravdess_8class.py`)

**Mục đích**: Tạo và huấn luyện mô hình trên dữ liệu RAVDESS.

**Quá trình huấn luyện chi tiết**:

```
1. TẢI DỮ LIỆU RAVDESS
   └─ Tập dữ liệu công khai, 2880 file .wav từ 24 diễn viên
   └─ 8 cảm xúc, mỗi lớp 360 file (hoặc chia không đều)

2. PHÂN TÍCH FILE
   ├─ Parse tên file theo RAVDESS format:
   │  NN-SS-TT-AA-DD-RR-EE.wav
   │  EE = emotion code (01-08)
   │
   └─ Ví dụ: 03-01-05-01-01-01-01.wav
      Actor 01, Emotion 05 = Angry

3. TRÍCH XUẤT FEATURES
   ├─ Cho mỗi file: extract_mfcc_only() → (1, 220, 40)
   ├─ Xếp thành mảng X: (2880, 220, 40)
   └─ Nhãn Y: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, ...]

4. CHIA DỮ LIỆU
   ├─ Train: 80% (2304 file)
   ├─ Test: 20% (576 file)
   └─ Shuffle: Xáo trộn để tránh bias

5. TIỀN XỬ LÝ NHÃN
   ├─ Chuyển nhãn integer → one-hot encoding
   ├─ Ví dụ: 3 → [0, 0, 0, 1, 0, 0, 0, 0]
   └─ Dùng cho categorical_crossentropy loss

6. BUILD MÔ HÌNH CNN+LSTM
   └─ (Chi tiết ở phần "Mô Hình Deep Learning" dưới)

7. HỌC MỐI QUAN HỆ PATTERNS
   ├─ Optimizer: Adam(learning_rate=1e-3)
   ├─ Loss: Categorical CrossEntropy
   ├─ Metrics: Accuracy
   ├─ Epochs: Tối đa 200, nhưng dừng sớm nếu validation không cải thiện
   ├─ Batch size: 32
   └─ Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

8. ĐÁNH GIÁ TRÊN TẬP TEST
   ├─ Classification Report: precision, recall, f1-score cho mỗi lớp
   ├─ Confusion Matrix: Ma trận nhầm lẫn
   ├─ Overall Accuracy: ~93-94% (tùy dữ liệu)
   └─ Per-class Accuracy: khác nhau theo lớp

9. LƯU LẠI MODEL & NHÃN
   ├─ Model: model/speech_emotion_lstm_8classes.keras
   ├─ Labels: model/speech_emotion_lstm_8classes.labels.json
   │  {
   │    "class_order": ["Neutral", "Calm", "Happy", ..., "Surprised"]
   │  }
   └─ Ghi chú: .labels.json để app.py load nhãn
```

**Lệnh huấn luyện**:

```bash
# Với đường dẫn RAVDESS dataset
py -3.13 scripts/train_ravdess_8class.py \
  --data-root "C:/Users/.../kagglehub/datasets/ravdess/..." \
  --output-model "model/speech_emotion_lstm_8classes.keras" \
  --seed 42
```

**Hàm chính trong train_ravdess_8class.py**:

```python
def build_dataset(data_root):
    """
    Xây dựng tập dữ liệu từ folder RAVDESS
    
    Trả về: (X, y, skipped_count)
    - X: (num_samples, 220, 40) - tất cả features
    - y: (num_samples,) - nhãn 0-7
    - skipped_count: số file bị bỏ qua (invalid)
    """
    # Duyệt tất cả file .wav
    for audio_path in data_root.rglob("*.wav"):
        # Parse RAVDESS emotion code từ tên file
        label_name = parse_ravdess_label(audio_path)
        if label_name is None:
            continue
        
        # Extract MFCC
        features = extract_mfcc_only(str(audio_path))
        if features is None:
            continue
        
        # Ghi lại
        X.append(features[0])
        y.append(CLASS_TO_INDEX[label_name])
    
    return np.asarray(X), np.asarray(y), skipped


def build_model(input_shape=(220, 40), num_classes=8):
    """
    Xây dựng mô hình CNN + Bidirectional LSTM
    Chi tiết ở phần "Mô Hình Deep Learning"
    """
    model = Sequential([...])
    model.compile(...)
    return model


def main():
    # Đọc arguments từ command line
    args = parser.parse_args()
    
    # Tải dữ liệu
    X, y, skipped = build_dataset(Path(args.data_root))
    print(f"Đã load {len(X)} samples, bỏ {skipped} file")
    
    # Kiểm tra đầy đủ 8 lớp
    validate_all_classes_present(y)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Chuyển nhãn → one-hot
    y_train_cat = to_categorical(y_train, num_classes=8)
    y_test_cat = to_categorical(y_test, num_classes=8)
    
    # Build & train
    model = build_model(input_shape=(220, 40), num_classes=8)
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=200,
        batch_size=32,
        callbacks=[EarlyStopping(...), ModelCheckpoint(...), ...]
    )
    
    # Đánh giá
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_labels))
    
    # Lưu model + nhãn
    model.save(args.output_model)
    with open(Path(args.output_model).with_suffix('.labels.json'), 'w') as f:
        json.dump({"class_order": CLASS_ORDER}, f)
```

---

## 🔄 Luồng Xử Lý Dữ Liệu

### Quá trình từ A-Z:

```
NGƯỜI DÙNG
  │
  ├─ 🔹 Tải file âm thanh lên (app.py)
  │
  ▼
STREAMLIT (app.py)
  │
  ├─ 🔹 Lưu file tạm trong /tmp/
  │
  ▼
TRÍCH XUẤT ĐẶC TRƯNG (feature_extraction.py)
  │
  ├─ 🔹 Load file với librosa: sample_rate=22050 Hz
  ├─ 🔹 (Tùy chọn) Cắt khoảng lặng
  ├─ 🔹 Tính MFCC: 40 coefficients
  ├─ 🔹 Chuẩn hóa độ dài: 220 frames
  │
  ▼
KIỂM ĐỊNH (preprocessing.py)
  │
  ├─ 🔹 Kiểm tra shape = (1, 220, 40)?
  ├─ 🔹 Kiểm tra dtype = float32?
  │
  ▼
DỰ ĐOÁN (app.py)
  │
  ├─ 🔹 Model load từ file .keras
  ├─ 🔹 model.predict(features)
  │       → tính toán từng tầng:
  │         • BatchNorm + Conv1D + MaxPool + Dropout
  │         • Bidirectional LSTM
  │         • Dense layers
  │         • Softmax activation
  │
  ▼
KẾT QUẢ (8 xác suất)
  │
  ├─ 🔹 Raw probs: [p0, p1, p2, ..., p7]
  │       ví dụ: [0.001, 0.002, 0.85, 0.05, ...]
  │
  ├─ 🔹 Lấy max: class_index = argmax = 2
  │
  ├─ 🔹 Độ tin cậy: confidence = max(probs) = 0.85 = 85%
  │
  ├─ 🔹 Nhãn: CLASS_ORDER[2] = "Happy" → "Vui vẻ"
  │
  ▼
HIỂN THỊ TRÊN UI (Streamlit)
  │
  ├─ 🔹 "Cảm xúc dự đoán: 🎉 Vui vẻ"
  ├─ 🔹 "Độ tin cậy: 85%"
  ├─ 🔹 "Chế độ dùng: trimmed"
```

---

## 📦 Thư Viện & Phụ Thuộc

Tất cả phụ thuộc được liệt kê trong `requirements.txt`:

```
streamlit
tensorflow==2.20.0
librosa
soundfile
numpy
scikit-learn
```

### Chi Tiết Từng Thư Viện:

| Thư Viện | Phiên Bản | Mục Đích | Cài Đặt |
|----------|-----------|---------|--------|
| **Streamlit** | mới nhất | Tạo giao diện web tương tác | `pip install streamlit` |
| **TensorFlow** | 2.20.0 | Framework deep learning, Keras 3 API | `pip install tensorflow==2.20.0` |
| **Librosa** | mới nhất | Xử lý tín hiệu âm thanh, tính MFCC | `pip install librosa` |
| **SoundFile** | mới nhất | Đọc/ghi file .wav | `pip install soundfile` |
| **NumPy** | mới nhất | Xử lý mảng số học | `pip install numpy` |
| **scikit-learn** | mới nhất | Chia train/test, metrics (precision, recall) | `pip install scikit-learn` |

### Tại Sao Cần Python 3.13?

- **TensorFlow 2.20.0** chỉ hỗ trợ Python 3.10 - 3.13
- Python 3.14 có thay đổi trong core APIs → TensorFlow chưa tương thích
- **Khuyến cáo**: Luôn thêm `-3.13` khi chạy lệnh:
  ```bash
  py -3.13 -m pip install ...
  py -3.13 -m streamlit run app.py
  ```

---

## 🧠 Mô Hình Deep Learning

### Kiến Trúc: CNN + Bidirectional LSTM

```
INPUT LAYER: (1, 220, 40)
├─ 1 = batch size (một file)
├─ 220 = số frames (thời gian)
└─ 40 = MFCC coefficients (đặc trưng)

    ▼
BATCH NORMALIZATION
├─ Chuẩn hóa giá trị input → trung bình 0, độ lệch chuẩn 1
└─ Lợi: Tăng tốc độ hội tụ, tránh vanishing gradient

    ▼
CONVOLUTIONAL BLOCK 1
├─ Conv1D(64 filters, kernel_size=5, padding='same')
│  └─ Trích xuất 64 đặc trưng cục bộ từ các frame liên tiếp
├─ BatchNormalization
├─ MaxPooling1D(pool_size=2)
│  └─ Giảm độ dài 220 → 110 (giữ lại các giá trị max)
└─ Dropout(0.2)
   └─ Vô hiệu hóa ngẫu nhiên 20% neuron → tránh overfitting

    ▼
CONVOLUTIONAL BLOCK 2
├─ Conv1D(128 filters, kernel_size=3, padding='same')
│  └─ Trích xuất 128 đặc trưng từ output của block 1
├─ BatchNormalization
├─ MaxPooling1D(pool_size=2)
│  └─ 110 → 55
└─ Dropout(0.25)

    ▼
BIDIRECTIONAL LSTM BLOCK 1
├─ Bidirectional LSTM(128 units, return_sequences=True)
│  ├─ Đọc chuỗi từ trái → phải (forward) + phải → trái (backward)
│  ├─ Học các mối quan hệ dài hạn trong chuỗi thời gian
│  └─ return_sequences=True: output (55, 256) - giữ lại toàn bộ timesteps
├─ BatchNormalization
└─ Dropout(0.2)

    ▼
BIDIRECTIONAL LSTM BLOCK 2
├─ Bidirectional LSTM(64 units, return_sequences=False)
│  ├─ Lớp thứ 2 LSTM, quy mô nhỏ hơn
│  └─ return_sequences=False: output (128,) - chỉ lấy output cuối
│     (128 = 64 forward + 64 backward)
└─ Dropout(0.2)

    ▼
DENSE LAYERS
├─ Dense(128, activation='relu')
│  └─ 128 neuron, hàm kích hoạt ReLU
├─ Dropout(0.4)
│  └─ Vô hiệu 40% neuron (cao hơn) → chống overfitting
│
└─ Dense(8, activation='softmax')
   ├─ 8 neuron = 8 lớp cảm xúc
   ├─ Softmax: chuyển đổi output thành xác suất (tổng = 1)
   └─ Output: [p_neutral, p_calm, p_happy, ..., p_surprised]

OUTPUT: 8 xác suất
├─ Ví dụ: [0.001, 0.002, 0.85, 0.05, 0.01, 0.009, 0.008, 0.091]
└─ argmax = 2 → "Happy"
```

### Tại Sao Kiến Trúc Này?

**CNN (Convolutional Neural Networks)**:
- 🔹 **Điểm mạnh**: Tìm các mẫu cục bộ trong tín hiệu (ví dụ: tiếng "s" hay "p")
- 🔹 **Lợi thế**: Chia sẻ trọng số → ít tham số, nhanh hơn
- 🔹 **Ứng dụng**: Các đặc trưng âm thanh thường cô đặc ở một số phần tần số

**LSTM (Long Short-Term Memory)**:
- 🔹 **Điểm mạnh**: Học các mối quan hệ dài hạn trong chuỗi
- 🔹 **Lợi thế**: Giải quyết bài toán vanishing/exploding gradients
- 🔹 **Ứng dụng**: Cảm xúc phụ thuộc vào toàn bộ chuỗi giọng nói, không chỉ một phần

**Bidirectional LSTM**:
- 🔹 **Tính năng**: Xử lý chuỗi cả hai hướng (trước-sau và sau-trước)
- 🔹 **Lợi thế**: Capture bối cảnh đầy đủ hơn
- 🔹 **Ví dụ**: Từ "read" có cách phát âm khác tùy vào ngữ cảnh trước/sau

### Hình Dạng (Shape) Qua Các Tầng:

```
Input:                   (batch, 220, 40)
After Conv1D + Pool:     (batch, 110, 64)
After Conv1D + Pool:     (batch, 55, 128)
After Bi-LSTM (seq):     (batch, 55, 256)     [256 = 128*2]
After Bi-LSTM (final):   (batch, 128)         [128 = 64*2]
After Dense(128):        (batch, 128)
After Dense(8):          (batch, 8)           [8 xác suất]
```

### Tham Số Huấn Luyện:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Hàm mất mát: Categorical CrossEntropy
# → Dùng khi có 3+ lớp, label đã là one-hot
# → Penalize nếu xác suất của lớp đúng thấp

# Optimizer: Adam
# → Adaptive learning rate
# → Kết hợp momentum + RMSprop
# → learning_rate=0.001: tốc độ học (nhỏ → chậm, lớn → có thể skip tối ưu)

# Metrics: Accuracy
# → Theo dõi % dự đoán đúng trong quá trình training
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### **Bước 1: Chuẩn Bị Môi Trường**

```bash
# Kiểm tra Python 3.13 đã cài chưa
py -3.13 --version
# Output: Python 3.13.x

# Tạo virtual environment (tùy chọn nhưng nên làm)
cd e:/nhandiencamxuc_LSTM-main/nhandiencamxuc_LSTM-main
py -3.13 -m venv .venv313

# Kích hoạt virtual environment
# Trên Windows:
.venv313\Scripts\activate
# Trên macOS/Linux:
source .venv313/bin/activate
```

### **Bước 2: Cài Đặt Phụ Thuộc**

```bash
# Cập nhật pip (tùy chọn)
py -3.13 -m pip install --upgrade pip

# Cài đặt từ requirements.txt
py -3.13 -m pip install -r requirements.txt

# Cài thêm kagglehub để download dữ liệu (nếu cần train lại)
py -3.13 -m pip install kagglehub
```

### **Bước 3: Download Dữ Liệu RAVDESS (Nếu Cần Train)**

```bash
# Chạy script download
py -3.13 download_ravdess.py

# Output:
# Dataset path: C:/Users/DELL/.cache/kagglehub/datasets/ravdess/...
# Lưu lại đường dẫn này để dùng ở bước 4
```

### **Bước 4: Huấn Luyện Mô Hình (Nếu Cần)**

```bash
# Với dữ liệu RAVDESS đầy đủ
py -3.13 scripts/train_ravdess_8class.py \
  --data-root "C:/Users/DELL/.cache/kagglehub/datasets/ravdess/..." \
  --output-model "model/speech_emotion_lstm_8classes.keras"

# Kết quả:
# ✅ Model lưu tại: model/speech_emotion_lstm_8classes.keras
# ✅ Labels lưu tại: model/speech_emotion_lstm_8classes.labels.json
# ✅ Test Accuracy: ~93%
```

### **Bước 5: Chạy Ứng Dụng Streamlit**

```bash
# Cách 1: Từ terminal
py -3.13 -m streamlit run app.py

# Cách 2: Nếu đã activate venv
streamlit run app.py

# Output sẽ hiển thị:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501

# Mở browser vào http://localhost:8501
```

### **Bước 6: Sử Dụng Ứng Dụng**

1. **Upload File Âm Thanh**:
   - Chọn file từ máy tính (.wav, .mp3, .ogg, .m4a)
   - Kích thước file: tối đa ~200MB (Streamlit default)

2. **Chọn Chế Độ Trích Xuất Đặc Trưng**:
   - **raw**: Dùng `extract_mfcc_only()` (không cắt khoảng lặng)
   - **trimmed**: Dùng `extract_features()` (có cắt khoảng lặng)
   - **ensemble**: Trung bình kết quả từ cả 2 phương pháp

3. **Xem Kết Quả**:
   - Cảm xúc dự đoán
   - Độ tin cậy (%)
   - Chế độ dùng (raw/trimmed/ensemble)
   - (Debug mode) Xác suất của mỗi lớp

---

## 📊 Quá Trình Huấn Luyện (Chi Tiết)

### Dữ Liệu RAVDESS:

```
📦 RAVDESS Dataset
├─ 👥 24 diễn viên (Actor_01 ... Actor_24)
├─ 🎤 1440 giọng nam + 1440 giọng nữ = 2880 file tổng
├─ 😊 8 cảm xúc:
│  ├─ Neutral (Trung lập)
│  ├─ Calm (Bình tĩnh)
│  ├─ Happy (Vui vẻ)
│  ├─ Sad (Buồn)
│  ├─ Angry (Tức giận)
│  ├─ Fearful (Sợ hãi)
│  ├─ Disgust (Ghê tởm)
│  └─ Surprised (Ngạc nhiên)
├─ 📁 Cấu trúc: Actor_01/03-01-05-01-01-01-01.wav
│  └─ NN-SS-TT-AA-DD-RR-EE.wav
│     ├─ NN: Modality (01=speech, 02=song)
│     ├─ SS: Vocal channel (01=speech, 02=song)
│     ├─ TT: Emotion (01-08)
│     ├─ AA: Intensity (01=normal, 02=strong)
│     ├─ DD: Statement (01-02)
│     ├─ RR: Repetition (01-02)
│     └─ EE: Actor (01-24)
└─ 🎵 Thời lượng: mỗi file ~3.5-4 giây @ 48kHz (downsampled → 22050 Hz)
```

### Các Giai Đoạn Huấn Luyện:

**1. Epoch 1-10: Học cơ bản**
```
Training Accuracy: 20% → 40%
Validation Accuracy: 15% → 35%
Loss: Cao, giảm nhanh
│ Nhận xét: Model bắt đầu học các mẫu cơ bản
```

**2. Epoch 11-50: Cải thiện nhanh**
```
Training Accuracy: 40% → 75%
Validation Accuracy: 35% → 70%
Loss: Giảm ổn định
│ Nhận xét: Model học được các đặc trưng chính
```

**3. Epoch 51-150: Tinh chỉnh**
```
Training Accuracy: 75% → 95%
Validation Accuracy: 70% → 92%
Loss: Giảm từ từ
│ Nhận xét: Model tinh chỉnh trọng số, học các chi tiết
```

**4. Epoch 151+ (Early Stopping)**
```
Training Accuracy: ~95%
Validation Accuracy: ~92% (không cải thiện thêm)
│ Nhận xét: Dừng training để tránh overfitting
│ Lý do: Validation accuracy không cải thiện 10 epoch liên tiếp
```

### Kết Quả Cuối Cùng (Ví Dụ):

```
Test Dataset: 576 files (20% của 2880)

Overall Accuracy: 93.75%

Per-Class Metrics:
┌─────────┬──────────┬────────┬────────┬──────────┐
│ Emotion │ Precision│ Recall │ F1     │ Support  │
├─────────┼──────────┼────────┼────────┼──────────┤
│ Neutral │  0.95    │ 0.88   │ 0.91   │ 72       │
│ Calm    │  0.92    │ 0.96   │ 0.94   │ 96       │
│ Happy   │  0.96    │ 0.97   │ 0.97   │ 96       │
│ Sad     │  0.91    │ 0.93   │ 0.92   │ 96       │
│ Angry   │  0.98    │ 0.94   │ 0.96   │ 96       │
│ Fearful │  0.88    │ 0.90   │ 0.89   │ 96       │
│ Disgust │  0.92    │ 0.91   │ 0.91   │ 96       │
│Surprised│  0.89    │ 0.95   │ 0.92   │ 72       │
└─────────┴──────────┴────────┴────────┴──────────┘

Confusion Matrix:
```
Điều này có nghĩa:
- **Precision (Độ chính xác)**: Trong những file được dự đoán là "Happy", 96% thực sự là "Happy"
- **Recall (Độ nhạy)**: Trong những file thực sự là "Happy", 97% được dự đoán đúng
- **F1-Score**: Trung bình điều hòa của precision & recall

---

## 💡 Sử Dụng Ứng Dụng

### Quy Trình Sử Dụng Cơ Bản:

```
1️⃣ Mở ứng dụng tại: http://localhost:8501

2️⃣ Sidebar (bên trái):
   ├─ ☑️ "Hiển thị bằng tiếng Việt"
   │  └─ Tick để hiển thị tên cảm xúc tiếng Việt
   │
   ├─ Chọn "Chế độ trích xuất đặc trưng"
   │  ├─ raw: Nhanh, có thể có nhiễu
   │  ├─ trimmed: Tập trung vào giọng nói, mạnh hơn
   │  └─ ensemble: Kết hợp cả 2, cực kỳ chính xác
   │
   ├─ ☑️ "Hiện thông tin model (debug)"
   │  └─ Xem model summary, kiến trúc, test random input

3️⃣ Main area (giữa):
   ├─ 📤 Upload file (.wav, .mp3, .ogg, .m4a)
   ├─ 🎵 Preview audio (nghe thử)
   └─ 📊 Xem kết quả dự đoán

4️⃣ Kết Quả:
   ├─ "Cảm xúc dự đoán: 😊 Vui vẻ"
   ├─ "Độ tin cậy: 92.5%"
   └─ "Chế độ dùng: ensemble"
```

### Lợi Ích Của Các Chế Độ:

| Chế Độ | Tốc Độ | Độ Chính Xác | Kích Thước Mô Hình | Tình Huống Dùng |
|--------|--------|-------------|-----------------|-----------------|
| **raw** | 🟢 Nhanh | 🟡 TB | 🟢 Nhỏ | File jammed, nhiễu ít |
| **trimmed** | 🟡 TB | 🟢 Cao | 🟢 Nhỏ | File sạch, tập trung vào giọng nói |
| **ensemble** | 🔴 Chậm | 🟢🟢 Rất cao | 🟡 Vừa | File quan trọng, cần độ chính xác cao |

---

## 🔧 Xử Lý Lỗi & Gỡ Rối

### Lỗi Thường Gặp:

#### ❌ **"Không tìm thấy TensorFlow"**

```
Lỗi: ModuleNotFoundError: No module named 'tensorflow'

Nguyên nhân: Sử dụng sai version Python

Cách Sửa:
1. Kiểm tra Python version:
   py -3.13 --version
   
2. Cài lại dependencies:
   py -3.13 -m pip install -r requirements.txt
```

#### ❌ **"File âm thanh không hợp lệ"**

```
Lỗi: librosa.exceptions.LibrosaError: ...

Nguyên nhân: 
- File bị hỏng
- Format không hỗ trợ (chỉ hỗ trợ .wav, .mp3, .ogg, .m4a)

Cách Sửa:
1. Kiểm tra file có tồn tại: ls -la file.wav
2. Dùng ffmpeg kiểm tra: ffmpeg -i file.wav
3. Convert sang .wav: ffmpeg -i file.mp3 -acodec pcm_s16le -ar 22050 file.wav
```

#### ❌ **"Shape không đúng (1, 220, 40)"**

```
Lỗi: Shape không đúng. Expected: (1, 220, 40), Got: (1, 150, 40)

Nguyên nhân: File âm thanh quá ngắn (< 1.5 giây)

Cách Sửa:
1. Dùng file dài hơn 2 giây
2. Hoặc edit feature_extraction.py: tăng max_len=220 → 150
```

#### ❌ **"Model không load được"**

```
Lỗi: Cannot load model. Error: ...

Nguyên nhân: File model bị hỏng, version TensorFlow không khớp

Cách Sửa:
1. Kiểm tra file model tồn tại:
   ls -la model/speech_emotion_lstm_8classes.keras
   
2. Kiểm tra file labels tồn tại:
   ls -la model/speech_emotion_lstm_8classes.labels.json
   
3. Huấn luyện lại model:
   py -3.13 scripts/train_ravdess_8class.py --data-root "path/to/ravdess"
```

#### ❌ **"Dự đoán chỉ là một lớp (Angry 100% mọi lúc)"**

```
Lỗi: Model dự đoán sai, collapse thành một lớp

Nguyên nhân: 
- Dataset không cân bằng (một lớp quá nhiều)
- Model bị overfitting
- Dữ liệu training không đầy đủ 8 lớp

Cách Sửa:
1. Kiểm tra dữ liệu training:
   py -3.13 scripts/train_ravdess_8class.py \
     --data-root "..." \
     --allow-missing-classes  # flag này để debug
     
2. Xem log dataset:
   Total valid files: 2880
   Label distribution:
   - Neutral: 192
   - Calm: 384
   - ...
   
3. Nếu thiếu lớp, kiểm tra đường dẫn RAVDESS
```

#### ❌ **"Streamlit không mở được"**

```
Lỗi: Address already in use (port 8501)

Nguyên nhân: Port 8501 đang bị chiếm bởi process khác

Cách Sửa:
1. Kill process đang chạy:
   py -3.13 -m streamlit run app.py --logger.level=debug
   
2. Hoặc chỉ định port khác:
   py -3.13 -m streamlit run app.py --server.port 8502
```

### Debug Mode:

**Kích hoạt debug mode trong app**:

```
1. Sidebar → ☑️ "Hiện thông tin model (debug)"
2. Xem:
   - Model architecture (summary)
   - Số lớp: 8
   - Random test prediction
```

**Hoặc từ terminal**:

```python
import numpy as np
from tensorflow import keras

# Load model
model = keras.models.load_model('model/speech_emotion_lstm_8classes.keras')

# Test với random input
x_test = np.random.randn(1, 220, 40).astype('float32')
pred = model.predict(x_test)
print("Prediction shape:", pred.shape)  # (1, 8)
print("Prediction:", pred[0])           # 8 xác suất
print("Class index:", np.argmax(pred))  # 0-7
```

---

## 🎓 Tóm Tắt Luồng Toàn Bộ

```
🎤 Người dùng upload file.wav
   │
   ▼
📂 Lưu file tạm
   │
   ▼
🎵 Librosa load file (22050 Hz)
   │
   ▼
📊 Trích MFCC: 40 hệ số/frame, 220 frames
   │
   ▼
✅ Kiểm tra shape (1, 220, 40)
   │
   ▼
🧠 Mô hình dự đoán:
   Input (1,220,40) → CNN → Bi-LSTM → Dense → Softmax
   │
   ▼
📈 Kết quả: 8 xác suất
   [p0, p1, p2, p3, p4, p5, p6, p7]
   │
   ▼
🎯 Lấy max → class index → nhãn tiếng Việt
   │
   ▼
🖥️ Hiển thị trên Streamlit UI
   "Cảm xúc: Vui vẻ (92% tin cậy)"
```

---

## 📚 Tài Liệu Tham Khảo

- **TensorFlow/Keras**: https://www.tensorflow.org/
- **Librosa (xử lý âm thanh)**: https://librosa.org/
- **Streamlit**: https://streamlit.io/
- **RAVDESS Dataset**: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- **MFCC giải thích**: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

---

**Hệ thống này được thiết kế để dễ sử dụng và mở rộng. Nếu bạn có bất kỳ câu hỏi, hãy tham khảo tài liệu hoặc kiểm tra code trong từng file!** 🚀
