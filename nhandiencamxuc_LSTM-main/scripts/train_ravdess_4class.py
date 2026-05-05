import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.preprocessing import extract_mfcc_only

# 8 lớp RAVDESS chuẩn
CLASS_ORDER = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_ORDER)}
RAVDESS_TO_CLASS = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}
FILE_PATTERN = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_ravdess_label(file_path: Path):
    match = FILE_PATTERN.match(file_path.name)
    if not match:
        return None

    emotion_code = match.group(3)
    return RAVDESS_TO_CLASS.get(emotion_code)


def build_dataset(data_root: Path):
    audio_files = sorted(data_root.rglob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"Không tìm thấy file .wav nào trong: {data_root}")

    features = []
    labels = []
    skipped = 0

    for audio_path in audio_files:
        label_name = parse_ravdess_label(audio_path)
        if label_name is None:
            skipped += 1
            continue

        extracted = extract_mfcc_only(str(audio_path))
        if extracted is None:
            skipped += 1
            continue

        # extract_mfcc_only trả về shape (1, 220, 40)
        features.append(extracted[0])
        labels.append(CLASS_TO_INDEX[label_name])

    if not features:
        raise RuntimeError("Không tạo được dữ liệu huấn luyện nào.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return x, y, skipped


def build_model(input_shape=(220, 40), num_classes=8):
    model = Sequential([
        Input(shape=input_shape),
        BatchNormalization(),
        Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an 8-class RAVDESS speech emotion model.")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Thư mục gốc chứa các Actor_* của RAVDESS (ví dụ: E:/Actor_01 hoặc E:/RAVDESS).",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="model/speech_emotion_lstm_8classes.keras",
        help="Đường dẫn model đầu ra.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Không tìm thấy data root: {data_root}")

    print(f"Đang quét dữ liệu từ: {data_root}")
    x, y, skipped = build_dataset(data_root)
    print(f"Tổng file hợp lệ: {len(x)} | Bỏ qua: {skipped}")
    print("Phân bố nhãn:", Counter(y))

    y_onehot = to_categorical(y, num_classes=len(CLASS_ORDER))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_onehot,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    y_train_labels = np.argmax(y_train, axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_labels,
    )

    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

    model = build_model(input_shape=x_train.shape[1:], num_classes=len(CLASS_ORDER))
    model.summary()

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ModelCheckpoint(
            filepath=str(output_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Tải lại model tốt nhất từ checkpoint trước khi đánh giá cuối.
    model = tf.keras.models.load_model(output_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_ORDER, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    label_map_path = output_path.with_suffix(".labels.json")
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "class_order": CLASS_ORDER,
                "class_to_index": CLASS_TO_INDEX,
                "source": "RAVDESS 8-class subset",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nĐã lưu model tại: {output_path}")
    print(f"Đã lưu nhãn tại: {label_map_path}")


if __name__ == "__main__":
    main()
