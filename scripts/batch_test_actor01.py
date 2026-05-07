from pathlib import Path
import json
import re
import sys
import numpy as np
import tensorflow as tf

# Ensure imports like `utils.*` work regardless of launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.feature_extraction import extract_features

DEFAULT_CLASS_ORDER_4 = ["Angry", "Happy", "Sad", "Neutral"]
DEFAULT_CLASS_ORDER_4_VI = ["Tức giận", "Vui vẻ", "Buồn", "Trung lập"]
DEFAULT_CLASS_ORDER_8 = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
DEFAULT_CLASS_ORDER_8_VI = ["Trung lập", "Bình tĩnh", "Vui vẻ", "Buồn", "Tức giận", "Sợ hãi", "Ghê tởm", "Ngạc nhiên"]
DEFAULT_VI_TRANSLATIONS = {
    "Angry": "Tức giận",
    "Calm": "Bình tĩnh",
    "Disgust": "Ghê tởm",
    "Fearful": "Sợ hãi",
    "Happy": "Vui vẻ",
    "Neutral": "Trung lập",
    "Sad": "Buồn",
    "Surprised": "Ngạc nhiên",
}


def resolve_class_labels(model, model_path: str):
    label_path = Path(model_path).with_suffix(".labels.json")
    output_units = int(model.output_shape[-1])

    if label_path.exists():
        try:
            with label_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            class_order = payload.get("class_order")
            if isinstance(class_order, list) and len(class_order) == output_units:
                class_order_vi = [DEFAULT_VI_TRANSLATIONS.get(name, name) for name in class_order]
                return class_order, class_order_vi
        except Exception:
            pass

    if output_units == 8:
        return DEFAULT_CLASS_ORDER_8, DEFAULT_CLASS_ORDER_8_VI

    if output_units == 4:
        return DEFAULT_CLASS_ORDER_4, DEFAULT_CLASS_ORDER_4_VI

    fallback = [f"Class {index + 1}" for index in range(output_units)]
    return fallback, fallback

RAVDESS_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}

pattern = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$", re.IGNORECASE)


def parse_label(name: str):
    m = pattern.match(name)
    if not m:
        return None
    return RAVDESS_MAP.get(m.group(3), None)


MODEL_CANDIDATES = [
    'model/speech_emotion_lstm_8classes.keras',
    'model/speech_emotion_lstm_improved.keras',
    'model/speech_emotion_lstm_4classes.keras',
]

for candidate in MODEL_CANDIDATES:
    if Path(candidate).exists():
        model_path = candidate
        model = tf.keras.models.load_model(model_path, compile=False)
        break
else:
    raise FileNotFoundError('Không tìm thấy model hợp lệ trong thư mục model/.')

EMOTIONS, EMOTIONS_VI = resolve_class_labels(model, model_path)
folder = Path(r'e:\Actor_01')
files = sorted(folder.glob('*.wav'))[:20]

print('File | TrueLabel | PredIdx | PredLabel | PredLabelVI | Probs')
for path in files:
    feats = extract_features(str(path))
    if feats is None:
        print(path.name, '| feature=None')
        continue
    pred = model.predict(feats, verbose=0)[0]
    idx = int(np.argmax(pred))
    probs = [round(float(x), 4) for x in pred]
    print(f'{path.name} | {parse_label(path.name)} | {idx} | {EMOTIONS[idx]} | {EMOTIONS_VI[idx]} | {probs}')
