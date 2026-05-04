from pathlib import Path
import re
import numpy as np
import tensorflow as tf

from utils.feature_extraction import extract_features

EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

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


model = tf.keras.models.load_model('model/speech_emotion_lstm_improved.keras', compile=False)
folder = Path(r'e:\Actor_01')
files = sorted(folder.glob('*.wav'))[:20]

print('File | TrueLabel | PredIdx | PredLabel | Probs')
for path in files:
    feats = extract_features(str(path))
    if feats is None:
        print(path.name, '| feature=None')
        continue
    pred = model.predict(feats, verbose=0)[0]
    idx = int(np.argmax(pred))
    probs = [round(float(x), 4) for x in pred]
    print(f'{path.name} | {parse_label(path.name)} | {idx} | {EMOTIONS[idx]} | {probs}')
