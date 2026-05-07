import numpy as np
import librosa


def _build_mfcc_features(audio_path, max_len=220, n_mfcc=40, sr=22050, trim_silence=True):
    """Tạo MFCC (220, 40) theo pipeline gốc, không chuẩn hóa để giữ đúng phân bố train."""
    y, _ = librosa.load(audio_path, sr=sr, duration=6.0)

    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
        window='hann',
        center=True,
    )

    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    return mfcc.T.astype(np.float32)


def extract_features(audio_path, max_len=220, n_mfcc=40, sr=22050):
    """Extractor khuyến nghị: MFCC gốc có cắt khoảng lặng nhẹ."""
    try:
        features = _build_mfcc_features(audio_path, max_len=max_len, n_mfcc=n_mfcc, sr=sr, trim_silence=True)
        return np.expand_dims(features, axis=0)
    except Exception as e:
        print(f"Lỗi extract_features: {e}")
        return None