import numpy as np
import librosa
import os
from pathlib import Path


def _build_mfcc_features(audio_path, max_len=220, n_mfcc=40, sr=22050, trim_silence=False):
    """Tạo MFCC (220, 40) theo kiểu gốc, không chuẩn hóa."""
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

def pad_or_truncate_mfcc(mfcc, max_len=220):
    """Đảm bảo MFCC có đúng độ dài 220 frames"""
    if mfcc.shape[1] > max_len:
        return mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        return np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')


def extract_mfcc_only(audio_path, max_len=220, n_mfcc=40):
    """
    Phiên bản MFCC không cắt khoảng lặng, dùng để fallback so sánh.
    """
    try:
        mfcc = _build_mfcc_features(audio_path, max_len=max_len, n_mfcc=n_mfcc, sr=22050, trim_silence=False)
        return np.expand_dims(mfcc, axis=0)
    except Exception as e:
        print(f"Lỗi extract_mfcc_only: {e}")
        return None


def get_audio_duration(audio_path):
    """Lấy thời lượng file âm thanh (giây)"""
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=1)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except:
        return 0.0


# Hàm hỗ trợ kiểm tra input hợp lệ cho model
def validate_input_shape(features):
    """Kiểm tra shape của features có đúng với model không"""
    expected_shape = (1, 220, 40)
    if features is None:
        return False, "Features is None"
    if features.shape != expected_shape:
        return False, f"Shape không đúng. Expected: {expected_shape}, Got: {features.shape}"
    return True, "Input shape hợp lệ"