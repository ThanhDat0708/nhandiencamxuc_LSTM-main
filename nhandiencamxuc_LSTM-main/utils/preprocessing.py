import numpy as np
import librosa
import os
from pathlib import Path

def pad_or_truncate_mfcc(mfcc, max_len=220):
    """Đảm bảo MFCC có đúng độ dài 220 frames"""
    if mfcc.shape[1] > max_len:
        return mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        return np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')


def extract_mfcc_only(audio_path, max_len=220, n_mfcc=40):
    """
    Phiên bản đơn giản chỉ trích xuất MFCC (không delta) 
    - Dùng để test hoặc khi muốn nhẹ hơn
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        
        mfcc = pad_or_truncate_mfcc(mfcc, max_len)
        mfcc = mfcc.T  # (220, 40)
        mfcc = np.expand_dims(mfcc, axis=0).astype(np.float32)
        
        return mfcc
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