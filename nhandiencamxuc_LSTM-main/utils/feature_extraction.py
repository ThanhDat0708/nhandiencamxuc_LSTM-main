import numpy as np
import librosa
import soundfile as sf

def extract_features(audio_path, max_len=220, n_mfcc=40, sr=22050):
    """
    Trích xuất features cải tiến để giảm bias, cố gắng khớp với lúc train model
    """
    try:
        # Load audio với trim silence mạnh hơn
        y, orig_sr = librosa.load(audio_path, sr=sr, duration=6.0)
        y, _ = librosa.effects.trim(y, top_db=25)   # trim silence mạnh
        
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode='constant')

        # MFCC + Delta + Delta2 (rất quan trọng)
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512,
            win_length=None,
            window='hann',
            center=True
        )
        
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Kết hợp và lấy đúng 40 features
        features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        features = features[:n_mfcc, :]   # giữ 40 dòng đầu

        # Padding hoặc truncate
        if features.shape[1] > max_len:
            features = features[:, :max_len]
        else:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

        # Quan trọng: Chuẩn hóa (Normalization) - giúp giảm bias mạnh
        features = librosa.util.normalize(features, axis=1)
        
        # Transpose → (time, feature)
        features = features.T.astype(np.float32)
        
        # Thêm batch dimension
        features = np.expand_dims(features, axis=0)
        
        return features

    except Exception as e:
        print(f"Lỗi extract_features: {e}")
        return None