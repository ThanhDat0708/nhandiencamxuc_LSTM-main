import tensorflow as tf
import os

model_path = "model/speech_emotion_lstm_improved.keras"
saved_model_path = "model/saved_model"

print("🔄 Đang thử load model với custom objects...")

# Định nghĩa custom objects để xử lý InputLayer
custom_objects = {
    'InputLayer': tf.keras.layers.InputLayer,
    'Masking': tf.keras.layers.Masking,
}

try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )
    print("✅ Load model thành công!")

    # Lưu lại dưới dạng SavedModel
    tf.saved_model.save(model, saved_model_path)
    print(f"✅ Đã chuyển đổi và lưu SavedModel tại: {saved_model_path}")
    print("Bạn có thể chạy app.py ngay bây giờ.")

except Exception as e:
    print(f"❌ Vẫn lỗi: {e}")
    print("\n💡 Khuyến nghị: Hãy nâng cấp TensorFlow bằng lệnh:")
    print("   pip install --upgrade tensorflow")