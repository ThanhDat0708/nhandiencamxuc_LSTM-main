# nhandiencamxuc_LSTM - Hệ Thống Nhận Diện Cảm Xúc Từ Giọng Nói

## 📋 Mô Tả
Hệ thống nhận diện cảm xúc từ giọng nói (Speech Emotion Recognition) sử dụng mô hình LSTM (Long Short-Term Memory). Dự án này kết hợp xử lý tín hiệu âm thanh và deep learning để phân loại cảm xúc từ các file âm thanh.

## 🎯 Tính Năng
- **Nhận diện cảm xúc**: Phân loại cảm xúc từ giọng nói (vui vẻ, buồn, tức giận, bình tĩnh, v.v.)
- **Xử lý âm thanh**: Trích xuất đặc trưng MFCC từ file âm thanh
- **Mô hình Deep Learning**: Sử dụng LSTM để học các mẫu cảm xúc phức tạp
- **Giao diện Web**: Ứng dụng Streamlit dễ sử dụng để test hệ thống

## 🚀 Test Hệ Thống
Bạn có thể test hệ thống nhận diện cảm xúc từ giọng nói qua ứng dụng Streamlit:

🔗 **[Test hệ thống tại đây](https://nhandiencamxuclstm-ufgyhsghjbfrqjon9w2rsu.streamlit.app/)**

Chỉ cần upload file âm thanh (định dạng: WAV, MP3, OGG) và hệ thống sẽ tự động phân loại cảm xúc.

## 🛠️ Yêu Cầu Kỹ Thuật
- Python 3.7+
- TensorFlow/Keras
- Librosa (xử lý âm thanh)
- Streamlit
- NumPy, Pandas

## 📦 Cài Đặt
```bash
pip install -r requirements.txt
```

## 🎬 Chạy Ứng Dụng Streamlit Cục Bộ
```bash
streamlit run app.py
```

## 📊 Dữ Liệu
Hệ thống được huấn luyện trên các tập dữ liệu cảm xúc từ giọng nói công khai, chứa các bản ghi âm có các cảm xúc khác nhau.

## 🤝 Đóng Góp
Mọi đóng góp đều được chào đón! Vui lòng fork repository và tạo pull request.

## 📄 Giấy Phép
MIT License
