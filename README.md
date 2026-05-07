# nhandiencamxuc_LSTM - Hệ Thống Nhận Diện Cảm Xúc Từ Giọng Nói

## 📋 Mô Tả
Hệ thống nhận diện cảm xúc từ giọng nói (Speech Emotion Recognition) sử dụng mô hình LSTM (Long Short-Term Memory). Dự án này kết hợp xử lý tín hiệu âm thanh và deep learning để phân loại cảm xúc từ các file âm thanh.

## 🎯 Tính Năng
- **Nhận diện cảm xúc**: Phân loại 8 cảm xúc từ giọng nói (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Xử lý âm thanh**: Trích xuất đặc trưng MFCC từ file âm thanh
- **Mô hình Deep Learning**: Sử dụng LSTM để học các mẫu cảm xúc phức tạp
- **Giao diện Web**: Ứng dụng Streamlit dễ sử dụng để test hệ thống

## 🚀 Test Hệ Thống
Bạn có thể test hệ thống nhận diện cảm xúc từ giọng nói qua ứng dụng Streamlit:

🔗 **[Test hệ thống tại đây](https://nhandiencamxuclstm-ufgyhsghjbfrqjon9w2rsu.streamlit.app/)**

Chỉ cần upload file âm thanh (định dạng: WAV, MP3, OGG) và hệ thống sẽ tự động phân loại cảm xúc.
Mô hình huấn luyện mặc định hiện nhắm tới 8 lớp RAVDESS.

## 🛠️ Yêu Cầu Kỹ Thuật
- Python 3.13
- TensorFlow/Keras 2.20.0+
- Librosa (xử lý âm thanh)
- Streamlit
- NumPy, Pandas

## 📦 Cài Đặt
```bash
py -3.13 -m pip install -r requirements.txt
```

Nếu `py` mặc định của máy đang trỏ sang Python 3.14, hãy luôn thêm `-3.13` khi cài đặt hoặc chạy dự án.

## 🔄 Chuyển Model
```bash
py -3.13 convert_model.py
```

## 🎬 Chạy Ứng Dụng Streamlit Cục Bộ
```bash
py -3.13 -m streamlit run app.py
```

## 📊 Dữ Liệu
Hệ thống được huấn luyện trên các tập dữ liệu cảm xúc từ giọng nói công khai, chứa các bản ghi âm có các cảm xúc khác nhau.

## 🤝 Đóng Góp
Mọi đóng góp đều được chào đón! Vui lòng fork repository và tạo pull request.

## 📄 Giấy Phép
MIT License
