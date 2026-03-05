# TỔNG QUAN DỰ ÁN

**Tên dự án:** Ứng dụng Nhận diện và Dịch Giọng nói Local AI (Windows)

## Mục tiêu Dự án

- **Chạy hoàn toàn local (offline):** Bảo mật dữ liệu tuyệt đối, không gửi âm
  thanh hay văn bản lên bất kỳ máy chủ nào.
- **Không mất phí API:** Miễn phí trọn đời do sử dụng các mô hình mã nguồn mở
  (Open-source AI models).
- **Hỗ trợ nhiều cấu hình Windows khác nhau:** Tương thích với dải phần cứng
  rộng từ laptop văn phòng đến máy trạm đồ họa.
- **Tự động tối ưu theo cấu hình phần cứng:** Tự động phát hiện và cấu hình mô
  hình AI để đạt hiệu suất tốt nhất dựa trên TÀI NGUYÊN (CPU/GPU/RAM) hiện có.

---

## 1. Tổng quan hệ thống

Kiến trúc hệ thống được thiết kế theo hướng module hóa, tách biệt hoàn toàn giữa
giao diện người dùng (Frontend) và lõi xử lý AI (Backend/Engine) để đảm bảo trải
nghiệm mượt mà, không bị treo ứng dụng (non-blocking UI).

- **Giao diện người dùng (UI):** Được xây dựng dựa trên framework `PySide6` (Qt
  cho Python), mang lại giao diện hiện đại, chuyên nghiệp, phản hồi nhanh và hỗ
  trợ tốt cho hệ điều hành Windows.
- **Lõi xử lý Trí tuệ Nhân tạo (AI Engine):**
  - Sử dụng **Python** làm ngôn ngữ lập trình chính.
  - Xử lý tensor và tính toán song song với **PyTorch**.
  - Nhận diện giọng nói (Speech-to-Text) thông qua mô hình **Whisper**.
  - Tách biệt và nhận diện người nói (Speaker Diarization) bằng mô hình
    **pyannote.audio**.
  - Dịch thuật văn bản lập tức bằng các mô hình dịch máy từ **HuggingFace** (như
    NLLB hoặc MarianMT).
- **Cơ chế tự động phát hiện phần cứng:** Hệ thống quét liên tục ở màn hình khởi
  động để thu thập thông tin về dung lượng RAM, số nhân CPU, và đặc biệt là sự
  hiện diện của GPU có hỗ trợ CUDA cùng với dung lượng VRAM. Dựa vào đó, ứng
  dụng sẽ đề xuất hoặc tự động áp dụng cấu hình (kích thước mô hình, precision
  fp16/int8) phù hợp nhất.

---

## 2. Các chức năng chính

- Nhận diện giọng nói thời gian thực từ micro (Real-time Microphone
  Transcription).
- Nhận diện trực tiếp âm thanh từ hệ thống máy tính (System Audio) như video
  YouTube, phim, hoặc cuộc gọi trực tuyến.
- Chuyển đổi định dạng file âm thanh/video sẵn có (mp3, wav, mp4, m4a...) sang
  văn bản.
- Phát hiện và phân biệt người nói (Speaker Diarization - ví dụ: Xử lý đoạn giao
  tiếp và gán nhãn Người nói 1, Người nói 2).
- Cung cấp tính năng gộp (Merge) hoặc tách (Split) nhãn người nói khi có sai sót
  xác định trong các tình huống hội thoại chồng chéo.
- Dịch tự động từ ngôn ngữ gốc sang nhiều ngôn ngữ đích phổ biến (Anh, Việt,
  Pháp, Nhật,...).
- Hiển thị văn bản kết quả trực quan, phân đoạn rõ ràng theo thời gian
  (timestamps) và theo người nói, thân thiện và dễ đọc.
- Xuất kết quả dán nhãn ra các định dạng chuẩn chuyên môn: `.txt` (văn bản
  thuần), `.docx` (Microsoft Word), và `.srt` (phụ đề video).
- Hiển thị chi tiết bảng thông số phần cứng hiện tại của hệ thống.
- Cấu hình linh hoạt mức hiệu suất nhận diện (Profiles): **Fast** (Nhanh - mô
  hình nhỏ), **Balanced** (Cân bằng - mô hình tầm trung), **Accurate** (Chính
  xác cao - mô hình lớn).
- Chức năng chọn chế độ Tính toán (Compute Mode) linh hoạt: **Auto** (Tự động ưu
  tiên GPU), **GPU** (bắt buộc dùng Card đồ họa), hoặc **CPU** (Sử dụng vi xử lý
  trung tâm).

---

## 3. Chức năng Cài đặt (Settings)

Giao diện cài đặt được phân chia khoa học thành các tab riêng biệt:

- **Hiệu suất xử lý (Performance):** Lựa chọn chuẩn Compute Mode (Auto/GPU/CPU),
  tự động điều tiết bộ nhớ (VRAM allocation), số luồng CPU (CPU threads).
- **Mức độ nhận diện (Transcription):** Chọn profile nhận diện
  (Fast/Balanced/Accurate), kích hoạt bộ lọc âm thanh nhiễu (noise suppression),
  tùy chỉnh VAD (Voice Activity Detection).
- **Dịch ngôn ngữ (Translation):** Thiết lập ngôn ngữ mặc định đầu vào, ngôn ngữ
  đích cần dịch sang, bật/tắt tự động dịch sau khi nhận diện xong.
- **Âm thanh (Audio):** Quản lý thiết bị đầu vào (Microphone/System Stereo Mix),
  mức khuếch đại âm lượng, kiểm tra tín hiệu đầu vào (test mic).
- **Lưu trữ (Storage):** Thiết lập thư mục lưu mặc định cho file kết quả xuất
  ra, quy tắc đặt tên file tự động (vd: Tên file gốc + timestamp), quản lý các
  bộ nhớ đệm (cache/temp files).
- **Thông tin hệ thống (System Info):** Phân tích chi tiết phần cứng (Tên CPU,
  Tên GPU, VRAM, RAM, Phiên bản CUDA, trạng thái tài nguyên hệ thống).

---

## 4. Yêu cầu kỹ thuật

- **Hệ điều hành:** Yêu cầu tối thiểu hệ điều hành Windows 10 (64-bit) hoặc mới
  hơn, bao gồm Windows 11.
- **Tính toán:** Ứng dụng hoạt động chéo tính năng trên cả nền tảng CPU và GPU.
- **Chống lỗi linh hoạt (Auto-fallback):** Nếu hệ thống yêu cầu chạy GPU nhưng
  thiết bị không có Card Nvidia hỗ trợ CUDA (hoặc lỗi driver), ứng dụng sẽ tự
  động chuyển mạch (fallback) sang chạy bằng CPU và hiển thị thông báo kịp thời
  cho người dùng mà không bị crash.
- **Kết nối Mạng:** Ứng dụng chạy offline hoàn toàn độc lập 100%, không cần kết
  nối mạng internet sau khi đã thiết lập/tải mô hình ban đầu.

---

## 5. Định hướng phát triển tương lai

- **Tự động tải model (Auto-download models):** Xây dựng trình quản lý tự động
  kiểm tra, tải xuống hoặc cập nhật các phiên bản mô hình ngôn ngữ mới ngay
  trong ứng dụng khi có mạng.
- **Thêm theme Dark/Light:** Hỗ trợ giao diện sáng/tối tự động đồng bộ theo tùy
  chỉnh của hệ điều hành Windows hoặc dễ dàng chuyển đổi thủ công.
- **Xuất phụ đề video (Video Subtitle Export):** Tích hợp công cụ tự động "đốt"
  (hardcode) phụ đề `.srt` trực tiếp lên file video gốc.
- **Tích hợp AI tóm tắt nội dung:** Tích hợp các mô hình ngôn ngữ Lớn (Local LLM
  như Llama 3 hoặc Mistral nhỏ) để phân tích, tóm tắt và trích xuất ý chính từ
  các đoạn hội thoại dài.
