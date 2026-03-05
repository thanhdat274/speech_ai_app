# KẾ HOẠCH XÂY DỰNG DỰ ÁN (TASK BUILD PLAN)

**Dự án:** Ứng dụng Nhận diện và Dịch Giọng nói Local AI (Windows)  
**Ngôn ngữ lập trình:** Python  
**Giao diện (UI):** PySide6  
**Công nghệ AI (Backend):** Whisper + PyTorch + pyannote.audio + HuggingFace
Translation Models

Dưới đây là danh sách phân rã các đầu việc (tasks) theo từng giai đoạn (phasing)
để triển khai và hoàn thiện dự án.

---

## PHASE 1 - Chuẩn bị môi trường (Environment Setup)

- [x] Cài đặt phiên bản Python chuẩn (khuyến nghị Python 3.10 hoặc 3.11).
- [x] Khởi tạo môi trường ảo (Virtual Environment) cho dự án.
- [x] Thiết lập `requirements.txt` và cài đặt các thư viện lõi.
- [x] Viết script kiểm tra cơ bản sự hiện diện của phần cứng.

## PHASE 2 - Xây dựng kiến trúc project (Backend Architecture)

- [x] Tạo cấu trúc thư mục tiêu chuẩn (`core/`, `docs/`, `main.py`).
- [x] **Tạo module Config (`config_manager.py`):** Quản lý lưu trữ settings
      JSON.
- [x] **Tạo module Hardware Detector:** Tích hợp trong `AIEngine`.
- [x] **Tạo module AI Engine (`ai_engine.py`):** Whisper, Pyannote, Translation.
- [x] **Tạo module Audio Capture (`audio_engine.py`):** Mic và WASAPI Loopback.
- [x] **Tạo module Translation Engine (`translation_engine.py`):** NLLB offline.

## PHASE 3 - Xây dựng Giao diện (UI Implementation)

- [x] **Tạo Main Window (`main.py`):** Giao diện PySide6 hiện đại.
- [x] **Tạo Sidebar:** Điều hướng linh hoạt.
- [x] **Tạo Transcript Panel:** Hiển thị kết quả real-time và hỗ trợ file.
- [x] **Tạo Settings Panel:** Cấu hình AI, Hardware, Audio, Devices.
- [x] **Nối UI với Backend (Signals/Slots):** Cấu trúc QThread orchestration
      hoàn chỉnh.

## PHASE 4 - Tối ưu hiệu suất (Performance Optimization)

- [x] Lập trình logic **Tự động detect cấu hình.**
- [x] Hiện thực hóa logic **Tự chọn model theo VRAM.**
- [x] Xử lý quản lý bộ nhớ và giải phóng VRAM.
- [x] **Giảm độ trễ (Latency tuning):** Chunking 2.0s và Noise Gate động.

## PHASE 5 - Kiểm thử đa nền tảng (Testing)

- [ ] **Test thiết bị có GPU:** Chạy kiểm thử tốc độ thời gian thực với mô hình
      lớn trên các dòng card RTX/GTX xem khả năng xử lý.
- [ ] **Test thiết bị không GPU:** Kiểm tra độ trễ và sự ổn định khi buộc chạy
      các tác vụ Heavy Inference thuần bằng CPU.
- [ ] **Test dung lượng RAM thấp:** Mô phỏng quy trình xử lý file rất dài (>1
      tiếng) trên máy có RAM giới hạn để kiểm tra hiện trạng tràn RAM (memory
      leak).
- [ ] **Test nhận diện YouTube (System Audio):** Chạy bật 1 video trên youtube,
      bắt luồng âm thanh hệ thống và kiểm tra văn bản sinh ra.
- [ ] **Test Diarization (Nhiều người nói):** Đoạn thu âm có 3 giọng người khác
      nhau để kiểm thử độ chính xác dán nhãn Speaker 1, 2, 3 của pyannote.

## PHASE 6 - Đóng gói và Phân phối (Packaging)

- [ ] Cài đặt `PyInstaller` và thiết kế file `.spec` tích hợp hook phù hợp.
- [ ] Đóng gói toàn bộ source code cùng các file mô hình cứng thành 1 thư mục
      Portable (hoặc 1 file `.exe` duy nhất).
- [ ] Triển khai ứng dụng xuất ra trên 1 máy ảo (hoặc một máy tính Windows hoàn
      toàn trắng) để chạy thử.
- [ ] Xử lý và vá triệt để tất cả các lỗi thiếu file DLL thư viện phụ trợ (đặc
      biệt là liên quan tới cuDNN hoặc ffmpeg).
- [ ] Rà soát, giảm thiểu khối lượng các thư viện không thiết yếu để tối ưu độ
      lớn (size) của thư mục chứa app.

## PHASE 7 - Nâng cấp tương lai (Future Upgrades)

- [ ] **Auto download model:** Lập trình pipeline tự động fetch link tải
      checkpoint mô hình qua HTTP/TDM khi chạy lần đầu.
- [ ] **Update system:** Thêm Auto-updater để tải bản patch tính năng từ server
      về ứng dụng của người dung.
- [ ] **Plugin mở rộng:** Cung cấp API đơn giản để app hỗ trợ các module gắn
      ngoài (Ví dụ: Plugin tích hợp phím tắt Global, plugin thông báo nổi màn
      hình).
