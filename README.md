# 👻 Ghost Radar (Flutter)

Ứng dụng Android **Flutter** mô phỏng radar và phân tích tín hiệu microphone trong vùng hạ tần **0,5–20 Hz**.

> ⚠️ Đây là ứng dụng nghiên cứu/giải trí. Microphone điện thoại thường có bộ lọc cắt dưới 20 Hz, nên app chỉ phân tích phần tín hiệu mà phần cứng cho phép.

## Tính năng

- 📡 Radar 360° quét liên tục
- 🎙️ Đọc microphone thời gian thực
- 📊 Phân tích biên độ + zero-crossing (ước lượng tần số)
- 🔎 Tập trung vùng 0,5–20 Hz
- 🎯 Hiển thị mức tín hiệu tương đối
- ⚠️ Cảnh báo khi năng lượng vượt ngưỡng
- 📱 APK Android ARM64 (tối ưu Samsung A17)
- 🤖 GitHub Actions tự động build APK

## Cấu trúc project

```
ghost_radar/
├── android/                 # Android native config
│   └── app/
│       ├── build.gradle     # minSdk 23, targetSdk 34, ARM64
│       └── src/main/
│           ├── AndroidManifest.xml
│           └── kotlin/.../MainActivity.kt
├── lib/
│   └── main.dart            # Toàn bộ code Flutter (UI + logic)
├── .github/
│   └── workflows/
│       └── build-apk.yml    # CI build APK
├── pubspec.yaml             # Dependencies
└── README.md
```

## Build APK trên GitHub

1. Push code lên GitHub
2. Vào tab **Actions**
3. Workflow **"Build Ghost Radar APK (Flutter)"** tự chạy
4. Đợi 10-20 phút → tải APK ở mục **Artifacts** (cuối trang workflow)

Hoặc tạo tag `v1.0.0` để tự động tạo Release.

## Chạy thử trên máy (cần Flutter SDK)

```bash
flutter pub get
flutter run            # kết nối điện thoại qua USB
flutter build apk --release --target-platform android-arm64
```

## Lưu ý khoa học

- Microphone smartphone thường có high-pass filter ở ~20-100 Hz
- Tần số 7.83 Hz (Schumann) hoặc 18.98 Hz nếu có cũng chỉ là dao động cơ học, không phải "tần số ma"
- Khoa học hiện tại chưa xác nhận khái niệm "tần số của ma"

## License

MIT
