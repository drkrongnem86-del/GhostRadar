// Ghost Radar v2.1.0 - "Tesla-style" multi-modal ghost detection
//
// v2.1.0 changes vs v2.0.2 - "Researcher Upgrade" (3 domains):
//   1. Waterfall Spectrogram (Sonic Visualiser / Praat style)
//      Real-time scrolling FFT heatmap, viridis colormap, 0-50Hz range
//   2. EVP auto-capture + Magnetometer EMF meter (GhostTube / SB7 style)
//      On ghost combo: save 10s WAV clip. EMF bar shows magnetometer field
//   3. On-device YOLO dataset export (CVAT / VIA style)
//      Save current frame + bboxes as YOLO format, zip + share
//
// v2.0.2 changes (kept):
//   - DetectionLogger: JSONL log file cho clinical review
//   - UI: nút Xem log / Chia sẻ log / Xóa log + event counter
//   - Foreground Service (Kotlin) giữ camera + mic khi screen off
//
// Audio pipeline (low-freq 0-20Hz):
//   PCM 44.1kHz → 4-stage IIR anti-alias (25Hz)
//                → decimate 441× → 100Hz
//                → 256-point Hann-windowed FFT
//                → 5 band energy (0-1, 1-3, 3-7, 7-15, 15-20Hz)
//                → per-band EMA baseline → alarm nếu band > sensitivity × baseline
//
// Compass (heading):
//   accelerometer (low-pass → gravity) + magnetometer → tilt-compensated heading
//   Mỗi alarm → tạo _Blip tại heading hiện tại
//   Blip fading 8s, blip sáng nhất = "mục tiêu"
//
// Camera + YOLOv8n TFLite Object Detection (Tesla-style):
//   - Live camera preview với âm bản filter (invert RGB) + scanlines
//   - startImageStream → CameraImage (YUV_420_888)
//   - YUV → RGB (BT.601) → resize 640×640 letterbox → normalize [0,1]
//   - TFLite interpreter (yolov8n.tflite) → output [1,84,8400]
//   - Parse 80 COCO class scores + xywh boxes
//   - NMS (IoU=0.5) để loại bỏ trùng lặp
//   - IoU-based tracker gán ID ổn định xuyên frame
//   - Persistence filter: chỉ hiện object thấy ≥3 frame liên tiếp
//   - Bounding box overlay: màu theo COCO class, label + confidence + track ID
//   - "Ghost Combo" alert khi audio alarm + visual detection cùng lúc
//
// Lưu ý khoa học:
//   - Phone mic cắt <20Hz bằng phần cứng
//   - Phone RGB camera không phải IR thật, "âm bản" là giả lập
//   - YOLOv8n trained on COCO 80-class (person/car/dog/cat/bird/...)
//   - TensorFlow Lite 0.11.0 chạy CPU inference (no GPU delegate để giảm APK)

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'native/byte_tracker.dart';
import 'native/camera_x.dart';
import 'native/reid_engine.dart';
import 'native/yuv_processor.dart';
import 'detection_logger.dart';
import 'waterfall_spectrogram.dart';
import 'emf_meter.dart';
import 'evp_recorder.dart';
import 'dataset_exporter.dart';
import 'package:share_plus/share_plus.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const GhostRadarApp());
}

// ====== COCO 80-class labels (cho YOLOv8) ======
// Bảng label theo thứ tự index 0..79 mà YOLOv8 trả về
const List<String> _cocoLabels = <String>[
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
  'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
  'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
  'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

// Màu highlight theo COCO class (Tesla-style neon palette)
const Map<String, Color> _cocoColors = <String, Color>{
  'person': Color(0xFF1FE033), // xanh lá neon
  'bicycle': Color(0xFFFFB300), // vàng
  'car': Color(0xFF00E5FF), // cyan
  'motorcycle': Color(0xFFFF6F00), // cam
  'airplane': Color(0xFF80D8FF),
  'bus': Color(0xFF40C4FF),
  'train': Color(0xFF18FFFF),
  'truck': Color(0xFF00B8D4),
  'boat': Color(0xFF84FFFF),
  'traffic light': Color(0xFFFFEB3B),
  'fire hydrant': Color(0xFFFF5252),
  'stop sign': Color(0xFFFF1744),
  'parking meter': Color(0xFFFFAB91),
  'bench': Color(0xFFBCAAA4),
  'bird': Color(0xFFFF80AB),
  'cat': Color(0xFFFFD180),
  'dog': Color(0xFFFFAB91),
  'horse': Color(0xFFA1887F),
  'sheep': Color(0xFFE0E0E0),
  'cow': Color(0xFF8D6E63),
  'elephant': Color(0xFF90A4AE),
  'bear': Color(0xFF6D4C41),
  'zebra': Color(0xFFEEEEEE),
  'giraffe': Color(0xFFFFB74D),
  'backpack': Color(0xFFB39DDB),
  'umbrella': Color(0xFF9FA8DA),
  'handbag': Color(0xFFF48FB1),
  'tie': Color(0xFF7986CB),
  'suitcase': Color(0xFF4DB6AC),
  'frisbee': Color(0xFF81C784),
  'skis': Color(0xFF4FC3F7),
  'snowboard': Color(0xFF64B5F6),
  'sports ball': Color(0xFFFF8A65),
  'kite': Color(0xFFAED581),
  'baseball bat': Color(0xFFDCE775),
  'baseball glove': Color(0xFFFFD54F),
  'skateboard': Color(0xFFBA68C8),
  'surfboard': Color(0xFF4DD0E1),
  'tennis racket': Color(0xFFA5D6A7),
  'bottle': Color(0xFF90CAF9),
  'wine glass': Color(0xFFCE93D8),
  'cup': Color(0xFF80DEEA),
  'fork': Color(0xFFB0BEC5),
  'knife': Color(0xFF90A4AE),
  'spoon': Color(0xFFBDBDBD),
  'bowl': Color(0xFFFFCC80),
  'banana': Color(0xFFFFF176),
  'apple': Color(0xFFE57373),
  'sandwich': Color(0xFFFFB74D),
  'orange': Color(0xFFFFB74D),
  'broccoli': Color(0xFF66BB6A),
  'carrot': Color(0xFFFF8A65),
  'hot dog': Color(0xFFFFAB91),
  'pizza': Color(0xFFFFCC80),
  'donut': Color(0xFFF8BBD0),
  'cake': Color(0xFFF48FB1),
  'chair': Color(0xFFA1887F),
  'couch': Color(0xFF8D6E63),
  'potted plant': Color(0xFF81C784),
  'bed': Color(0xFFBCAAA4),
  'dining table': Color(0xFFA1887F),
  'toilet': Color(0xFFE0E0E0),
  'tv': Color(0xFF64B5F6),
  'laptop': Color(0xFF42A5F5),
  'mouse': Color(0xFF90CAF9),
  'remote': Color(0xFF7986CB),
  'keyboard': Color(0xFF9FA8DA),
  'cell phone': Color(0xFF40C4FF),
  'microwave': Color(0xFFB0BEC5),
  'oven': Color(0xFF78909C),
  'toaster': Color(0xFFFFCC80),
  'sink': Color(0xFF80CBC4),
  'refrigerator': Color(0xFFB0BEC5),
  'book': Color(0xFFD7CCC8),
  'clock': Color(0xFFFFB74D),
  'vase': Color(0xFFCE93D8),
  'scissors': Color(0xFF90CAF9),
  'teddy bear': Color(0xFFBCAAA4),
  'hair drier': Color(0xFFB0BEC5),
  'toothbrush': Color(0xFF80CBC4),
};

Color _colorForCoco(String label) {
  return _cocoColors[label] ?? const Color(0xFF1FE033);
}

// ====== YOLOv8n detector (TFLite) ======
// Detect 80 class COCO, output xywh center-format, post-process gồm:
//   - lọc theo confidence threshold
//   - decode class id = argmax(class_scores)
//   - NMS theo IoU threshold
class _YoloDetection {
  _YoloDetection({
    required this.bbox,
    required this.classIndex,
    required this.className,
    required this.confidence,
  });
  final Rect bbox; // xyxy trong image space (đã un-letterbox)
  final int classIndex;
  final String className;
  final double confidence;
}

class _YoloDetector {
  _YoloDetector({
    this.inputSize = 640,
    this.confThreshold = 0.40,
    this.iouThreshold = 0.50,
  });
  final int inputSize;
  final double confThreshold;
  final double iouThreshold;

  Interpreter? _interpreter;
  late List<int> _inShape;
  late List<int> _outShape;

  bool get isReady => _interpreter != null;

  /// Use GPU delegate for 2-3x faster inference (Samsung A17 Snapdragon 6100+)
  bool useGpu = true;

  Future<void> load() async {
    Interpreter? interpreter;
    String mode = 'cpu';
    try {
      if (useGpu) {
        final gpuDelegate = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: true, // float16 OK for detection
            // 0 = FAST_SINGLE_ANSWER (best for live camera)
            inferencePreference: 0,
            // 0 = AUTO priority for all
            inferencePriority1: 0,
            inferencePriority2: 0,
            inferencePriority3: 0,
          ),
        );
        final gpuOpts = InterpreterOptions()
          ..threads = 2
          ..addDelegate(gpuDelegate);
        interpreter = await Interpreter.fromAsset(
          'yolov8n.tflite',
          options: gpuOpts,
        );
        mode = 'gpu';
        // ignore: avoid_print
        print('YOLO loaded with GPU delegate');
      }
    } catch (e) {
      // ignore: avoid_print
      print('YOLO GPU load failed, falling back to CPU: $e');
      interpreter = null;
    }
    if (interpreter == null) {
      final opts = InterpreterOptions()
        ..threads = 4
        ..useNnApiForAndroid = false;
      interpreter = await Interpreter.fromAsset(
        'yolov8n.tflite',
        options: opts,
      );
    }
    _interpreter = interpreter;
    _inShape = _interpreter!.getInputTensor(0).shape;
    _outShape = _interpreter!.getOutputTensor(0).shape;
    // ignore: avoid_print
    print('YOLO loaded mode=$mode in=$_inShape out=$_outShape');
  }

  // Chuyển YUV_420_888 frame từ camera thành input float32 NCHW đã normalize.
  // Kết quả: Float32List length = 1*3*inputSize*inputSize, layout NCHW
  //          (vì TFLite YOLOv8 mặc định nhận NCHW float32)
  // Đồng thời trả về scale + offset để un-letterbox khi decode box.
  Float32List preprocessYuv(CameraImage image, double rotationDeg) {
    final int inW = image.width;
    final int inH = image.height;
    final int s = inputSize;

    // Letterbox: scale giữ aspect ratio, pad còn 640x640
    final double scale = math.min(s / inW, s / inH);
    final int newW = (inW * scale).round();
    final int newH = (inH * scale).round();
    final int padX = (s - newW) ~/ 2;
    final int padY = (s - newH) ~/ 2;

    final Float32List out = Float32List(1 * 3 * s * s);

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixStride = uPlane.bytesPerPixel ?? 1;
    final yBytes = yPlane.bytes;
    final uBytes = uPlane.bytes;
    final vBytes = vPlane.bytes;

    // For each output pixel (x, y) in [0, s):
    //  - map to source (sx, sy) in original image
    //  - read Y/U/V, convert YUV→RGB, normalize
    //  - store in NCHW [channel * s * s + y * s + x]
    for (int oy = 0; oy < s; oy++) {
      // Map output y → source y (revert letterbox y)
      final int sy = (((oy - padY) / scale).floor()).clamp(0, inH - 1);
      for (int ox = 0; ox < s; ox++) {
        final int sx = (((ox - padX) / scale).floor()).clamp(0, inW - 1);
        final int yIdx = sy * yRowStride + sx;
        final int uvIdx = (sy ~/ 2) * uvRowStride + (sx ~/ 2) * uvPixStride;
        final int yv = yBytes[yIdx];
        final int uv = uBytes[uvIdx];
        final int vv = vBytes[uvIdx];
        // BT.601 limited range: R = Y + 1.402*(V-128)
        //                       G = Y - 0.344136*(U-128) - 0.714136*(V-128)
        //                       B = Y + 1.772*(U-128)
        double r = yv + 1.402 * (vv - 128);
        double g = yv - 0.344136 * (uv - 128) - 0.714136 * (vv - 128);
        double b = yv + 1.772 * (uv - 128);
        r = r.clamp(0, 255) / 255.0;
        g = g.clamp(0, 255) / 255.0;
        b = b.clamp(0, 255) / 255.0;
        // NCHW layout: index = c*s*s + oy*s + ox
        final int base = oy * s + ox;
        out[0 * s * s + base] = r;
        out[1 * s * s + base] = g;
        out[2 * s * s + base] = b;
      }
    }
    // RotationDeg is reserved (nếu cần transpose sau này)
    return out;
  }

  // Run inference. Returns raw output [1, 84, 8400] (Float32List).
  Float32List runInference(Float32List input) {
    final interp = _interpreter!;
    // Reshape input về [1, 3, s, s]
    final inputReshaped = input.reshape([1, 3, inputSize, inputSize]);
    // Output buffer: shape [1, 84, 8400] (YOLOv8 export mặc định)
    final int outH = _outShape[1]; // 84
    final int outW = _outShape[2]; // 8400
    final Float32List rawOut = Float32List(outH * outW);
    final outReshaped = rawOut.reshape([1, outH, outW]);
    interp.run(inputReshaped, outReshaped);
    return rawOut;
  }

  // Decode output [1, 84, 8400] → List<_YoloDetection> sau NMS
  // Layout: 84 = [x_center, y_center, w, h, class0_score, ..., class79_score]
  //        8400 anchors. output transpose: out[channel, anchor] - cần loop theo anchor
  List<_YoloDetection> postprocess(
    Float32List rawOut,
    int origW,
    int origH, {
    int letterboxPadX = 0,
    int letterboxPadY = 0,
    double letterboxScale = 1.0,
  }) {
    final int outH = _outShape[1]; // 84
    final int outW = _outShape[2]; // 8400
    // Bước 1: filter theo confidence (max class score > threshold)
    final List<_YoloDetection> candidates = <_YoloDetection>[];
    for (int a = 0; a < outW; a++) {
      // Find max class score
      double maxScore = 0;
      int maxClass = 0;
      for (int c = 4; c < outH; c++) {
        final double s = rawOut[c * outW + a];
        if (s > maxScore) {
          maxScore = s;
          maxClass = c - 4;
        }
      }
      if (maxScore < confThreshold) continue;
      // Box: out[0..3, a] = x_center, y_center, w, h (in 640x640 space)
      final double cx = rawOut[0 * outW + a];
      final double cy = rawOut[1 * outW + a];
      final double w = rawOut[2 * outW + a];
      final double h = rawOut[3 * outW + a];
      // Un-letterbox: về original image coords
      final double cx0 = (cx - letterboxPadX) / letterboxScale;
      final double cy0 = (cy - letterboxPadY) / letterboxScale;
      final double w0 = w / letterboxScale;
      final double h0 = h / letterboxScale;
      final double x0 = (cx0 - w0 / 2).clamp(0.0, origW.toDouble());
      final double y0 = (cy0 - h0 / 2).clamp(0.0, origH.toDouble());
      final double x1 = (cx0 + w0 / 2).clamp(0.0, origW.toDouble());
      final double y1 = (cy0 + h0 / 2).clamp(0.0, origH.toDouble());
      candidates.add(_YoloDetection(
        bbox: Rect.fromLTRB(x0, y0, x1, y1),
        classIndex: maxClass,
        className: maxClass >= 0 && maxClass < _cocoLabels.length
            ? _cocoLabels[maxClass]
            : 'cls$maxClass',
        confidence: maxScore,
      ));
    }
    // Bước 2: NMS
    return _nms(candidates, iouThreshold);
  }

  // Compute letterbox params từ image dims
  ({int padX, int padY, double scale}) computeLetterbox(int w, int h) {
    final double scale = math.min(inputSize / w, inputSize / h);
    final int newW = (w * scale).round();
    final int newH = (h * scale).round();
    return (
      padX: (inputSize - newW) ~/ 2,
      padY: (inputSize - newH) ~/ 2,
      scale: scale,
    );
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  // Non-Maximum Suppression theo IoU
  static List<_YoloDetection> _nms(
    List<_YoloDetection> dets,
    double iouThresh,
  ) {
    if (dets.isEmpty) return dets;
    // Sort giảm dần theo confidence
    dets.sort((a, b) => b.confidence.compareTo(a.confidence));
    final List<_YoloDetection> keep = <_YoloDetection>[];
    final List<bool> suppressed = List<bool>.filled(dets.length, false);
    for (int i = 0; i < dets.length; i++) {
      if (suppressed[i]) continue;
      keep.add(dets[i]);
      for (int j = i + 1; j < dets.length; j++) {
        if (suppressed[j]) continue;
        if (dets[i].classIndex != dets[j].classIndex) continue;
        if (_iou(dets[i].bbox, dets[j].bbox) > iouThresh) {
          suppressed[j] = true;
        }
      }
    }
    return keep;
  }

  static double _iou(Rect a, Rect b) {
    final double ix0 = math.max(a.left, b.left);
    final double iy0 = math.max(a.top, b.top);
    final double ix1 = math.min(a.right, b.right);
    final double iy1 = math.min(a.bottom, b.bottom);
    final double iw = math.max(0.0, ix1 - ix0);
    final double ih = math.max(0.0, iy1 - iy0);
    final double inter = iw * ih;
    final double union = a.width * a.height + b.width * b.height - inter;
    if (union <= 0) return 0;
    return inter / union;
  }
}

// ====== IoU-based tracker (single-class-agnostic: match theo bbox) ======
// Mỗi track giữ bbox history + số frame đã thấy. Status tentative → confirmed.
class _Track {
  _Track({
    required this.id,
    required this.className,
    required this.bbox,
    required this.confidence,
    required this.lastSeen,
  });
  final int id;
  String className;
  Rect bbox;
  double confidence;
  DateTime lastSeen;
  int framesSeen = 1;
  int missedFrames = 0;
  // Lịch sử bbox (để vẽ trail fade)
  final List<Rect> history = <Rect>[];
  static const int _maxHistory = 12;
  bool get isConfirmed => framesSeen >= 3;
  void update(Rect newBbox, String newClass, double newConf, DateTime now) {
    bbox = newBbox;
    className = newClass;
    confidence = newConf;
    lastSeen = now;
    framesSeen++;
    missedFrames = 0;
    history.add(newBbox);
    if (history.length > _maxHistory) history.removeAt(0);
  }
  void markMissed() {
    missedFrames++;
  }
}

class _Tracker {
  _Tracker({this.iouMatchThreshold = 0.30, this.maxMissedFrames = 5});
  final double iouMatchThreshold;
  final int maxMissedFrames;
  int _nextId = 1;
  final Map<int, _Track> _tracks = <int, _Track>{};
  // Tracks còn "fresh" để hiển thị
  List<_Track> get active {
    return _tracks.values
        .where((t) => t.missedFrames <= maxMissedFrames)
        .toList();
  }

  // Update: input detections không có ID → match với track hiện tại qua IoU
  // Output: List<_Track> (active tracks sau update, có ID ổn định)
  List<_Track> update(List<_YoloDetection> dets, DateTime now) {
    // 1. Đánh dấu tất cả track chưa match
    final List<int> matchedTrackIds = <int>[];
    final List<bool> detMatched = List<bool>.filled(dets.length, false);

    // 2. Với mỗi detection, tìm track tốt nhất (cùng class + IoU cao nhất)
    for (int di = 0; di < dets.length; di++) {
      final _YoloDetection det = dets[di];
      int? bestTrackId;
      double bestIou = iouMatchThreshold;
      for (final t in _tracks.values) {
        if (matchedTrackIds.contains(t.id)) continue;
        if (t.className != det.className) continue;
        final double iou = _iou(t.bbox, det.bbox);
        if (iou > bestIou) {
          bestIou = iou;
          bestTrackId = t.id;
        }
      }
      if (bestTrackId != null) {
        final t = _tracks[bestTrackId]!;
        t.update(det.bbox, det.className, det.confidence, now);
        matchedTrackIds.add(bestTrackId);
        detMatched[di] = true;
      }
    }

    // 3. Detection chưa match → tạo track mới
    for (int di = 0; di < dets.length; di++) {
      if (detMatched[di]) continue;
      final det = dets[di];
      final t = _Track(
        id: _nextId++,
        className: det.className,
        bbox: det.bbox,
        confidence: det.confidence,
        lastSeen: now,
      );
      t.history.add(det.bbox);
      _tracks[t.id] = t;
    }

    // 4. Track không match → tăng missed
    for (final t in _tracks.values) {
      if (!matchedTrackIds.contains(t.id)) {
        t.markMissed();
      }
    }

    // 5. Xóa track quá miss
    _tracks.removeWhere((_, t) => t.missedFrames > maxMissedFrames);

    return active;
  }

  static double _iou(Rect a, Rect b) {
    final double ix0 = math.max(a.left, b.left);
    final double iy0 = math.max(a.top, b.top);
    final double ix1 = math.min(a.right, b.right);
    final double iy1 = math.min(a.bottom, b.bottom);
    final double iw = math.max(0.0, ix1 - ix0);
    final double ih = math.max(0.0, iy1 - iy0);
    final double inter = iw * ih;
    final double union = a.width * a.height + b.width * b.height - inter;
    if (union <= 0) return 0;
    return inter / union;
  }
}

class GhostRadarApp extends StatelessWidget {
  const GhostRadarApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ghost Radar',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF050505),
        primaryColor: const Color(0xFF1FE033),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF1FE033),
          secondary: Color(0xFFFF3B3B),
        ),
        useMaterial3: true,
      ),
      home: const RadarScreen(),
    );
  }
}

class _Blip {
  _Blip({
    required this.angleDeg,
    required this.energy,
    required this.bandIndex,
    required this.detectedAt,
  });
  final double angleDeg;
  final double energy;
  final int bandIndex;
  final DateTime detectedAt;

  double get ageSeconds =>
      DateTime.now().difference(detectedAt).inMilliseconds / 1000.0;
  double get fade => (1.0 - (ageSeconds / 8.0)).clamp(0.0, 1.0);
  double get score => energy * fade;
}

class _DetectedObj {
  _DetectedObj({
    required this.label,
    required this.confidence,
    required this.boundingBox, // Rect in image pixel coordinates
    required this.detectedAt,
    this.trackId,
    this.trackHistory,
    this.reidentified = false,
  });
  final String label;
  final double confidence;
  final Rect boundingBox;
  final DateTime detectedAt;
  final int? trackId;
  final List<Rect>? trackHistory;
  final bool reidentified;

  double get ageSeconds =>
      DateTime.now().difference(detectedAt).inMilliseconds / 1000.0;
  double get fade => (1.0 - (ageSeconds / 3.0)).clamp(0.0, 1.0);
}

class RadarScreen extends StatefulWidget {
  const RadarScreen({super.key});

  @override
  State<RadarScreen> createState() => _RadarScreenState();
}

class _RadarScreenState extends State<RadarScreen>
    with SingleTickerProviderStateMixin, WidgetsBindingObserver {
  // ====== Audio: anti-alias + decimation ======
  static const double _lpA = 157.0796 / (44100 + 157.0796);
  double _lp1 = 0, _lp2 = 0, _lp3 = 0, _lp4 = 0;
  int _decCount = 0;
  static const int _decFactor = 441;

  // ====== Low-rate circular buffer ======
  static const int _fftSize = 256;
  static const double _binHz = 100 / _fftSize;
  final Float64List _lrBuffer = Float64List(_fftSize);
  int _lrFilled = 0;
  int _lrWriteIdx = 0;
  bool _bufferReady = false;

  late final Float64List _fftRe;
  late final Float64List _fftIm;
  late final Float64List _hann;
  late final Float64List _windowed;

  // ====== 5 bands ======
  static const List<String> _bandNames = <String>[
    '0–1 Hz',
    '1–3 Hz',
    '3–7 Hz',
    '7–15 Hz',
    '15–20 Hz',
  ];
  static const List<List<int>> _bandBins = <List<int>>[
    <int>[0, 1, 2],
    <int>[3, 4, 5, 6, 7],
    <int>[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    <int>[
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37
    ],
    <int>[38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
  ];
  final List<double> _bandEnergy = List<double>.filled(5, 0);
  final List<double> _bandBaseline = List<double>.filled(5, 1.0);
  static const double _emaAlpha = 0.92;
  int _baselineWarmupLeft = 30;

  // ====== Compass (manual heading) ======
  double _heading = 0;
  bool _hasCompass = false;
  double _gx = 0, _gy = 0, _gz = 0;
  double _mx = 0, _my = 0, _mz = 0;
  StreamSubscription<AccelerometerEvent>? _accelSub;
  StreamSubscription<MagnetometerEvent>? _magSub;

  // ====== Camera (CameraX native v2.0+) + YOLOv8n TFLite (GPU) + ByteTrack + ReID =====
  GhostRadarCamera? _ghostCamera;
  int? _cameraTextureId;
  YuvPreprocessor? _yuvProc;
  final ByteTracker _byteTracker = ByteTracker();
  ReIdEngine? _reid;
  PersonFeatureExtractor? _featureExtractor;
  int _reidMatched = 0; // counter for UI
  StreamSubscription<CameraFrame>? _frameSub;
  String _cameraError = '';
  _YoloDetector? _yolo;
  bool _isDetecting = false;
  int _frameCounter = 0;
  static const int _detectEveryNFrames = 3; // xử lý mỗi 3 frame (GPU nhanh hơn)

  // Confirmed tracks (giữ ~3 giây history) - dùng để render
  final List<_DetectedObj> _detections = <_DetectedObj>[];
  static const double _detectionLifetimeSec = 3.0;
  // Map: label → count (đếm nhanh các track confirmed)
  final Map<String, int> _objectCounts = <String, int>{};
  // Top detection
  String _topLabel = '--';
  double _topConfidence = 0;

  // ====== Blips ======
  final List<_Blip> _blips = <_Blip>[];
  static const int _maxBlips = 24;
  static const double _blipLifetimeSec = 8.0;
  _Blip? _target;

  // ====== Detection / Alarm ======
  bool _isAlarming = false;
  int _alarmCount = 0;
  String _lastAlarmSummary = '';
  bool _alarmMuted = false;
  Timer? _alarmHapticTimer;
  bool _ghostCombo = false; // audio + visual cùng alarm

  // ====== Logging throttle (v2.0.2) ======
  // Track per-id last-log time to avoid flooding JSONL with every frame
  static const Duration _detectionLogInterval = Duration(seconds: 8);
  final Map<int, DateTime> _lastDetectionLog = <int, DateTime>{};
  int _loggedDetectionCount = 0;

  // ====== v2.1.0 Researcher Upgrade ======
  // Keys to control child widgets from this state (call pushFrame, push, etc.)
  final GlobalKey<State<WaterfallSpectrogram>> _waterfallKey =
      GlobalKey<State<WaterfallSpectrogram>>();
  final GlobalKey<State<EmfMeter>> _emfKey = GlobalKey<State<EmfMeter>>();
  EvpRecorder? _evpRecorder;
  int _evpCount = 0;
  int _datasetCount = 0;

  // ====== UI state ======
  bool _isScanning = false;
  String _statusText = 'CHƯA KHỞI ĐỘNG';
  String _noteText =
      'Bấm BẤT ĐẦU QUÉT. Mic dò 0-20Hz, camera YOLOv8n 80-class COCO real-time + NMS + tracking. Cùng trigger → GHOST COMBO 🔴';
  double _level = 0.0;
  double _angle = 0.0;
  double _sensitivity = 2.5;
  String _dominantBandName = '--';
  double _dominantBandHz = 0;
  int _comboCount = 0;

  // ====== Audio plumbing ======
  AudioRecorder? _recorder;
  StreamSubscription<Uint8List>? _streamSub;
  Timer? _analysisTimer;
  late final Ticker _ticker;

  // ====== Foreground Service bridge (v2.0.2) ======
  static const MethodChannel _fgsChannel =
      MethodChannel('ghost_radar/fgs');
  DateTime _lastFgsUpdate = DateTime.fromMillisecondsSinceEpoch(0);

  static const List<String> _compassDirs = <String>[
    'B', 'BĐ', 'Đ', 'ĐN', 'N', 'NT', 'T', 'TB',
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _ticker = Ticker(_onTick)..start();

    _hann = Float64List(_fftSize);
    for (int i = 0; i < _fftSize; i++) {
      _hann[i] = 0.5 * (1 - math.cos(2 * math.pi * i / (_fftSize - 1)));
    }
    _fftRe = Float64List(_fftSize);
    _fftIm = Float64List(_fftSize);
    _windowed = Float64List(_fftSize);

    _startSensors();
    _initCamera();
    _initObjectDetector();

    // v2.1.0: load existing dataset count on app start
    DatasetExporter.getSampleCount().then((n) {
      if (mounted) setState(() => _datasetCount = n);
    });
    DatasetExporter.writeClassesTxt();
  }

  void _onTick(Duration elapsed) {
    if (!mounted) return;
    setState(() {
      if (_isScanning) {
        _angle = (_angle + 2.5) % 360.0;
      }
      // Dọn detection cũ mỗi tick
      _detections.removeWhere((d) => d.ageSeconds > _detectionLifetimeSec);
      if (_detections.isEmpty) {
        _objectCounts.clear();
        _topLabel = '--';
        _topConfidence = 0;
      }
      // Kiểm tra ghost combo
      _checkGhostCombo();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _ticker.dispose();
    _analysisTimer?.cancel();
    _alarmHapticTimer?.cancel();
    _streamSub?.cancel();
    _accelSub?.cancel();
    _magSub?.cancel();
    _frameSub?.cancel();
    _yolo?.close();
    _reid?.close();
    _recorder?.dispose();
    _ghostCamera?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _ghostCamera?.stopCamera();
    } else if (state == AppLifecycleState.resumed) {
      if (_cameraTextureId != null) {
        _ghostCamera?.startCamera(textureId: _cameraTextureId!);
      } else {
        _initCamera();
      }
    }
  }

  // ====== Sensors ======
  void _startSensors() {
    try {
      _accelSub = accelerometerEventStream(
        samplingPeriod: const Duration(milliseconds: 100),
      ).listen((AccelerometerEvent e) {
        const double a = 0.15;
        _gx = a * e.x + (1 - a) * _gx;
        _gy = a * e.y + (1 - a) * _gy;
        _gz = a * e.z + (1 - a) * _gz;
      }, onError: (Object _) {});
    } catch (_) {}
    try {
      _magSub = magnetometerEventStream(
        samplingPeriod: const Duration(milliseconds: 100),
      ).listen((MagnetometerEvent e) {
        _mx = e.x;
        _my = e.y;
        _mz = e.z;
        _updateHeading();
      }, onError: (Object _) {});
    } catch (_) {}
  }

  void _updateHeading() {
    final double gNorm = math.sqrt(_gx * _gx + _gy * _gy + _gz * _gz);
    if (gNorm < 0.1) return;
    final double ngx = _gx / gNorm;
    final double ngy = _gy / gNorm;
    final double ngz = _gz / gNorm;

    final double mNorm = math.sqrt(_mx * _mx + _my * _my + _mz * _mz);
    if (mNorm < 0.1) return;
    final double nmx = _mx / mNorm;
    final double nmy = _my / mNorm;
    final double nmz = _mz / mNorm;

    final double ex = nmy * ngz - nmz * ngy;
    final double ey = nmz * ngx - nmx * ngz;
    final double ez = nmx * ngy - nmy * ngx;

    final double ny = ngz * ex - ngx * ez;

    double headingRad = math.atan2(ey, ny);
    double headingDeg = headingRad * 180 / math.pi;
    if (headingDeg < 0) headingDeg += 360;

    if (headingDeg.isFinite && mounted) {
      setState(() {
        _heading = headingDeg;
        _hasCompass = true;
      });
    }

    // v2.1.0: push raw magnetometer reading to EMF meter (raw µT)
    // sensors_plus returns magnetometer in microtesla already
    (_emfKey.currentState as dynamic)?.push(_mx, _my, _mz);
  }

  // ====== Camera (CameraX native v2.0+) ======
  Future<void> _initCamera() async {
    try {
      final camStatus = await Permission.camera.request();
      if (!camStatus.isGranted) {
        if (mounted) setState(() => _cameraError = 'Không có quyền camera');
        return;
      }
      // Init YUV preprocessor
      _yuvProc = YuvPreprocessor.create(inputSize: 640);
      // Init native CameraX bridge
      _ghostCamera = GhostRadarCamera(inputSize: 640);
      // Create Flutter texture for preview
      final textureId = await _ghostCamera!.createTexture();
      _cameraTextureId = textureId;
      // Start camera (preview + frame stream)
      await _ghostCamera!.startCamera(textureId: textureId);
      _frameSub = _ghostCamera!.frames.listen(_onCameraFrame);
      if (!mounted) return;
      setState(() {
        _cameraError = '';
      });
    } catch (e) {
      if (mounted) setState(() => _cameraError = 'Lỗi CameraX: $e');
    }
  }

  // ====== YOLOv8n TFLite Object Detector + Re-ID (v2.0.1) ======
  Future<void> _initObjectDetector() async {
    try {
      _yolo = _YoloDetector(
        inputSize: 640,
        confThreshold: 0.40,
        iouThreshold: 0.50,
      );
      await _yolo!.load();
      // Re-ID: MobileNetV2 feature extractor
      _reid = ReIdEngine();
      try {
        await _reid!.load();
        _featureExtractor = PersonFeatureExtractor(_reid!);
        // ignore: avoid_print
        print('Re-ID ready (feature dim ${_reid!.featDim})');
      } catch (e) {
        // ignore: avoid_print
        print('Re-ID disabled (load failed): $e');
        _reid = null;
      }
    } catch (e) {
      if (mounted) {
        setState(() => _cameraError = 'YOLO load fail: $e');
      }
    }
  }

  void _onCameraFrame(CameraFrame frame) {
    if (_yolo == null || !_yolo!.isReady || _isDetecting) return;
    if (_yuvProc == null) return;
    _frameCounter++;
    if (_frameCounter % _detectEveryNFrames != 0) return;
    _isDetecting = true;
    _runDetectionV2(frame);
  }

  /// v2.0 detection pipeline:
  ///   1. Preprocess YUV -> NCHW Float32 (Dart fallback for now; native JNI hooked up)
  ///   2. TFLite GPU inference
  ///   3. Decode + NMS (xywh -> xyxy)
  ///   4. ByteTrack + Kalman update
  Future<void> _runDetectionV2(CameraFrame frame) async {
    try {
      final yolo = _yolo;
      final proc = _yuvProc;
      if (yolo == null || !yolo.isReady || proc == null) {
        _isDetecting = false;
        return;
      }
      // 1. Preprocess (Dart-side; native JNI hooked for future use)
      proc.preprocess(
        yPlane: frame.y,
        uPlane: frame.u,
        vPlane: frame.v,
        yRowStride: frame.yRowStride,
        uvRowStride: frame.uvRowStride,
        uvPixelStride: frame.uvPixelStride,
        width: frame.width,
        height: frame.height,
      );
      final lb = proc.letterbox;
      // 2. Inference (Float32List NCHW; runInference does the reshape)
      final raw = yolo.runInference(proc.outputBuffer);
      // 3. Postprocess: decode + NMS
      final dets = yolo.postprocess(
        raw,
        frame.width,
        frame.height,
        letterboxPadX: lb.padX,
        letterboxPadY: lb.padY,
        letterboxScale: lb.scale,
      );
      // 4. ByteTrack + Kalman
      final btDets = dets
          .map((d) => YoloDetectionForTracker(
                bbox: BoxInternal.fromRect(d.bbox),
                className: d.className,
                confidence: d.confidence,
              ))
          .toList();
      final confirmed = _byteTracker.update(btDets);
      // 4b. Re-ID: extract features for confirmed person tracks
      if (_reid != null && _featureExtractor != null) {
        for (final t in confirmed) {
          if (!t.isConfirmed) continue;
          if (t.className != 'person') continue;
          if (t.feature != null) continue; // already have feature
          if (t.hits % 5 != 0) continue; // extract every 5 frames
          final feat = _featureExtractor!.extract(
            frame,
            t.lastBbox.x1,
            t.lastBbox.y1,
            t.lastBbox.x2,
            t.lastBbox.y2,
          );
          if (feat != null) {
            t.feature = feat;
          }
        }
        // 4c. For newly-created tracks (age 0, hits 1), try to match with gallery
        for (final t in confirmed) {
          if (t.age > 0) continue; // not a new track
          if (t.className != 'person') continue;
          // Re-ID: try match with gallery
          if (t.feature == null) {
            final feat = _featureExtractor!.extract(
              frame,
              t.lastBbox.x1,
              t.lastBbox.y1,
              t.lastBbox.x2,
              t.lastBbox.y2,
            );
            if (feat != null) {
              t.feature = feat;
              final matchId = _reid!.findMatch(feat, t.className);
              if (matchId != null) {
                t.reidentified = true;
                t.originalId = t.id;
                t.id = matchId; // re-assign original ID
                _reidMatched++;
                // ignore: avoid_print
                print('Re-ID matched: new track -> ID $matchId');
              }
            }
          }
        }
        // 4d. Save to gallery for tracks that have been missed for a while
        for (final t in confirmed) {
          if (t.isConfirmed && t.feature != null && t.timeSinceUpdate >= 3) {
            // Track is about to be lost - save feature to gallery
            _reid!.addToGallery(t, t.feature);
          }
        }
      }
      // 5. Map to UI
      _onTracksReady(confirmed, Size(frame.width.toDouble(), frame.height.toDouble()));
    } catch (e) {
      // ignore: avoid_print
      print('Detection error: $e');
    } finally {
      _isDetecting = false;
    }
  }

  void _onTracksReady(List<TrackedObject> tracks, Size imageSize) {
    if (!mounted) return;
    final now = DateTime.now();
    _detections.removeWhere((d) => d.ageSeconds > _detectionLifetimeSec);
    final counts = <String, int>{};
    String topLabel = '--';
    double topConf = 0;
    for (final t in tracks) {
      if (!t.isConfirmed) continue;
      counts[t.className] = (counts[t.className] ?? 0) + 1;
      if (t.confidence > topConf) {
        topConf = t.confidence;
        topLabel = t.className;
      }
      _detections.add(_DetectedObj(
        label: t.className,
        confidence: t.confidence,
        boundingBox: t.lastBbox.toRect(),
        detectedAt: now,
        trackId: t.id,
        reidentified: t.reidentified,
      ));

      // v2.0.2: log detection throttled theo track_id
      // Tránh ghi 1 track 30 lần/giây. Chỉ log lần đầu hoặc sau _detectionLogInterval.
      final lastLogged = _lastDetectionLog[t.id];
      final bool isNewTrack = lastLogged == null;
      final bool isStale =
          lastLogged != null && now.difference(lastLogged) >= _detectionLogInterval;
      if (isNewTrack || isStale) {
        _lastDetectionLog[t.id] = now;
        final bbox = t.lastBbox.toRect();
        DetectionLogger.instance.logDetection(
          className: t.className,
          confidence: t.confidence,
          bbox: <double>[
            bbox.left,
            bbox.top,
            bbox.right,
            bbox.bottom,
          ],
          trackId: t.id,
          headingDeg: _heading,
          lowfreqHz: _dominantBandHz > 0 ? _dominantBandHz : null,
        );
        _loggedDetectionCount++;
      }
    }
    // Cleanup: bỏ last-log timestamps cho tracks đã biến mất >30s
    _lastDetectionLog.removeWhere((_, ts) => now.difference(ts) > const Duration(seconds: 30));
    if (_detections.length > 60) {
      _detections.removeRange(0, _detections.length - 60);
    }
    setState(() {
      _objectCounts
        ..clear()
        ..addAll(counts);
      _topLabel = topLabel;
      _topConfidence = topConf;
    });

    // v2.0.2: push counts to Foreground Service notification
    unawaited(_updateForegroundService(_detections.length));
  }

  void _checkGhostCombo() {
    final bool hasAudioAlarm = _isAlarming;
    final bool hasVisualDetection = _detections.isNotEmpty;
    final bool newCombo = hasAudioAlarm && hasVisualDetection;
    if (newCombo && !_ghostCombo) {
      _comboCount++;
      // Combo haptic mạnh hơn
      HapticFeedback.heavyImpact();
      SystemSound.play(SystemSoundType.click);
      // Log ghost combo (v2.0.2)
      if (_detections.isNotEmpty) {
        final top = _detections.reduce((a, b) => a.confidence > b.confidence ? a : b);
        DetectionLogger.instance.logGhostCombo(
          className: top.label,
          confidence: top.confidence,
          trackId: top.trackId ?? -1,
          headingDeg: _heading,
        );
        // v2.1.0: auto-capture EVP snapshot (10s WAV) on ghost combo
        _captureEvpSnapshot(
          className: top.label,
          confidence: top.confidence,
          trackId: top.trackId,
          headingDeg: _heading,
        );
      }
    }
    _ghostCombo = newCombo;
  }

  /// v2.1.0: Save current 10s EVP rolling buffer to a WAV file.
  Future<void> _captureEvpSnapshot({
    required String className,
    required double confidence,
    required int? trackId,
    required double headingDeg,
  }) async {
    final evp = _evpRecorder;
    if (evp == null || !evp.hasFullBuffer) return;
    try {
      final path = await evp.saveCurrentSnapshot(
        heading: headingDeg,
        className: className,
        confidence: confidence,
        trackId: trackId,
      );
      if (mounted) {
        setState(() {
          _evpCount++;
        });
      }
      // ignore: avoid_print
      print('EVP saved: $path');
    } catch (e) {
      // ignore: avoid_print
      print('EVP save failed: $e');
    }
  }

  // ====== Compass helpers ======
  String _angleToCompass(double deg) {
    final double d = deg % 360;
    final int idx = ((d + 22.5) ~/ 45) % 8;
    return '${d.toStringAsFixed(0)}° ${_compassDirs[idx]}';
  }

  // ====== Permissions ======
  Future<bool> _ensureMicPermission() async {
    final p = await Permission.microphone.request();
    return p.isGranted;
  }

  // ====== Start / Stop scan ======
  Future<void> startScan() async {
    if (_isScanning) return;
    final ok = await _ensureMicPermission();
    if (!ok) {
      setState(() {
        _statusText = 'KHÔNG CÓ QUYỀN MICRO';
        _noteText =
            'Vào Cài đặt → Ứng dụng → Ghost Radar → Quyền → cấp Microphone.';
      });
      return;
    }

    _recorder ??= AudioRecorder();
    try {
      final stream = await _recorder!.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 44100,
          numChannels: 1,
          autoGain: false,
          echoCancel: false,
          noiseSuppress: false,
        ),
      );
      _streamSub = stream.listen(
        _onAudioData,
        onError: (Object e) {
          if (!mounted) return;
          setState(() {
            _statusText = 'LỖI AUDIO STREAM';
            _noteText = '$e';
          });
        },
        cancelOnError: false,
      );
    } catch (e) {
      setState(() {
        _statusText = 'KHÔNG MỞ ĐƯỢC MICRO';
        _noteText = 'Lỗi: $e';
      });
      return;
    }

    _lp1 = _lp2 = _lp3 = _lp4 = 0;
    _decCount = 0;
    _lrFilled = 0;
    _lrWriteIdx = 0;
    _bufferReady = false;
    for (int i = 0; i < 5; i++) {
      _bandEnergy[i] = 0;
      _bandBaseline[i] = 1.0;
    }
    _baselineWarmupLeft = 30;
    _isAlarming = false;
    _alarmCount = 0;
    _lastAlarmSummary = '';
    _alarmMuted = false;
    _blips.clear();
    _target = null;
    _dominantBandName = '--';
    _dominantBandHz = 0;
    _comboCount = 0;
    _ghostCombo = false;
    _detections.clear();
    _objectCounts.clear();
    _topLabel = '--';

    _analysisTimer?.cancel();
    _analysisTimer = Timer.periodic(
      const Duration(milliseconds: 200),
      (_) => _analyzeBuffer(),
    );

    setState(() {
      _isScanning = true;
      _statusText = 'ĐANG QUÉT 0–20 Hz + ML';
      _noteText = _hasCompass
          ? 'Mic dò âm thanh hạ tần + camera dò đối tượng real-time. Cả hai cùng trigger = GHOST COMBO. Cầm thẳng đứng, xoay người chậm để dò hướng.'
          : 'Mic dò 0-20Hz + camera dò object. Không đọc được la bàn — hướng blip sẽ không chính xác.';
    });

    // v2.0.2: Promote to Foreground Service so Android keeps us alive
    // when the screen turns off during a long clinical review session.
    unawaited(_startForegroundService());

    // v2.1.0: Init EVP rolling recorder (10s @ 100Hz = 1000 samples)
    _evpRecorder = EvpRecorder(lookbackSec: 10, sampleRate: 100);
    // Clear waterfall history on new scan
    (_waterfallKey.currentState as dynamic)?.clear?.call();
  }

  Future<void> stopScan() async {
    if (!_isScanning) return;
    _analysisTimer?.cancel();
    _analysisTimer = null;
    _alarmHapticTimer?.cancel();
    _alarmHapticTimer = null;
    try {
      await _streamSub?.cancel();
    } catch (_) {}
    _streamSub = null;
    try {
      await _recorder?.stop();
    } catch (_) {}
    if (!mounted) return;
    setState(() {
      _isScanning = false;
      _isAlarming = false;
      _statusText = 'ĐÃ DỪNG';
      _noteText =
          'Đã dừng. Báo động: $_alarmCount lần · Blip: ${_blips.length} · Ghost combo: $_comboCount';
    });

    // v2.0.2: Stop foreground service
    unawaited(_stopForegroundService());

    // v2.1.0: Release EVP recorder
    _evpRecorder?.dispose();
    _evpRecorder = null;
  }

  void toggleMute() => setState(() => _alarmMuted = !_alarmMuted);

  // ====== Foreground Service helpers (v2.0.2) ======
  Future<void> _startForegroundService() async {
    // Android 13+: request POST_NOTIFICATIONS first
    try {
      final notifStatus = await Permission.notification.request();
      if (!notifStatus.isGranted) {
        // continue anyway - service can run, just no visible notification
      }
    } catch (_) {}
    try {
      await _fgsChannel.invokeMethod('startForegroundService');
    } catch (e) {
      // Best-effort: log but don't fail scan
      // ignore: avoid_print
      print('GhostRadar FGS start failed: $e');
    }
  }

  Future<void> _stopForegroundService() async {
    try {
      await _fgsChannel.invokeMethod('stopForegroundService');
    } catch (_) {}
  }

  Future<void> _updateForegroundService(int tracks) async {
    // Throttle: chỉ update mỗi 2s để tránh spam notification
    final now = DateTime.now();
    if (now.difference(_lastFgsUpdate) < const Duration(seconds: 2)) return;
    _lastFgsUpdate = now;
    try {
      await _fgsChannel.invokeMethod('updateForegroundService', <String, dynamic>{
        'tracks': tracks,
        'logs': _loggedDetectionCount,
      });
    } catch (_) {}
  }

  // ====== Audio callback ======
  void _onAudioData(Uint8List bytes) {
    if (bytes.isEmpty) return;
    final List<int> ints = _recorder!.convertBytesToInt16(bytes);
    if (ints.isEmpty) return;
    for (int i = 0; i < ints.length; i++) {
      final double x = ints[i] / 32768.0;
      final double y1 = _lpA * x + (1 - _lpA) * _lp1;
      _lp1 = y1;
      final double y2 = _lpA * y1 + (1 - _lpA) * _lp2;
      _lp2 = y2;
      final double y3 = _lpA * y2 + (1 - _lpA) * _lp3;
      _lp3 = y3;
      final double y4 = _lpA * y3 + (1 - _lpA) * _lp4;
      _lp4 = y4;

      _decCount++;
      if (_decCount >= _decFactor) {
        _decCount = 0;
        _lrBuffer[_lrWriteIdx] = y4;
        _lrWriteIdx = (_lrWriteIdx + 1) % _fftSize;
        if (_lrFilled < _fftSize) {
          _lrFilled++;
          if (_lrFilled == _fftSize) _bufferReady = true;
        }
        // v2.1.0: push to EVP rolling buffer (only the 100Hz decimated output,
        // so EVP WAV = 100Hz mono 16-bit, same as our spectrogram data)
        _evpRecorder?.pushSample(y4);
      }
    }
  }

  void _linearize(Float64List dst) {
    int idx = _lrWriteIdx;
    for (int i = 0; i < _fftSize; i++) {
      dst[i] = _lrBuffer[idx];
      idx = (idx + 1) % _fftSize;
    }
  }

  void _fft() {
    final n = _fftSize;
    int j = 0;
    for (int i = 0; i < n; i++) {
      if (i < j) {
        final double tr = _fftRe[i];
        _fftRe[i] = _fftRe[j];
        _fftRe[j] = tr;
        final double ti = _fftIm[i];
        _fftIm[i] = _fftIm[j];
        _fftIm[j] = ti;
      }
      int m = n >> 1;
      while (m >= 1 && j >= m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }
    for (int size = 2; size <= n; size <<= 1) {
      final int half = size >> 1;
      final double tablestep = -2 * math.pi / size;
      for (int i = 0; i < n; i += size) {
        for (int k = 0; k < half; k++) {
          final double angle = k * tablestep;
          final double cos = math.cos(angle);
          final double sin = math.sin(angle);
          final int a = i + k;
          final int b = a + half;
          final double tre = _fftRe[b] * cos - _fftIm[b] * sin;
          final double tim = _fftRe[b] * sin + _fftIm[b] * cos;
          _fftRe[b] = _fftRe[a] - tre;
          _fftIm[b] = _fftIm[a] - tim;
          _fftRe[a] += tre;
          _fftIm[a] += tim;
        }
      }
    }
  }

  // ====== Analysis ======
  void _analyzeBuffer() {
    if (!_isScanning || !_bufferReady) return;

    _linearize(_windowed);

    double mean = 0;
    for (int i = 0; i < _fftSize; i++) {
      mean += _windowed[i];
    }
    mean /= _fftSize;
    for (int i = 0; i < _fftSize; i++) {
      _fftRe[i] = (_windowed[i] - mean) * _hann[i];
      _fftIm[i] = 0;
    }

    _fft();

    final int nyquistBin = _fftSize >> 1;
    final List<double> mag2 = List<double>.filled(5, 0);
    // v2.1.0: also collect per-bin magnitude for waterfall spectrogram
    final List<double> magPerBin = List<double>.filled(nyquistBin, 0);
    int peakBin = -1;
    double peakVal = 0;
    for (int b = 1; b < nyquistBin; b++) {
      final double re = _fftRe[b];
      final double im = _fftIm[b];
      final double m2 = re * re + im * im;
      magPerBin[b] = m2; // store raw magnitude squared for waterfall
      for (int band = 0; band < 5; band++) {
        if (_bandBins[band].contains(b)) {
          mag2[band] += m2;
          break;
        }
      }
      if (b <= 51 && m2 > peakVal) {
        peakVal = m2;
        peakBin = b;
      }
    }

    for (int i = 0; i < 5; i++) {
      _bandEnergy[i] = mag2[i] / _fftSize;
    }

    for (int i = 0; i < 5; i++) {
      if (_baselineWarmupLeft > 0) {
        _bandBaseline[i] = math.max(_bandEnergy[i], 1e-12);
      } else {
        _bandBaseline[i] =
            _emaAlpha * _bandBaseline[i] + (1 - _emaAlpha) * _bandEnergy[i];
        if (_bandBaseline[i] < 1e-12) _bandBaseline[i] = 1e-12;
      }
    }
    if (_baselineWarmupLeft > 0) _baselineWarmupLeft--;

    int bestBand = -1;
    double bestRatio = 0;
    double bestHz = 0;
    for (int i = 0; i < 5; i++) {
      final double r = _bandEnergy[i] / _bandBaseline[i];
      if (r > bestRatio) {
        bestRatio = r;
        bestBand = i;
        if (peakBin >= _bandBins[i].first && peakBin <= _bandBins[i].last) {
          bestHz = peakBin * _binHz;
        } else {
          bestHz = ((_bandBins[i].first + _bandBins[i].last) / 2.0) * _binHz;
        }
      }
    }

    final double level01 =
        ((bestRatio - 1.0) / math.max(0.5, _sensitivity - 1.0)).clamp(0.0, 1.0);

    String status;
    String note;
    bool newAlarm = false;
    if (_baselineWarmupLeft > 0) {
      status = '⏳ ĐANG THU BASELINE';
      note =
          'Còn ${(_baselineWarmupLeft * 0.2).toStringAsFixed(0)}s để thiết lập đường nền. Hãy giữ yên tĩnh.';
    } else if (bestRatio >= _sensitivity) {
      status = '🚨 BÁO ĐỘNG ${_bandNames[bestBand]}';
      note =
          'Phát hiện năng lượng ${bestRatio.toStringAsFixed(1)}× đường nền '
          'trong dải ${_bandNames[bestBand]} (≈ ${bestHz.toStringAsFixed(1)} Hz).';
      newAlarm = true;
    } else if (bestRatio >= _sensitivity * 0.6) {
      status = '⚠ CÓ TÍN HIỆU';
      note =
          'Tín hiệu nhô lên ở ${_bandNames[bestBand]} '
          '(${(bestRatio).toStringAsFixed(1)}× nền, ≈ ${bestHz.toStringAsFixed(1)} Hz).';
    } else {
      status = '✓ NỀN THẤP';
      note =
          'Yên tĩnh. Đang quét 0–20 Hz, ngưỡng = ${_sensitivity.toStringAsFixed(1)}× nền.';
    }

    _blips.removeWhere((b) => b.ageSeconds > _blipLifetimeSec);
    _Blip? best;
    for (final b in _blips) {
      if (best == null || b.score > best.score) best = b;
    }
    _target = best;

    if (!mounted) return;
    setState(() {
      _level = level01;
      _statusText = status;
      _noteText = note;
      _dominantBandName = bestBand >= 0 ? _bandNames[bestBand] : '--';
      _dominantBandHz = bestHz;
    });

    if (newAlarm) {
      _triggerAlarm(bestBand, bestHz, bestRatio);
    } else if (_isAlarming) {
      _stopAlarm();
    }

    // v2.1.0: push FFT magnitude frame to waterfall spectrogram
    // (skip first few frames during baseline warmup to avoid noise)
    if (_baselineWarmupLeft < 25) {
      (_waterfallKey.currentState as dynamic)?.pushFrame(magPerBin);
    }
  }

  void _triggerAlarm(int band, double hz, double ratio) {
    if (!_isAlarming) {
      _isAlarming = true;
      _alarmCount += 1;
      _lastAlarmSummary =
          '${_bandNames[band]} · ≈ ${hz.toStringAsFixed(1)} Hz · ${ratio.toStringAsFixed(1)}× nền';
      // v2.0.2: log first audio alarm event
      DetectionLogger.instance.logAudioAlarm(
        bandIndex: band,
        hz: hz,
        ratio: ratio,
        headingDeg: _heading,
      );
      _alarmHapticTimer?.cancel();
      _alarmHapticTimer = Timer.periodic(
        const Duration(milliseconds: 350),
        (_) {
          if (!_isAlarming || !mounted) return;
          if (!_alarmMuted) {
            HapticFeedback.heavyImpact();
            SystemSound.play(SystemSoundType.click);
          }
        },
      );
    } else {
      _lastAlarmSummary =
          '${_bandNames[band]} · ≈ ${hz.toStringAsFixed(1)} Hz · ${ratio.toStringAsFixed(1)}× nền';
    }

    final double energy01 =
        (ratio / math.max(2.0, _sensitivity)).clamp(0.0, 1.0);
    _blips.add(_Blip(
      angleDeg: _heading,
      energy: energy01,
      bandIndex: band,
      detectedAt: DateTime.now(),
    ));
    if (_blips.length > _maxBlips) {
      _blips.removeRange(0, _blips.length - _maxBlips);
    }
  }

  void _stopAlarm() {
    _isAlarming = false;
    _alarmHapticTimer?.cancel();
    _alarmHapticTimer = null;
  }

  // ====== UI ======
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('GHOST RADAR · 0–20 Hz · ML · IR'),
        centerTitle: true,
        backgroundColor: Colors.black,
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(10.0),
          child: Column(
            children: [
              _alarmBanner(),
              if (_ghostCombo) _comboBanner(),
              // ===== 2 cột: radar (trái) + camera IR + ML (phải) =====
              SizedBox(
                height: 300,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(child: _radarPanel()),
                    const SizedBox(width: 8),
                    Expanded(child: _cameraPanel()),
                  ],
                ),
              ),
              const SizedBox(height: 8),
              _detectionPanel(),
              const SizedBox(height: 8),
              _infoPanel(),
              const SizedBox(height: 8),
              _bandChart(),
              const SizedBox(height: 8),
              _sensitivityRow(),
              const SizedBox(height: 6),
              _navigationPanel(),
              const SizedBox(height: 6),
              _note(),
              const SizedBox(height: 6),
              _buttons(),
              const SizedBox(height: 6),
              _logButtons(),
              const SizedBox(height: 6),
              _datasetButtons(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _alarmBanner() {
    if (!_isAlarming) return const SizedBox.shrink();
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFFF3B3B),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          const Icon(Icons.warning_amber_rounded, color: Colors.white, size: 22),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              '🚨 $_lastAlarmSummary',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _comboBanner() {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFFFF0080), Color(0xFFFF3B3B)],
        ),
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFFFF0080).withOpacity(0.6),
            blurRadius: 20,
            spreadRadius: 1,
          ),
        ],
      ),
      child: const Row(
        children: [
          Icon(Icons.flash_on, color: Colors.white, size: 26),
          SizedBox(width: 6),
          Expanded(
            child: Text(
              '👻 GHOST COMBO · Audio + Visual cùng dò được',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 13,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _radarPanel() {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 120),
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: _isAlarming
              ? const Color(0xFFFF3B3B)
              : const Color(0xFF1FE033).withOpacity(0.5),
          width: _isAlarming ? 4.0 : 1.5,
        ),
        boxShadow: _isAlarming
            ? [
                BoxShadow(
                  color: const Color(0xFFFF3B3B).withOpacity(0.45),
                  blurRadius: 20,
                  spreadRadius: 2,
                ),
              ]
            : null,
      ),
      child: Stack(
        children: [
          Positioned.fill(
            child: CustomPaint(
              painter: _RadarPainter(
                sweepAngleDeg: _angle,
                headingDeg: _heading,
                level: _level,
                alarm: _isAlarming,
                blips: List<_Blip>.unmodifiable(_blips),
                target: _target,
                hasCompass: _hasCompass,
              ),
            ),
          ),
          Positioned(
            top: 4,
            left: 4,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              color: const Color(0xFF1FE033).withOpacity(0.75),
              child: const Text('RADAR',
                  style: TextStyle(
                      fontSize: 9,
                      color: Colors.black,
                      fontWeight: FontWeight.bold)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _cameraPanel() {
    final Color borderColor = _ghostCombo
        ? const Color(0xFFFF0080)
        : (_isAlarming
            ? const Color(0xFFFF3B3B)
            : (_detections.isNotEmpty
                ? const Color(0xFF1FE033)
                : const Color(0xFF1FE033).withOpacity(0.5)));
    return AnimatedContainer(
      duration: const Duration(milliseconds: 120),
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: borderColor,
          width: (_isAlarming || _ghostCombo) ? 4.0 : 1.5,
        ),
        boxShadow: _ghostCombo
            ? [
                BoxShadow(
                  color: const Color(0xFFFF0080).withOpacity(0.55),
                  blurRadius: 22,
                  spreadRadius: 2,
                ),
              ]
            : _isAlarming
                ? [
                    BoxShadow(
                      color: const Color(0xFFFF3B3B).withOpacity(0.45),
                      blurRadius: 20,
                      spreadRadius: 2,
                    ),
                  ]
                : null,
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(11),
        child: Stack(
          fit: StackFit.expand,
          children: [
            _cameraContent(),
            // Bounding boxes cho ML Kit detections
            _detectionOverlay(),
            // Scanlines retro
            IgnorePointer(
              child: Column(
                children: List<Widget>.generate(60, (i) {
                  return Expanded(
                    child: Container(
                      color: i.isEven
                          ? Colors.transparent
                          : Colors.black.withOpacity(0.30),
                    ),
                  );
                }),
              ),
            ),
            // Label
            Positioned(
              top: 4,
              left: 4,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                color: _ghostCombo
                    ? const Color(0xFFFF0080).withOpacity(0.95)
                    : (_isAlarming
                        ? const Color(0xFFFF3B3B).withOpacity(0.85)
                        : (_detections.isNotEmpty
                            ? const Color(0xFF1FE033).withOpacity(0.85)
                            : const Color(0xFF1FE033).withOpacity(0.75))),
                child: Text(
                  _ghostCombo
                      ? '👻 COMBO'
                      : (_isAlarming
                          ? 'IR · ALARM'
                          : (_detections.isNotEmpty
                              ? 'YOLO · ${_objectCounts.values.fold(0, (a, b) => a + b)} obj'
                              : 'IR · ÂM BẢN')),
                  style: const TextStyle(
                      fontSize: 9,
                      color: Colors.black,
                      fontWeight: FontWeight.bold),
                ),
              ),
            ),
            // Crosshair
            Center(
              child: IgnorePointer(
                child: SizedBox(
                  width: 50,
                  height: 50,
                  child: CustomPaint(painter: _CrosshairPainter(alarm: _isAlarming || _ghostCombo)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _cameraContent() {
    if (_cameraError.isNotEmpty) {
      return Container(
        color: const Color(0xFF0A0A0A),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text(
              _cameraError,
              textAlign: TextAlign.center,
              style: const TextStyle(
                  color: Color(0xFFFF8080), fontSize: 11),
            ),
          ),
        ),
      );
    }
    if (_cameraTextureId == null) {
      return Container(
        color: const Color(0xFF0A0A0A),
        child: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor:
                      AlwaysStoppedAnimation<Color>(Color(0xFF1FE033)),
                ),
              ),
              SizedBox(height: 6),
              Text('Đang mở CameraX…',
                  style:
                      TextStyle(fontSize: 10, color: Color(0xFF808080))),
            ],
          ),
        ),
      );
    }
    // Render Flutter texture bound to CameraX SurfaceTexture
    return FittedBox(
      fit: BoxFit.cover,
      child: SizedBox(
        width: 480, // CameraX target resolution (see CameraService.kt)
        height: 640,
        child: ColorFiltered(
          colorFilter: const ColorFilter.matrix(<double>[
            -1, 0, 0, 0, 255, //
            0, -1, 0, 0, 255, //
            0, 0, -1, 0, 255, //
            0, 0, 0, 1, 0, //
          ]),
          child: Texture(textureId: _cameraTextureId!),
        ),
      ),
    );
  }

  Widget _detectionOverlay() {
    if (_detections.isEmpty) return const SizedBox.shrink();
    return LayoutBuilder(
      builder: (context, constraints) {
        // CameraX v2.0: image size is fixed 480x640 (sensor landscape).
        // FittedBox.cover fills display while preserving aspect.
        final Size box = Size(constraints.maxWidth, constraints.maxHeight);
        // Image is rendered via Texture with SizedBox(480x640) then FittedBox.cover
        // BoxFit.cover: scale = max(box.w/img.w, box.h/img.h)
        const double imgW = 640;
        const double imgH = 480;
        final double scale = math.max(box.width / imgW, box.height / imgH);
        final double scaledW = imgW * scale;
        final double scaledH = imgH * scale;
        final double dx = (box.width - scaledW) / 2.0;
        final double dy = (box.height - scaledH) / 2.0;
        return CustomPaint(
          painter: _DetectionPainter(
            detections: _detections,
            imgSize: const Size(imgW, imgH),
            scale: scale,
            dx: dx,
            dy: dy,
          ),
        );
      },
    );
  }

  Widget _detectionPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF0E0E0E),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: _detections.isNotEmpty
              ? const Color(0xFF1FE033).withOpacity(0.5)
              : const Color(0xFF252525),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.center_focus_strong,
                  color: Color(0xFF1FE033), size: 18),
              const SizedBox(width: 6),
              const Text(
                'ML KIT OBJECT DETECTION',
                style: TextStyle(
                    fontSize: 12,
                    color: Color(0xFFB0B0B0),
                    fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              if (_objectCounts.isNotEmpty)
                Text(
                  '${_objectCounts.values.fold(0, (a, b) => a + b)} đối tượng',
                  style: const TextStyle(
                      fontSize: 11, color: Color(0xFF1FE033)),
                ),
            ],
          ),
          const SizedBox(height: 6),
          if (_objectCounts.isEmpty)
            const Text(
              'Chưa phát hiện đối tượng nào. Đưa camera về phía có người/đồ vật.',
              style: TextStyle(fontSize: 12, color: Color(0xFF808080)),
            )
          else
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: _objectCounts.entries
                  .map((e) => Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1FE033).withOpacity(0.15),
                          border: Border.all(
                              color: const Color(0xFF1FE033)
                                  .withOpacity(0.5)),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          '${e.key} ×${e.value}',
                          style: const TextStyle(
                              fontSize: 11, color: Color(0xFF1FE033)),
                        ),
                      ))
                  .toList(),
            ),
          if (_topLabel != '--') ...[
            const SizedBox(height: 6),
            Row(
              children: [
                const Icon(Icons.bolt, color: Color(0xFFFFB300), size: 14),
                const SizedBox(width: 4),
                Text(
                  'Mạnh nhất: $_topLabel (${(_topConfidence * 100).toStringAsFixed(0)}%)',
                  style: const TextStyle(
                      fontSize: 12, color: Color(0xFFFFB300)),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _infoPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF101010),
        border: Border.all(
          color: _ghostCombo
              ? const Color(0xFFFF0080)
              : _isAlarming
                  ? const Color(0xFFFF3B3B)
                  : const Color(0xFF1FE033).withOpacity(0.5),
        ),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _statusText,
            style: TextStyle(
              fontSize: 15,
              color: _ghostCombo
                  ? const Color(0xFFFF0080)
                  : _isAlarming
                      ? const Color(0xFFFF3B3B)
                      : const Color(0xFF1FE033),
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Dải mạnh nhất: $_dominantBandName · ≈ ${_dominantBandHz.toStringAsFixed(2)} Hz',
            style: const TextStyle(fontSize: 12),
          ),
          Text(
            '🧭 Bạn đang quay mặt về: ${_angleToCompass(_heading)}'
            '${!_hasCompass ? " (không có la bàn)" : ""}',
            style: const TextStyle(fontSize: 12),
          ),
          if (_isScanning)
            Text(
              'Báo động: $_alarmCount lần · Ghost combo: $_comboCount · '
              'EVP clips: $_evpCount · Dataset: $_datasetCount samples',
              style: const TextStyle(fontSize: 11, color: Color(0xFFB0B0B0)),
            ),
        ],
      ),
    );
  }

  Widget _bandChart() {
    // v2.1.0: combine waterfall spectrogram + 5-band bars + EMF meter
    double maxE = 0;
    for (final e in _bandEnergy) {
      if (e > maxE) maxE = e;
    }
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xFF0E0E0E),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF252525)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title row + EMF meter
          Row(
            children: [
              const Expanded(
                child: Text('WATERFALL + 5 DẢI TẦN (0–20 Hz)',
                    style: TextStyle(
                        fontSize: 12,
                        color: Color(0xFFB0B0B0),
                        fontWeight: FontWeight.bold)),
              ),
              EmfMeter(
                key: _emfKey,
                height: 80,
                width: 55,
              ),
            ],
          ),
          const SizedBox(height: 6),
          // Waterfall spectrogram (v2.1.0)
          WaterfallSpectrogram(
            key: _waterfallKey,
            numBins: _fftSize ~/ 2, // 128 bins (Nyquist)
            maxFrames: 30, // ~77s history
            height: 120,
          ),
          const SizedBox(height: 8),
          // 5 band bars
          for (int i = 0; i < 5; i++) _bandBar(i, maxE),
        ],
      ),
    );
  }

  Widget _bandBar(int i, double maxE) {
    final double e = _bandEnergy[i];
    final double b = _bandBaseline[i];
    final double ratio = b > 0 ? e / b : 0;
    final double widthFrac = maxE > 0 ? (e / maxE).clamp(0.0, 1.0) : 0.0;
    final Color barColor = ratio >= _sensitivity
        ? const Color(0xFFFF3B3B)
        : ratio >= _sensitivity * 0.6
            ? const Color(0xFFFFB300)
            : const Color(0xFF1FE033);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(
            width: 70,
            child: Text(
              _bandNames[i],
              style: const TextStyle(fontSize: 12, color: Color(0xFFB0B0B0)),
            ),
          ),
          Expanded(
            child: Stack(
              children: [
                Container(
                  height: 16,
                  decoration: BoxDecoration(
                    color: const Color(0xFF1A1A1A),
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),
                FractionallySizedBox(
                  widthFactor: widthFrac,
                  child: Container(
                    height: 16,
                    decoration: BoxDecoration(
                      color: barColor,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 6),
          SizedBox(
            width: 56,
            child: Text(
              '${ratio.toStringAsFixed(1)}×',
              textAlign: TextAlign.right,
              style: const TextStyle(fontSize: 11, color: Color(0xFF808080)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sensitivityRow() {
    return Row(
      children: [
        const Text('Độ nhạy', style: TextStyle(fontSize: 13)),
        const SizedBox(width: 8),
        Expanded(
          child: SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: const Color(0xFF1FE033),
              inactiveTrackColor: const Color(0xFF303030),
              thumbColor: const Color(0xFF1FE033),
            ),
            child: Slider(
              min: 1.2,
              max: 6.0,
              divisions: 24,
              value: _sensitivity,
              label: '${_sensitivity.toStringAsFixed(1)}×',
              onChanged: _isScanning
                  ? (v) => setState(() => _sensitivity = v)
                  : null,
            ),
          ),
        ),
        SizedBox(
          width: 42,
          child: Text(
            '${_sensitivity.toStringAsFixed(1)}×',
            textAlign: TextAlign.right,
            style: const TextStyle(fontSize: 13, color: Color(0xFFB0B0B0)),
          ),
        ),
      ],
    );
  }

  Widget _navigationPanel() {
    if (_target == null) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: const Color(0xFF0B0B0B),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(color: const Color(0xFF252525)),
        ),
        child: const Row(
          children: [
            Icon(Icons.explore, color: Color(0xFF606060), size: 18),
            SizedBox(width: 6),
            Expanded(
              child: Text(
                '🎯 Mục tiêu: chưa có. Xoay người chậm 360° và chờ blip đỏ đầu tiên.',
                style: TextStyle(fontSize: 12, color: Color(0xFF808080)),
              ),
            ),
          ],
        ),
      );
    }
    final t = _target!;
    double rel = t.angleDeg - _heading;
    while (rel > 180) {
      rel -= 360;
    }
    while (rel < -180) {
      rel += 360;
    }
    final String turn = rel.abs() < 8
        ? 'Đang đối diện'
        : rel > 0
            ? 'Xoay phải ${rel.toStringAsFixed(0)}°'
            : 'Xoay trái ${rel.abs().toStringAsFixed(0)}°';
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A0808),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xFFFF3B3B).withOpacity(0.6)),
      ),
      child: Row(
        children: [
          const Icon(Icons.gps_fixed, color: Color(0xFFFF3B3B), size: 22),
          const SizedBox(width: 6),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '🎯 Mục tiêu: ${_angleToCompass(t.angleDeg)}  '
                  '(${_bandNames[t.bandIndex]})',
                  style: const TextStyle(
                      fontSize: 13,
                      color: Color(0xFFFF8080),
                      fontWeight: FontWeight.bold),
                ),
                Text(
                  '$turn · ${(t.energy * _sensitivity).toStringAsFixed(1)}× nền'
                  ' · còn ${(_blipLifetimeSec - t.ageSeconds).toStringAsFixed(0)}s',
                  style: const TextStyle(fontSize: 12, color: Color(0xFFB0B0B0)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _note() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xFF0B0B0B),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        _noteText,
        textAlign: TextAlign.center,
        style: const TextStyle(fontSize: 11, color: Color(0xFFB0B0B0)),
      ),
    );
  }

  Widget _buttons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isScanning ? null : startScan,
            icon: const Icon(Icons.play_arrow),
            label: const Text('BẮT ĐẦU'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF1FE033),
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(vertical: 11),
              textStyle: const TextStyle(
                  fontSize: 13, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isAlarming ? toggleMute : null,
            icon: Icon(
                _alarmMuted ? Icons.volume_off : Icons.notifications_active),
            label: Text(_alarmMuted ? 'BẬT LOA' : 'TẮT LOA'),
            style: ElevatedButton.styleFrom(
              backgroundColor: _alarmMuted
                  ? const Color(0xFF404040)
                  : const Color(0xFFFFB300),
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(vertical: 11),
              textStyle: const TextStyle(
                  fontSize: 13, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isScanning ? stopScan : null,
            icon: const Icon(Icons.stop),
            label: const Text('DỪNG'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFFF3B3B),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 11),
              textStyle: const TextStyle(
                  fontSize: 13, fontWeight: FontWeight.bold),
            ),
          ),
        ),
      ],
    );
  }

  // ====== Log management (v2.0.2) ======

  Future<void> _showLogDialog() async {
    final lines = await DetectionLogger.instance.tailCurrentLog(50);
    if (!mounted) return;
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF101010),
        title: Row(
          children: [
            const Icon(Icons.description, color: Color(0xFF00E5FF), size: 20),
            const SizedBox(width: 8),
            const Text('Detection log (50 dòng cuối)',
                style: TextStyle(color: Colors.white, fontSize: 16)),
          ],
        ),
        content: SizedBox(
          width: double.maxFinite,
          height: 400,
          child: lines.isEmpty
              ? const Center(
                  child: Text('Chưa có log nào',
                      style: TextStyle(color: Colors.white54)),
                )
              : ListView.builder(
                  itemCount: lines.length,
                  itemBuilder: (_, i) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 1),
                    child: SelectableText(
                      lines[i],
                      style: const TextStyle(
                        color: Color(0xFFB0BEC5),
                        fontSize: 10,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
                ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('ĐÓNG', style: TextStyle(color: Color(0xFF00E5FF))),
          ),
        ],
      ),
    );
  }

  Future<void> _shareLog() async {
    final path = await DetectionLogger.instance.getCurrentLogPath();
    if (path == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Chưa có log để chia sẻ'), duration: Duration(seconds: 2)),
      );
      return;
    }
    try {
      await Share.shareXFiles(
        [XFile(path)],
        text: 'Ghost Radar log (${_loggedDetectionCount} detections)',
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text('Chia sẻ lỗi: $e'), duration: const Duration(seconds: 2)),
      );
    }
  }

  Future<void> _clearLog() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF101010),
        title: const Text('Xóa toàn bộ log?',
            style: TextStyle(color: Colors.white, fontSize: 16)),
        content: const Text(
          'Sẽ xóa tất cả file detection_log trong app. Không thể khôi phục.',
          style: TextStyle(color: Colors.white70, fontSize: 13),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('HỦY', style: TextStyle(color: Colors.white54)),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('XÓA', style: TextStyle(color: Color(0xFFFF3B3B))),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    await DetectionLogger.instance.clearAll();
    _lastDetectionLog.clear();
    _loggedDetectionCount = 0;
    if (!mounted) return;
    setState(() {});
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Đã xóa log'), duration: Duration(seconds: 2)),
    );
  }

  Widget _logButtons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _showLogDialog,
            icon: const Icon(Icons.list_alt, size: 18),
            label: Text('XEM LOG ($_loggedDetectionCount)'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF263238),
              foregroundColor: const Color(0xFF00E5FF),
              padding: const EdgeInsets.symmetric(vertical: 9),
              textStyle:
                  const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _shareLog,
            icon: const Icon(Icons.share, size: 18),
            label: const Text('CHIA SẺ'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF263238),
              foregroundColor: const Color(0xFFFFB300),
              padding: const EdgeInsets.symmetric(vertical: 9),
              textStyle:
                  const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _clearLog,
            icon: const Icon(Icons.delete_outline, size: 18),
            label: const Text('XÓA LOG'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF263238),
              foregroundColor: const Color(0xFFFF5252),
              padding: const EdgeInsets.symmetric(vertical: 9),
              textStyle:
                  const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ),
        ),
      ],
    );
  }

  // ====== Dataset export (v2.1.0) ======

  Future<void> _exportDataset() async {
    final count = await DatasetExporter.getSampleCount();
    if (count == 0) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Dataset trống. Bấm "LƯU FRAME" khi có detection.'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }
    try {
      final size = await DatasetExporter.exportAndShare();
      if (size == null) return;
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Đã xuất $count samples (${(size / 1024).toStringAsFixed(0)} KB)'),
          duration: const Duration(seconds: 2),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Xuất lỗi: $e')),
      );
    }
  }

  Future<void> _saveCurrentFrame() async {
    // v2.1.0: save current detection snapshot as YOLO sample.
    // v2.1.0 limitation: native frame-grab MethodChannel deferred to v2.1.1;
    // for now we render a label image with detection summary as the JPEG.
    if (!_isScanning) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Cần BẮT ĐẦU quét trước')),
      );
      return;
    }
    if (_detections.isEmpty) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Chưa có detection nào để lưu')),
      );
      return;
    }
    try {
      final jpegBytes = _buildLabelImage();
      final bboxes = _detections
          .map((d) => ImageBBox(
                className: d.label,
                cx: (d.boundingBox.left + d.boundingBox.right) / 2,
                cy: (d.boundingBox.top + d.boundingBox.bottom) / 2,
                w: d.boundingBox.width,
                h: d.boundingBox.height,
              ))
          .toList();
      final base = await DatasetExporter.saveSample(
        jpegBytes: jpegBytes,
        imageWidth: 640,
        imageHeight: 480,
        bboxes: bboxes,
      );
      await DatasetExporter.writeClassesTxt();
      if (!mounted) return;
      setState(() {
        _datasetCount++;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Đã lưu $base (${bboxes.length} boxes)'),
          duration: const Duration(seconds: 2),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Lưu lỗi: $e')),
      );
    }
  }

  /// Build a 640x480 label image with detection summary + bbox overlays.
  /// v2.1.0: synthesized placeholder until native frame-grab lands in v2.1.1.
  Uint8List _buildLabelImage() {
    final image = img.Image(width: 640, height: 480);
    // Dark background
    img.fill(image, color: img.ColorRgb8(8, 8, 12));
    // Border
    img.drawRect(image,
        x1: 0, y1: 0, x2: 639, y2: 479,
        color: img.ColorRgb8(40, 40, 40),
        thickness: 2);
    // Title
    img.drawString(image, 'Ghost Radar v2.1.0 - YOLO sample',
        font: img.arial24,
        x: 12, y: 8,
        color: img.ColorRgb8(31, 224, 51));
    img.drawString(image, 'Detections: ${_detections.length}',
        font: img.arial14,
        x: 12, y: 38,
        color: img.ColorRgb8(255, 255, 255));
    final summary = _detections.take(3).map((d) =>
        '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%').join(' | ');
    img.drawString(image, summary,
        font: img.arial14,
        x: 12, y: 58,
        color: img.ColorRgb8(0, 229, 255));
    img.drawString(image,
        'Heading: ${_heading.toStringAsFixed(0)} deg  |  Combo: $_comboCount  |  EVP: $_evpCount',
        font: img.arial14,
        x: 12, y: 78,
        color: img.ColorRgb8(176, 176, 176));
    // Draw bboxes scaled to 640x480
    for (final d in _detections) {
      final double x1 = d.boundingBox.left.clamp(0, 639).toDouble();
      final double y1 = d.boundingBox.top.clamp(0, 479).toDouble();
      final double x2 = d.boundingBox.right.clamp(0, 639).toDouble();
      final double y2 = d.boundingBox.bottom.clamp(0, 479).toDouble();
      final color = _cocoColors[d.label] ?? const Color(0xFF1FE033);
      img.drawRect(image,
          x1: x1.toInt(),
          y1: y1.toInt(),
          x2: x2.toInt(),
          y2: y2.toInt(),
          color: img.ColorRgb8(color.red, color.green, color.blue),
          thickness: 2);
      img.drawString(image,
          '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%',
          font: img.arial14,
          x: x1.toInt().clamp(0, 500),
          y: (y1.toInt() - 18).clamp(0, 460),
          color: img.ColorRgb8(color.red, color.green, color.blue));
    }
    // Footer
    img.drawString(image,
        'YOLO format: <class_id> <cx> <cy> <w> <h> (normalized 0..1)',
        font: img.arial14,
        x: 12, y: 450,
        color: img.ColorRgb8(120, 120, 120));
    return Uint8List.fromList(img.encodeJpg(image, quality: 88));
  }

  Widget _datasetButtons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _saveCurrentFrame,
            icon: const Icon(Icons.save_alt, size: 18),
            label: Text('LƯU FRAME (${_datasetCount})'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF263238),
              foregroundColor: const Color(0xFF76FF03),
              padding: const EdgeInsets.symmetric(vertical: 9),
              textStyle:
                  const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _exportDataset,
            icon: const Icon(Icons.archive, size: 18),
            label: const Text('XUẤT DATASET'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF263238),
              foregroundColor: const Color(0xFF40C4FF),
              padding: const EdgeInsets.symmetric(vertical: 9),
              textStyle:
                  const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () async {
              await DatasetExporter.clearAll();
              if (!mounted) return;
              setState(() {
                _datasetCount = 0;
              });
              if (!mounted) return;
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Đã xóa dataset')),
              );
            },
            icon: const Icon(Icons.delete_sweep, size: 18),
            label: const Text('XÓA DS'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF263238),
              foregroundColor: const Color(0xFFFF5252),
              padding: const EdgeInsets.symmetric(vertical: 9),
              textStyle:
                  const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ),
        ),
      ],
    );
  }
}

// ============================================================
// Radar painter
// ============================================================
class _RadarPainter extends CustomPainter {
  _RadarPainter({
    required this.sweepAngleDeg,
    required this.headingDeg,
    required this.level,
    required this.alarm,
    required this.blips,
    required this.target,
    required this.hasCompass,
  });
  final double sweepAngleDeg;
  final double headingDeg;
  final double level;
  final bool alarm;
  final List<_Blip> blips;
  final _Blip? target;
  final bool hasCompass;

  double _compassToMath(double compassDeg) =>
      (compassDeg - 90) * math.pi / 180.0;

  @override
  void paint(Canvas canvas, Size size) {
    final Offset c = Offset(size.width / 2, size.height / 2);
    final double rMax = math.min(size.width, size.height) * 0.40;

    final ringColor =
        alarm ? const Color(0xFFFF3B3B) : const Color(0xFF1FE033);
    final ringPaint = Paint()
      ..color = ringColor.withOpacity(0.4)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;
    final crossPaint = Paint()
      ..color = ringColor.withOpacity(0.35)
      ..strokeWidth = 0.7;
    final outerPaint = Paint()
      ..color = ringColor.withOpacity(0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = alarm ? 2.5 : 1.5;

    for (final frac in [0.33, 0.66, 1.0]) {
      canvas.drawCircle(c, rMax * frac, ringPaint);
    }
    canvas.drawLine(
        Offset(c.dx - rMax, c.dy), Offset(c.dx + rMax, c.dy), crossPaint);
    canvas.drawLine(
        Offset(c.dx, c.dy - rMax), Offset(c.dx, c.dy + rMax), crossPaint);
    canvas.drawCircle(c, rMax, outerPaint);

    _drawCompassLabels(canvas, c, rMax);

    // Sweep arc 30°
    final sweepPaint = Paint()
      ..shader = SweepGradient(
        startAngle: 0,
        endAngle: math.pi / 6,
        colors: <Color>[
          ringColor.withOpacity(0.0),
          ringColor.withOpacity(0.0),
          ringColor.withOpacity(0.35),
        ],
        stops: const <double>[0.0, 0.7, 1.0],
        transform:
            GradientRotation(_compassToMath(sweepAngleDeg) + math.pi / 2),
      ).createShader(Rect.fromCircle(center: c, radius: rMax));
    final sweepPath = Path()
      ..moveTo(c.dx, c.dy)
      ..lineTo(c.dx + rMax, c.dy)
      ..arcTo(
        Rect.fromCircle(center: c, radius: rMax),
        0,
        math.pi / 6,
        false,
      )
      ..close();
    canvas.drawPath(sweepPath, sweepPaint);

    for (final b in blips) {
      _drawBlip(canvas, c, rMax, b, alarm);
    }

    if (target != null) {
      _drawTarget(canvas, c, rMax, target!, alarm);
    }

    _drawHeadingNeedle(canvas, c, rMax, alarm);

    final centerPaint = Paint()
      ..color = (alarm ? const Color(0xFFFF3B3B) : const Color(0xFF1FE033))
          .withOpacity(level > 0.05 || alarm ? 0.95 : 0.6);
    final dotR = 5.0 + level.clamp(0.0, 1.0) * 10.0;
    canvas.drawCircle(c, dotR, centerPaint);
  }

  void _drawCompassLabels(Canvas canvas, Offset c, double r) {
    final labelR = r + 12;
    const style = TextStyle(
      color: Color(0xFF808080),
      fontSize: 11,
      fontWeight: FontWeight.bold,
    );
    const dirs = ['B', 'Đ', 'N', 'T'];
    final angles = <double>[0, 90, 180, 270];
    for (int i = 0; i < 4; i++) {
      final a = _compassToMath(angles[i]);
      final dx = c.dx + labelR * math.cos(a);
      final dy = c.dy + labelR * math.sin(a);
      final tp = TextPainter(
        text: TextSpan(text: dirs[i], style: style),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(dx - tp.width / 2, dy - tp.height / 2));
    }
  }

  void _drawBlip(Canvas canvas, Offset c, double rMax, _Blip b, bool alarm) {
    final double a = _compassToMath(b.angleDeg);
    final double rr = rMax * 0.7;
    final double x = c.dx + rr * math.cos(a);
    final double y = c.dy + rr * math.sin(a);
    final double size = 6.0 + b.energy * 14.0;
    final double alpha = b.fade * (alarm ? 1.0 : 0.95);

    final Color col = b.score > 0.7
        ? const Color(0xFFFF3B3B)
        : b.score > 0.4
            ? const Color(0xFFFFB300)
            : const Color(0xFF1FE033);

    final glow = Paint()
      ..color = col.withOpacity(alpha * 0.35)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
    canvas.drawCircle(Offset(x, y), size * 1.6, glow);

    final core = Paint()..color = col.withOpacity(alpha);
    canvas.drawCircle(Offset(x, y), size, core);

    final inner = Paint()..color = Colors.white.withOpacity(alpha * 0.9);
    canvas.drawCircle(Offset(x, y), size * 0.35, inner);
  }

  void _drawTarget(
      Canvas canvas, Offset c, double rMax, _Blip t, bool alarm) {
    final double a = _compassToMath(t.angleDeg);
    final double rr = rMax * 0.7;
    final double x = c.dx + rr * math.cos(a);
    final double y = c.dy + rr * math.sin(a);

    final ring = Paint()
      ..color = const Color(0xFFFF3B3B).withOpacity(0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawCircle(Offset(x, y), 22, ring);

    final cross = Paint()
      ..color = const Color(0xFFFF3B3B).withOpacity(0.85)
      ..strokeWidth = 1.5;
    canvas.drawLine(Offset(x - 26, y), Offset(x - 8, y), cross);
    canvas.drawLine(Offset(x + 8, y), Offset(x + 26, y), cross);
    canvas.drawLine(Offset(x, y - 26), Offset(x, y - 8), cross);
    canvas.drawLine(Offset(x, y + 8), Offset(x, y + 26), cross);

    final arrowPaint = Paint()
      ..color = const Color(0xFFFF8080).withOpacity(0.9)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;
    final double startX = c.dx + (rMax * 0.05) * math.cos(a);
    final double startY = c.dy + (rMax * 0.05) * math.sin(a);
    final double endX = x - 30 * math.cos(a);
    final double endY = y - 30 * math.sin(a);
    final path = Path()
      ..moveTo(startX, startY)
      ..lineTo(endX, endY);
    canvas.drawPath(path, arrowPaint);
    final headLen = 10.0;
    final headAngle = 0.5;
    final ax1 = endX - headLen * math.cos(a - headAngle);
    final ay1 = endY - headLen * math.sin(a - headAngle);
    final ax2 = endX - headLen * math.cos(a + headAngle);
    final ay2 = endY - headLen * math.sin(a + headAngle);
    final head = Path()
      ..moveTo(endX, endY)
      ..lineTo(ax1, ay1)
      ..lineTo(ax2, ay2)
      ..close();
    canvas.drawPath(
        head, Paint()..color = const Color(0xFFFF8080).withOpacity(0.9));
  }

  void _drawHeadingNeedle(Canvas canvas, Offset c, double rMax, bool alarm) {
    final double a = _compassToMath(headingDeg);
    final double rr = rMax * 0.92;
    final double x = c.dx + rr * math.cos(a);
    final double y = c.dy + rr * math.sin(a);
    final paint = Paint()
      ..color = (alarm
              ? const Color(0xFFFF8080)
              : const Color(0xFF80FF80))
          .withOpacity(0.7)
      ..strokeWidth = 2.0;
    canvas.drawLine(c, Offset(x, y), paint);
    final headLen = 12.0;
    final headAngle = 0.45;
    final ax1 = x - headLen * math.cos(a - headAngle);
    final ay1 = y - headLen * math.sin(a - headAngle);
    final ax2 = x - headLen * math.cos(a + headAngle);
    final ay2 = y - headLen * math.sin(a + headAngle);
    final head = Path()
      ..moveTo(x, y)
      ..lineTo(ax1, ay1)
      ..lineTo(ax2, ay2)
      ..close();
    canvas.drawPath(head, Paint()..color = paint.color);
  }

  @override
  bool shouldRepaint(covariant _RadarPainter old) =>
      old.sweepAngleDeg != sweepAngleDeg ||
      old.headingDeg != headingDeg ||
      old.level != level ||
      old.alarm != alarm ||
      old.blips.length != blips.length ||
      old.target != target ||
      old.hasCompass != hasCompass;
}

// ============================================================
// Detection painter: bounding boxes cho ML Kit detections
// ============================================================
class _DetectionPainter extends CustomPainter {
  _DetectionPainter({
    required this.detections,
    required this.imgSize,
    required this.scale,
    required this.dx,
    required this.dy,
  });
  final List<_DetectedObj> detections;
  final Size imgSize;
  final double scale;
  final double dx;
  final double dy;

  Rect _mapRect(Rect r) {
    return Rect.fromLTWH(
      dx + r.left * scale,
      dy + r.top * scale,
      r.width * scale,
      r.height * scale,
    );
  }

  @override
  void paint(Canvas canvas, Size size) {
    for (final d in detections) {
      final Rect mapped = _mapRect(d.boundingBox);
      final Color c = _colorForCoco(d.label);
      final double fade = d.fade;
      // ----- 1. Trail history (fading polylines) -----
      final hist = d.trackHistory;
      if (hist != null && hist.length >= 2) {
        for (int i = 1; i < hist.length; i++) {
          final p0 = _mapRect(hist[i - 1]).center;
          final p1 = _mapRect(hist[i]).center;
          final double tFade = (i / hist.length) * fade;
          if (tFade <= 0) continue;
          final paint = Paint()
            ..color = c.withOpacity(0.6 * tFade)
            ..strokeWidth = 1.4
            ..strokeCap = StrokeCap.round;
          canvas.drawLine(p0, p1, paint);
          // Dot ở đầu
          canvas.drawCircle(p1, 1.6, Paint()..color = c.withOpacity(tFade));
        }
      }
      // ----- 2. Bounding box (Tesla-style: 4 corner brackets) -----
      final boxPaint = Paint()
        ..color = c.withOpacity(fade)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..strokeCap = StrokeCap.square;
      // Vẽ 4 góc (bracket style như xe Tesla)
      final double cornerLen = math.min(14.0, math.min(mapped.width, mapped.height) / 3);
      // Top-left
      canvas.drawLine(mapped.topLeft, mapped.topLeft.translate(cornerLen, 0), boxPaint);
      canvas.drawLine(mapped.topLeft, mapped.topLeft.translate(0, cornerLen), boxPaint);
      // Top-right
      canvas.drawLine(mapped.topRight, mapped.topRight.translate(-cornerLen, 0), boxPaint);
      canvas.drawLine(mapped.topRight, mapped.topRight.translate(0, cornerLen), boxPaint);
      // Bottom-left
      canvas.drawLine(mapped.bottomLeft, mapped.bottomLeft.translate(cornerLen, 0), boxPaint);
      canvas.drawLine(mapped.bottomLeft, mapped.bottomLeft.translate(0, -cornerLen), boxPaint);
      // Bottom-right
      canvas.drawLine(mapped.bottomRight, mapped.bottomRight.translate(-cornerLen, 0), boxPaint);
      canvas.drawLine(mapped.bottomRight, mapped.bottomRight.translate(0, -cornerLen), boxPaint);
      // ----- 3. Label background + text -----
      final reidBadge = d.reidentified ? ' (re-id)' : '';
      final labelText = d.trackId != null
          ? '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}% #${d.trackId}$reidBadge'
          : '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%$reidBadge';
      final tp = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.black,
            fontSize: 9,
            fontWeight: FontWeight.bold,
            fontFamily: 'monospace',
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final labelBg = Rect.fromLTWH(
        mapped.left,
        (mapped.top - tp.height - 4).clamp(0.0, size.height - tp.height),
        tp.width + 6,
        tp.height + 2,
      );
      canvas.drawRect(labelBg, Paint()..color = c.withOpacity(fade));
      tp.paint(canvas, Offset(labelBg.left + 3, labelBg.top + 1));
      // ----- 4. Status badge (OK = filled dot, TENT = outlined) -----
      final badgeCenter = Offset(mapped.right - 4, mapped.top + 4);
      final badgePaint = Paint()
        ..color = c.withOpacity(fade)
        ..style = PaintingStyle.fill;
      canvas.drawCircle(badgeCenter, 2.5, badgePaint);
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter old) =>
      old.detections.length != detections.length ||
      old.scale != scale ||
      old.dx != dx ||
      old.dy != dy;
}

class _CrosshairPainter extends CustomPainter {
  _CrosshairPainter({required this.alarm});
  final bool alarm;
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = (alarm ? const Color(0xFFFF3B3B) : const Color(0xFF80FF80))
          .withOpacity(0.5)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;
    final cx = size.width / 2;
    final cy = size.height / 2;
    canvas.drawCircle(Offset(cx, cy), size.width * 0.35, paint);
    canvas.drawLine(Offset(0, cy), Offset(size.width, cy), paint);
    canvas.drawLine(Offset(cx, 0), Offset(cx, size.height), paint);
  }

  @override
  bool shouldRepaint(covariant _CrosshairPainter old) => old.alarm != alarm;
}
