// Ghost Radar v2.0+ - Native YUV preprocessing via JNI.
//
// Dart-side binding to the native C++ JNI function in yuv_processor.cpp.
// Inputs: Y/U/V ByteBuffers from CameraX (YUV_420_888) + dimensions.
// Output: Float32List NCHW [1, 3, inputSize, inputSize] (zero-alloc reused buffer).
//
// Uses direct ByteBuffer when available; falls back to copy otherwise.

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class YuvPreprocessor {
  YuvPreprocessor({this.inputSize = 640});

  static const MethodChannel _method = MethodChannel('ghost_radar/camera');

  final int inputSize;

  // Pre-allocated output buffer (reused across frames).
  late final Float32List _outBuffer;
  late final int _outSize;

  // Letterbox params cache (recomputed when image size changes)
  ({int padX, int padY, double scale})? _letterbox;
  int? _lastWidth;
  int? _lastHeight;

  YuvPreprocessor._init(this.inputSize) {
    _outSize = 1 * 3 * inputSize * inputSize;
    _outBuffer = Float32List(_outSize);
  }

  factory YuvPreprocessor.create({int inputSize = 640}) {
    final p = YuvPreprocessor._init(inputSize);
    return p;
  }

  Float32List get outputBuffer => _outBuffer;
  int get outputSize => _outSize;

  /// Run preprocessing via native JNI.
  /// Returns the same Float32List (reused buffer) - reshaped to [1, 3, S, S] for TFLite.
  ///
  /// NOTE: Currently uses Dart-side preprocessing as a fallback. The native C++
  /// JNI path requires the ByteBuffer to be direct (allocated off-heap). CameraX
  /// ImageProxy buffers are direct, but the platform channel currently wraps them
  /// in heap ByteBuffer for transport. To fully utilize native preprocessing, the
  /// EventChannel payload should use StandardMessageCodec with direct buffer support.
  ///
  /// For now, we expose the letterbox params via JNI and do the pixel-level YUV->RGB
  /// in Dart (which is already fast enough on modern ARM CPUs).
  Float32List preprocess({
    required dynamic yPlane,
    required dynamic uPlane,
    required dynamic vPlane,
    required int yRowStride,
    required int uvRowStride,
    required int uvPixelStride,
    required int width,
    required int height,
  }) {
    // Recompute letterbox if image size changed
    if (_lastWidth != width || _lastHeight != height) {
      _letterbox = _computeLetterbox(width, height);
      _lastWidth = width;
      _lastHeight = height;
    }
    final lb = _letterbox!;

    _yuvToRgbNchw(
      yPlane: yPlane,
      uPlane: uPlane,
      vPlane: vPlane,
      yRowStride: yRowStride,
      uvRowStride: uvRowStride,
      uvPixelStride: uvPixelStride,
      width: width,
      height: height,
      padX: lb.padX,
      padY: lb.padY,
      scale: lb.scale,
    );
    return _outBuffer;
  }

  /// Recompute letterbox via JNI (faster + cleaner than Dart math).
  Future<void> ensureLetterboxFromNative({
    required int width,
    required int height,
  }) async {
    if (_lastWidth == width && _lastHeight == height && _letterbox != null) return;
    final res = await _method.invokeMapMethod<String, dynamic>(
      'letterboxParams',
      {'width': width, 'height': height, 'inputSize': inputSize},
    );
    if (res != null) {
      _letterbox = (
        padX: res['padX'] as int,
        padY: res['padY'] as int,
        scale: (res['scale'] as num).toDouble(),
      );
      _lastWidth = width;
      _lastHeight = height;
    } else {
      _letterbox = _computeLetterbox(width, height);
    }
  }

  ({int padX, int padY, double scale}) get letterbox {
    if (_letterbox == null) {
      throw StateError('Letterbox not computed yet');
    }
    return _letterbox!;
  }

  ({int padX, int padY, double scale}) _computeLetterbox(int w, int h) {
    final scale = (inputSize / w < inputSize / h) ? inputSize / w : inputSize / h;
    final newW = (w * scale).round();
    final newH = (h * scale).round();
    return (
      padX: (inputSize - newW) ~/ 2,
      padY: (inputSize - newH) ~/ 2,
      scale: scale,
    );
  }

  /// YUV_420_888 -> RGB letterbox -> NCHW Float32 [0,1]
  /// Inputs are ByteBuffer from CameraX (Y, U, V planes).
  /// Same algorithm as yuv_processor.cpp but in Dart (fallback for now).
  void _yuvToRgbNchw({
    required dynamic yPlane,
    required dynamic uPlane,
    required dynamic vPlane,
    required int yRowStride,
    required int uvRowStride,
    required int uvPixelStride,
    required int width,
    required int height,
    required int padX,
    required int padY,
    required double scale,
  }) {
    final ByteBuffer yBuf = yPlane as ByteBuffer;
    final ByteBuffer uBuf = uPlane as ByteBuffer;
    final ByteBuffer vBuf = vPlane as ByteBuffer;

    final Uint8List yBytes = yBuf.asUint8List();
    final Uint8List uBytes = uBuf.asUint8List();
    final Uint8List vBytes = vBuf.asUint8List();

    final int S = inputSize;
    final int outSize = _outSize;
    // Fill padding area with 0.5 (gray)
    for (int i = 0; i < outSize; i++) {
      _outBuffer[i] = 0.5;
    }

    final int newW = (width * scale).round();
    final int newH = (height * scale).round();

    for (int oy = 0; oy < newH; oy++) {
      final int sy = (oy / scale).floor();
      if (sy < 0 || sy >= height) continue;
      for (int ox = 0; ox < newW; ox++) {
        final int sx = (ox / scale).floor();
        if (sx < 0 || sx >= width) continue;
        final int yIdx = sy * yRowStride + sx;
        final int uvIdx = (sy ~/ 2) * uvRowStride + (sx ~/ 2) * uvPixelStride;
        final int yv = yBytes[yIdx];
        final int uv = uBytes[uvIdx];
        final int vv = vBytes[uvIdx];
        // BT.601 limited range
        double r = yv + 1.402 * (vv - 128);
        double g = yv - 0.344136 * (uv - 128) - 0.714136 * (vv - 128);
        double b = yv + 1.772 * (uv - 128);
        r = (r < 0 ? 0 : (r > 255 ? 255 : r)) / 255.0;
        g = (g < 0 ? 0 : (g > 255 ? 255 : g)) / 255.0;
        b = (b < 0 ? 0 : (b > 255 ? 255 : b)) / 255.0;
        final int dx = padX + ox;
        final int dy = padY + oy;
        final int base = dy * S + dx;
        _outBuffer[0 * S * S + base] = r;
        _outBuffer[1 * S * S + base] = g;
        _outBuffer[2 * S * S + base] = b;
      }
    }
  }
}

/// Reshape a flat Float32List into TFLite's expected NCHW tensor.
/// Returns List<dynamic> because tflite_flutter's reshape returns a dynamic view.
List nchwTensor(Float32List flat, int s) {
  return flat.reshape([1, 3, s, s]);
}

/// Convert output to [Int8List] for TFLite if needed (NNAPI delegate).
/// Currently not used.
extension TensorOps on Float32List {
  List toNchw(int s) => this.reshape([1, 3, s, s]);
}
