// Ghost Radar v2.0+ - Native CameraX + YUV preprocessing bridge.
//
// CameraX (Kotlin) sends raw YUV_420_888 frames to Dart via EventChannel.
// Dart passes Y/U/V planes to native C++ via JNI (YuvProcessor.kt -> yuv_processor.cpp)
// which produces a Float32 NCHW tensor [1, 3, 640, 640] ready for YOLOv8.

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

/// Bridge to CameraX (Kotlin) + native C++ YUV preprocessing.
///
/// Lifecycle:
///   1. createTexture() -> textureId (int)
///   2. startCamera(textureId) -> preview texture ready, frames begin streaming
///   3. onFrame -> YUV plane ByteBuffers
///   4. preprocess() -> Float32List NCHW (zero-copy direct buffer)
class GhostRadarCamera {
  GhostRadarCamera({
    this.inputSize = 640,
  });

  static const MethodChannel _method =
      MethodChannel('ghost_radar/camera');
  static const EventChannel _eventFrames =
      EventChannel('ghost_radar/camera_frames');

  final int inputSize;

  int? _textureId;
  StreamSubscription<dynamic>? _frameSub;
  final StreamController<CameraFrame> _frameCtrl =
      StreamController<CameraFrame>.broadcast();

  /// Stream of YUV frames coming from CameraX.
  Stream<CameraFrame> get frames => _frameCtrl.stream;

  /// Create a Flutter Texture backed by a native SurfaceTexture.
  Future<int> createTexture() async {
    final id = await _method.invokeMethod<int>('createTexture');
    if (id == null) {
      throw StateError('Failed to create texture');
    }
    _textureId = id;
    return id;
  }

  /// Start camera preview + frame streaming.
  Future<void> startCamera({required int textureId}) async {
    _textureId = textureId;
    await _method.invokeMethod<bool>('startCamera', {
      'textureId': textureId,
    });
    _frameSub ??= _eventFrames.receiveBroadcastStream().listen(
      _onFrameEvent,
      onError: (Object e) {
        // ignore: avoid_print
        print('Frame stream error: $e');
      },
    );
  }

  Future<void> stopCamera() async {
    _frameSub?.cancel();
    _frameSub = null;
    await _method.invokeMethod<bool>('stopCamera');
  }

  Future<void> disposeTexture() async {
    final id = _textureId;
    if (id != null) {
      await _method.invokeMethod<bool>('disposeTexture', {'textureId': id});
    }
    _textureId = null;
  }

  Future<void> dispose() async {
    await stopCamera();
    await disposeTexture();
    await _frameCtrl.close();
  }

  /// Get letterbox parameters for un-letterboxing YOLO output boxes.
  Future<({int padX, int padY, double scale})> letterboxParams({
    required int width,
    required int height,
  }) async {
    final res = await _method.invokeMapMethod<String, dynamic>(
      'letterboxParams',
      {'width': width, 'height': height, 'inputSize': inputSize},
    );
    if (res == null) {
      throw StateError('Failed to get letterbox params');
    }
    return (
      padX: res['padX'] as int,
      padY: res['padY'] as int,
      scale: (res['scale'] as num).toDouble(),
    );
  }

  void _onFrameEvent(dynamic event) {
    if (event is! Map) return;
    try {
      final frame = CameraFrame.fromMap(event, inputSize: inputSize);
      if (!_frameCtrl.isClosed) {
        _frameCtrl.add(frame);
      }
    } catch (e) {
      // ignore: avoid_print
      print('Frame parse error: $e');
    }
  }
}

/// Single YUV frame from CameraX.
class CameraFrame {
  CameraFrame({
    required this.width,
    required this.height,
    required this.y,
    required this.u,
    required this.v,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
    required this.rotation,
  });

  final int width;
  final int height;
  final ByteBuffer y;
  final ByteBuffer u;
  final ByteBuffer v;
  final int yRowStride;
  final int uvRowStride;
  final int uvPixelStride;
  final int rotation;

  static CameraFrame fromMap(Map event, {required int inputSize}) {
    return CameraFrame(
      width: event['width'] as int,
      height: event['height'] as int,
      y: event['y'] as ByteBuffer,
      u: event['u'] as ByteBuffer,
      v: event['v'] as ByteBuffer,
      yRowStride: event['yRowStride'] as int,
      uvRowStride: event['uvRowStride'] as int,
      uvPixelStride: event['uvPixelStride'] as int,
      rotation: event['rotation'] as int? ?? 0,
    );
  }
}
