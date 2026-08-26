// Ghost Radar v2.0.1 - Re-ID (Re-identification) Engine
//
// Re-ID solves the "lost track" problem: when a person walks out of frame
// and walks back in (e.g. through a doorway), ByteTrack would give them a
// new ID. With Re-ID, we use appearance features (MobileNetV2 embedding) to
// re-assign the original ID.
//
// Pipeline:
//   1. YOLOv8 detects person -> bbox
//   2. Crop person region from frame
//   3. Re-ID: MobileNetV2 (ImageNet pretrained) -> 1280-dim feature
//   4. When track is lost (after maxAge), save feature to gallery
//   5. New track appears -> compare with gallery via cosine similarity
//   6. If sim > threshold (0.65), re-assign old ID with "(re-id)" badge
//
// Note: We use ImageNet-pretrained MobileNetV2 (not trained on Market-1501
// for Re-ID) as a "good enough" feature extractor. For production-grade Re-ID,
// swap in a model trained on person re-id datasets.

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart';

import 'byte_tracker.dart';
import 'camera_x.dart';

/// Gallery entry: lost track with its last feature
class _GalleryEntry {
  _GalleryEntry({
    required this.id,
    required this.className,
    required this.feature,
    required this.lostAt,
  });
  final int id;
  final String className;
  final Float32List feature; // L2-normalized
  final DateTime lostAt;
  bool get isExpired =>
      DateTime.now().difference(lostAt).inSeconds > 30; // 30s window
}

/// Re-ID Engine: maintains gallery of lost tracks + matches new tracks
class ReIdEngine {
  ReIdEngine({
    this.matchThreshold = 0.65, // cosine sim
    this.galleryTtlSec = 30,
  });

  final double matchThreshold;
  final int galleryTtlSec;

  Interpreter? _interpreter;
  late List<int> _inShape;
  late List<int> _outShape;

  final List<_GalleryEntry> _gallery = <_GalleryEntry>[];

  bool get isReady => _interpreter != null;
  int get gallerySize => _gallery.length;

  Future<void> load() async {
    try {
      final opts = InterpreterOptions()..threads = 2;
      _interpreter = await Interpreter.fromAsset(
        'reid_mobilenet.tflite',
        options: opts,
      );
      _inShape = _interpreter!.getInputTensor(0).shape;
      _outShape = _interpreter!.getOutputTensor(0).shape;
      // ignore: avoid_print
      print('ReID loaded in=$_inShape out=$_outShape');
    } catch (e) {
      // ignore: avoid_print
      print('ReID load fail: $e');
      rethrow;
    }
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  int get inputH => _inShape[1];
  int get inputW => _inShape[2];
  int get featDim => _outShape.reduce((a, b) => a * b);

  /// Extract L2-normalized feature from RGB bytes
  /// input: RGB bytes (inputW*inputH*3), normalized to [0, 1]
  Float32List runFeature(Float32List input) {
    final interp = _interpreter!;
    final reshaped = input.reshape([1, inputH, inputW, 3]);
    final outSize = featDim;
    final outBuf = Float32List(outSize);
    final outReshaped = outBuf.reshape(_outShape);
    interp.run(reshaped, outReshaped);
    return _l2Normalize(outBuf);
  }

  Float32List _l2Normalize(Float32List v) {
    double sum = 0;
    for (final f in v) {
      sum += f * f;
    }
    final norm = math.sqrt(sum);
    if (norm == 0) return v;
    final out = Float32List(v.length);
    for (int i = 0; i < v.length; i++) {
      out[i] = v[i] / norm;
    }
    return out;
  }

  /// Add a confirmed track to the gallery when it's lost
  void addToGallery(TrackedObject track, Float32List? feature) {
    if (feature == null) return;
    if (track.hits < 3) return; // only confirmed tracks
    _gallery.add(_GalleryEntry(
      id: track.id,
      className: track.className,
      feature: feature,
      lostAt: DateTime.now(),
    ));
    // Limit gallery size
    if (_gallery.length > 30) {
      _gallery.removeAt(0);
    }
  }

  /// Find best gallery match for a new feature
  /// Returns gallery entry ID if matched, else null
  int? findMatch(Float32List newFeature, String className) {
    _purgeExpired();
    int? bestId;
    double bestSim = matchThreshold;
    for (final entry in _gallery) {
      if (entry.className != className) continue;
      final double sim = _cosineSim(newFeature, entry.feature);
      if (sim > bestSim) {
        bestSim = sim;
        bestId = entry.id;
      }
    }
    if (bestId != null) {
      // Remove from gallery
      _gallery.removeWhere((e) => e.id == bestId);
    }
    return bestId;
  }

  void _purgeExpired() {
    _gallery.removeWhere((e) => e.isExpired);
  }

  static double _cosineSim(Float32List a, Float32List b) {
    if (a.length != b.length) return 0;
    double dot = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  void reset() {
    _gallery.clear();
  }
}

/// Extract a person crop from YUV frame and run Re-ID feature extraction
class PersonFeatureExtractor {
  PersonFeatureExtractor(this._reid);

  final ReIdEngine _reid;

  /// Extract feature for a person bbox in original image coords
  /// Returns null if bbox is invalid
  Float32List? extract(
    CameraFrame frame,
    double x1,
    double y1,
    double x2,
    double y2,
  ) {
    final int inW = frame.width;
    final int inH = frame.height;
    // Clip bbox to image bounds
    final int cx1 = x1.clamp(0, inW - 1).toInt();
    final int cy1 = y1.clamp(0, inH - 1).toInt();
    final int cx2 = x2.clamp(0, inW - 1).toInt();
    final int cy2 = y2.clamp(0, inH - 1).toInt();
    if (cx2 - cx1 < 8 || cy2 - cy1 < 8) return null;

    final int cropW = cx2 - cx1;
    final int cropH = cy2 - cy1;

    final int outH = _reid.inputH;
    final int outW = _reid.inputW;

    final Uint8List yBytes = frame.y.asUint8List();
    final Uint8List uBytes = frame.u.asUint8List();
    final Uint8List vBytes = frame.v.asUint8List();
    final int yRowStride = frame.yRowStride;
    final int uvRowStride = frame.uvRowStride;
    final int uvPixelStride = frame.uvPixelStride;

    final Float32List input = Float32List(outH * outW * 3);
    final double scaleX = cropW / outW;
    final double scaleY = cropH / outH;
    const double inv255 = 1.0 / 255.0;

    for (int oy = 0; oy < outH; oy++) {
      final int sy = cy1 + (oy * scaleY).floor();
      if (sy >= inH) break;
      for (int ox = 0; ox < outW; ox++) {
        final int sx = cx1 + (ox * scaleX).floor();
        if (sx >= inW) break;
        final int yIdx = sy * yRowStride + sx;
        final int uvIdx = (sy >> 1) * uvRowStride + (sx >> 1) * uvPixelStride;
        final int yv = yBytes[yIdx];
        final int uv = uBytes[uvIdx];
        final int vv = vBytes[uvIdx];
        double r = yv + 1.402 * (vv - 128);
        double g = yv - 0.344136 * (uv - 128) - 0.714136 * (vv - 128);
        double b = yv + 1.772 * (uv - 128);
        r = (r < 0 ? 0 : (r > 255 ? 255 : r)) * inv255;
        g = (g < 0 ? 0 : (g > 255 ? 255 : g)) * inv255;
        b = (b < 0 ? 0 : (b > 255 ? 255 : b)) * inv255;
        final int base = (oy * outW + ox) * 3;
        input[base + 0] = r;
        input[base + 1] = g;
        input[base + 2] = b;
      }
    }
    return _reid.runFeature(input);
  }
}
