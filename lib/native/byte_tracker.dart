// Ghost Radar v2.0+ - ByteTrack multi-object tracker with Kalman Filter.
//
// ByteTrack (Zhang et al. 2022) is a state-of-the-art tracker that:
//   1. Splits detections into HIGH (conf > 0.5) and LOW (0.1 < conf < 0.5)
//   2. Predicts tracks with Kalman filter
//   3. First associates HIGH detections with predicted tracks (IoU)
//   4. Then associates LOW detections with remaining tracks (IoU)
//   5. New tracks: created from unmatched HIGH detections
//
// Kalman filter state: [cx, cy, w, h, vx, vy, vw, vh] (8-dim)
// Measurement: [cx, cy, w, h] (4-dim)
// Standard constant-velocity model.

import 'dart:math' as math;
import 'dart:ui';

class BoxInternal {
  BoxInternal(this.x1, this.y1, this.x2, this.y2);
  final double x1, y1, x2, y2;
  double get w => x2 - x1;
  double get h => y2 - y1;
  double get cx => (x1 + x2) / 2;
  double get cy => (y1 + y2) / 2;
  Rect toRect() => Rect.fromLTRB(x1, y1, x2, y2);

  static BoxInternal fromRect(Rect r) =>
      BoxInternal(r.left, r.top, r.right, r.bottom);
}

class _KalmanFilter {
  // State: x = [cx, cy, w, h, vx, vy, vw, vh] (8x1)
  // Measurement: z = [cx, cy, w, h] (4x1)
  // Standard CV model with constant velocity

  final int stateDim = 8;
  final int measDim = 4;
  // State transition (8x8)
  late List<List<double>> _F;
  // Measurement matrix (4x8)
  late List<List<double>> _H;
  // Process noise covariance (8x8) - tuned for slow-moving objects
  late List<List<double>> _Q;
  // Measurement noise covariance (4x4) - tuned for YOLO box noise
  late List<List<double>> _R;

  // State estimate
  late List<double> _x; // 8x1
  // Covariance
  late List<List<double>> _P; // 8x8

  _KalmanFilter() {
    _F = List.generate(stateDim, (_) => List.filled(stateDim, 0));
    for (int i = 0; i < 4; i++) _F[i][i] = 1;
    for (int i = 0; i < 4; i++) _F[i][i + 4] = 1; // position += velocity

    _H = List.generate(measDim, (_) => List.filled(stateDim, 0));
    for (int i = 0; i < 4; i++) _H[i][i] = 1;

    _Q = List.generate(stateDim, (_) => List.filled(stateDim, 0));
    for (int i = 0; i < stateDim; i++) {
      _Q[i][i] = (i < 4) ? 1.0 : 10.0; // velocity noise higher
    }

    _R = List.generate(measDim, (_) => List.filled(measDim, 0));
    for (int i = 0; i < measDim; i++) {
      _R[i][i] = 5.0; // measurement noise
    }

    _x = List.filled(stateDim, 0);
    _P = List.generate(stateDim, (_) => List.filled(stateDim, 0));
    for (int i = 0; i < stateDim; i++) {
      _P[i][i] = 100.0; // initial uncertainty
    }
  }

  void init(BoxInternal box) {
    _x[0] = box.cx;
    _x[1] = box.cy;
    _x[2] = box.w;
    _x[3] = box.h;
    _x[4] = 0; // vx
    _x[5] = 0; // vy
    _x[6] = 0; // vw
    _x[7] = 0; // vh
    // Reset covariance
    for (int i = 0; i < stateDim; i++) {
      for (int j = 0; j < stateDim; j++) {
        _P[i][j] = (i == j) ? 100.0 : 0;
      }
    }
  }

  BoxInternal predict() {
    // x = F * x
    final List<double> newX = List.filled(stateDim, 0);
    for (int i = 0; i < stateDim; i++) {
      double s = 0;
      for (int j = 0; j < stateDim; j++) {
        s += _F[i][j] * _x[j];
      }
      newX[i] = s;
    }
    _x = newX;
    // P = F * P * F^T + Q
    _P = _add(_matMul(_matMul(_F, _P), _transpose(_F)), _Q);
    return _boxFromState();
  }

  void update(BoxInternal meas) {
    // y = z - H * x (innovation)
    final List<double> z = [meas.cx, meas.cy, meas.w, meas.h];
    final List<double> hx = List.filled(measDim, 0);
    for (int i = 0; i < measDim; i++) {
      double s = 0;
      for (int j = 0; j < stateDim; j++) {
        s += _H[i][j] * _x[j];
      }
      hx[i] = s;
    }
    final List<double> y = List.generate(measDim, (i) => z[i] - hx[i]);

    // S = H * P * H^T + R
    final List<List<double>> hp = _matMul(_H, _P);
    final List<List<double>> s = _add(_matMul(hp, _transpose(_H)), _R);

    // K = P * H^T * S^-1
    final List<List<double>> sInv = _invert(s);
    final List<List<double>> k = _matMul(_matMul(_P, _transpose(_H)), sInv);

    // x = x + K * y
    for (int i = 0; i < stateDim; i++) {
      double sy = 0;
      for (int j = 0; j < measDim; j++) {
        sy += k[i][j] * y[j];
      }
      _x[i] += sy;
    }

    // P = (I - K * H) * P
    final List<List<double>> kh = _matMul(k, _H);
    final List<List<double>> ikh = _identity(stateDim);
    for (int i = 0; i < stateDim; i++) {
      for (int j = 0; j < stateDim; j++) {
        ikh[i][j] -= kh[i][j];
      }
    }
    _P = _matMul(ikh, _P);
  }

  BoxInternal getState() => _boxFromState();

  BoxInternal _boxFromState() {
    final cx = _x[0];
    final cy = _x[1];
    final w = math.max(2, _x[2]); // avoid degenerate
    final h = math.max(2, _x[3]);
    return BoxInternal(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2);
  }

  // Linear algebra helpers
  static List<List<double>> _add(List<List<double>> a, List<List<double>> b) {
    final r = List.generate(a.length, (_) => List<double>.filled(a[0].length, 0));
    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < a[0].length; j++) {
        r[i][j] = a[i][j] + b[i][j];
      }
    }
    return r;
  }

  static List<List<double>> _matMul(List<List<double>> a, List<List<double>> b) {
    final r = List.generate(a.length, (_) => List<double>.filled(b[0].length, 0));
    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < b[0].length; j++) {
        double s = 0;
        for (int k = 0; k < b.length; k++) {
          s += a[i][k] * b[k][j];
        }
        r[i][j] = s;
      }
    }
    return r;
  }

  static List<List<double>> _transpose(List<List<double>> a) {
    final r = List.generate(a[0].length, (_) => List<double>.filled(a.length, 0));
    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < a[0].length; j++) {
        r[j][i] = a[i][j];
      }
    }
    return r;
  }

  static List<List<double>> _identity(int n) {
    final r = List.generate(n, (_) => List<double>.filled(n, 0));
    for (int i = 0; i < n; i++) r[i][i] = 1;
    return r;
  }

  static List<List<double>> _invert(List<List<double>> a) {
    // Gauss-Jordan for small matrix (4x4)
    final n = a.length;
    final aug = List.generate(n, (i) => List<double>.from(a[i]));
    final inv = _identity(n);
    for (int i = 0; i < n; i++) {
      // Find pivot
      int piv = i;
      for (int j = i + 1; j < n; j++) {
        if (aug[j][i].abs() > aug[piv][i].abs()) piv = j;
      }
      if (piv != i) {
        final tmp = aug[i];
        aug[i] = aug[piv];
        aug[piv] = tmp;
        final tmpI = inv[i];
        inv[i] = inv[piv];
        inv[piv] = tmpI;
      }
      if (aug[i][i].abs() < 1e-9) {
        // Singular - return identity (Kalman will rely on Q)
        return _identity(n);
      }
      final div = aug[i][i];
      for (int j = 0; j < n; j++) {
        aug[i][j] /= div;
        inv[i][j] /= div;
      }
      for (int j = 0; j < n; j++) {
        if (j == i) continue;
        final factor = aug[j][i];
        for (int k = 0; k < n; k++) {
          aug[j][k] -= factor * aug[i][k];
          inv[j][k] -= factor * inv[i][k];
        }
      }
    }
    return inv;
  }
}

/// A single tracked object (Kalman-filtered).
class TrackedObject {
  TrackedObject({
    required this.id,
    required this.className,
    required BoxInternal initialBbox,
    required this.firstSeen,
  })  : kf = _KalmanFilter(),
        age = 0,
        hits = 1,
        hitsStreak = 1,
        timeSinceUpdate = 0,
        lastBbox = initialBbox {
    kf.init(initialBbox);
  }

  final int id;
  String className;
  final _KalmanFilter kf;
  int age; // frames since creation
  int hits; // total matched frames
  int hitsStreak; // consecutive matches
  int timeSinceUpdate; // consecutive unmatched
  BoxInternal lastBbox;
  final DateTime firstSeen;

  /// Predict the next bbox (call before matching).
  BoxInternal predictNext() {
    final pred = kf.predict();
    lastBbox = pred;
    return pred;
  }

  /// Update with matched detection.
  void update(BoxInternal bbox) {
    kf.update(bbox);
    lastBbox = bbox;
    hits++;
    hitsStreak++;
    timeSinceUpdate = 0;
  }

  void markMissed() {
    hitsStreak = 0;
    timeSinceUpdate++;
  }

  /// Confidence = hits (more hits = more stable track).
  double get confidence => math.min(1.0, hits / 5.0);

  /// Confirmed after 3+ hits.
  bool get isConfirmed => hits >= 3;
}

/// Single detection from YOLO (one frame).
class YoloDetectionForTracker {
  YoloDetectionForTracker({
    required this.bbox,
    required this.className,
    required this.confidence,
  });
  final BoxInternal bbox;
  final String className;
  final double confidence;
}

/// ByteTrack v2.0 - tracks per-class, two-stage association.
class ByteTracker {
  ByteTracker({
    this.maxAge = 30,
    this.minHits = 3,
    this.iouThreshold = 0.3,
    this.highThreshold = 0.5,
    this.lowThreshold = 0.1,
  });

  final int maxAge;
  final int minHits;
  final double iouThreshold;
  final double highThreshold;
  final double lowThreshold;

  final Map<String, List<TrackedObject>> _tracksByClass =
      <String, List<TrackedObject>>{};
  int _nextId = 1;

  /// Run one tracking step.
  /// Input: detections from YOLO this frame.
  /// Output: confirmed tracks (sorted by ID).
  List<TrackedObject> update(List<YoloDetectionForTracker> detections) {
    // Group detections by class
    final Map<String, List<YoloDetectionForTracker>> detsByClass = <String, List<YoloDetectionForTracker>>{};
    for (final d in detections) {
      detsByClass.putIfAbsent(d.className, () => <YoloDetectionForTracker>[]).add(d);
    }

    // Process all known classes + new ones
    final allClasses = <String>{
      ..._tracksByClass.keys,
      ...detsByClass.keys,
    };

    for (final cls in allClasses) {
      _processClass(cls, detsByClass[cls] ?? <YoloDetectionForTracker>[]);
    }

    // Collect all confirmed tracks
    final result = <TrackedObject>[];
    for (final tracks in _tracksByClass.values) {
      for (final t in tracks) {
        if (t.isConfirmed) {
          result.add(t);
        }
      }
    }
    result.sort((a, b) => a.id.compareTo(b.id));
    return result;
  }

  void _processClass(String cls, List<YoloDetectionForTracker> dets) {
    final tracks = _tracksByClass.putIfAbsent(cls, () => <TrackedObject>[]);

    // Step 1: predict all tracks
    for (final t in tracks) {
      t.predictNext();
      t.age++;
    }

    if (dets.isEmpty) {
      // All tracks missed
      for (final t in tracks) {
        t.markMissed();
      }
      _removeDead(tracks);
      return;
    }

    // Step 2: split detections by confidence
    final highDets = <YoloDetectionForTracker>[];
    final lowDets = <YoloDetectionForTracker>[];
    for (final d in dets) {
      if (d.confidence >= highThreshold) {
        highDets.add(d);
      } else if (d.confidence >= lowThreshold) {
        lowDets.add(d);
      }
    }

    // Step 3: first association (high confidence)
    final unmatchedTracks1 = <TrackedObject>[];
    final unmatchedDets1 = <YoloDetectionForTracker>[];
    if (highDets.isNotEmpty) {
      _associate(tracks, highDets, unmatchedTracks1, unmatchedDets1);
    } else {
      unmatchedTracks1.addAll(tracks);
    }

    // Step 4: second association (low confidence, only unmatched tracks)
    final unmatchedDets2 = <YoloDetectionForTracker>[];
    _associate(unmatchedTracks1, lowDets, <TrackedObject>[], unmatchedDets2);

    // Step 5: unmatched high-conf detections -> new tracks
    for (final d in unmatchedDets1) {
      final t = TrackedObject(
        id: _nextId++,
        className: cls,
        initialBbox: d.bbox,
        firstSeen: DateTime.now(),
      );
      tracks.add(t);
    }

    // Step 6: mark all unmatched tracks as missed
    for (final t in tracks) {
      if (t.timeSinceUpdate > 0) t.markMissed();
    }
    // Also: any track not in unmatchedTracks1 (already updated) and not in matched set
    final updatedIds = tracks.where((t) => t.timeSinceUpdate == 0).map((t) => t.id).toSet();
    for (final t in tracks) {
      if (!updatedIds.contains(t.id) && t.timeSinceUpdate == 0) {
        // Was updated this frame; skip
        continue;
      }
    }

    _removeDead(tracks);
  }

  void _associate(
    List<TrackedObject> tracks,
    List<YoloDetectionForTracker> dets,
    List<TrackedObject> unmatchedTracks,
    List<YoloDetectionForTracker> unmatchedDets,
  ) {
    if (tracks.isEmpty) {
      unmatchedDets.addAll(dets);
      return;
    }
    if (dets.isEmpty) {
      unmatchedTracks.addAll(tracks);
      return;
    }
    // IoU matrix (track x det)
    final int nt = tracks.length;
    final int nd = dets.length;
    final iouMatrix = List.generate(nt, (_) => List<double>.filled(nd, 0));
    for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nd; j++) {
        iouMatrix[i][j] = _iou(tracks[i].lastBbox, dets[j].bbox);
      }
    }
    // Greedy IoU matching (Hungarian would be better, but greedy is fine for small N)
    final matchedT = <int>{};
    final matchedD = <int>{};
    // Sort by IoU desc
    final pairs = <List<int>>[];
    for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nd; j++) {
        if (iouMatrix[i][j] >= iouThreshold) {
          pairs.add([i, j]);
        }
      }
    }
    pairs.sort((a, b) => iouMatrix[b[0]][b[1]].compareTo(iouMatrix[a[0]][a[1]]));
    for (final p in pairs) {
      if (matchedT.contains(p[0]) || matchedD.contains(p[1])) continue;
      matchedT.add(p[0]);
      matchedD.add(p[1]);
      tracks[p[0]].update(dets[p[1]].bbox);
    }
    for (int i = 0; i < nt; i++) {
      if (!matchedT.contains(i)) unmatchedTracks.add(tracks[i]);
    }
    for (int j = 0; j < nd; j++) {
      if (!matchedD.contains(j)) unmatchedDets.add(dets[j]);
    }
  }

  void _removeDead(List<TrackedObject> tracks) {
    tracks.removeWhere((t) => t.timeSinceUpdate > maxAge);
  }

  static double _iou(BoxInternal a, BoxInternal b) {
    final double ix0 = math.max(a.x1, b.x1);
    final double iy0 = math.max(a.y1, b.y1);
    final double ix1 = math.min(a.x2, b.x2);
    final double iy1 = math.min(a.y2, b.y2);
    final double iw = math.max(0.0, ix1 - ix0);
    final double ih = math.max(0.0, iy1 - iy0);
    final double inter = iw * ih;
    final double union = a.w * a.h + b.w * b.h - inter;
    if (union <= 0) return 0;
    return inter / union;
  }

  /// Get track by ID (for rendering specific tracks).
  TrackedObject? findById(int id) {
    for (final tracks in _tracksByClass.values) {
      for (final t in tracks) {
        if (t.id == id) return t;
      }
    }
    return null;
  }

  /// Clear all tracks (e.g. on camera reset).
  void reset() {
    _tracksByClass.clear();
    _nextId = 1;
  }
}
