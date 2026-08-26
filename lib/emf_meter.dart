// Ghost Radar v2.1.0 - EMF Meter
//
// Magnetometer-based electromagnetic field meter (GhostTube / SB7 style).
// Computes field magnitude from raw mx/my/mz, displays as vertical bar.
//
// Earth's field: ~25-65 µT (varies by location).
// Deviations of >10 µT from baseline = "anomaly" (in the spirit of EVP apps).
// Note: real phone magnetometer is uncalibrated, so absolute values are
// approximate. Use the BASELINE offset to track relative changes.
//
// API:
//   - EmfMeter.fromMagnetometer(mx, my, mz)  // µT (assuming sensors_plus uT)
//   - push(mx, my, mz)  // add a new sample
//   - magnitude  // current total field in µT
//   - delta      // deviation from baseline in µT

import 'dart:math' as math;
import 'package:flutter/material.dart';

class EmfReading {
  const EmfReading({
    required this.magnitude,
    required this.delta,
    required this.isAnomaly,
  });
  final double magnitude; // µT
  final double delta; // µT from baseline
  final bool isAnomaly; // >10 µT from baseline
}

class EmfMeter extends StatefulWidget {
  const EmfMeter({
    super.key,
    this.height = 100,
    this.width = 60,
    this.anomalyThreshold = 10.0,
  });

  final double height;
  final double width;
  final double anomalyThreshold;

  @override
  State<EmfMeter> createState() => _EmfMeterState();
}

class _EmfMeterState extends State<EmfMeter> {
  /// Current magnitude (µT).
  double _magnitude = 0;

  /// Running baseline (EMA of magnitude).
  double _baseline = 50.0; // Earth field typical default
  static const double _emaAlpha = 0.995; // very slow baseline adaptation

  /// History for sparkline (last 60 samples).
  final List<double> _history = <double>[];
  static const int _maxHistory = 60;

  void push(double mx, double my, double mz) {
    final double mag = math.sqrt(mx * mx + my * my + mz * mz);
    setState(() {
      _magnitude = mag;
      _baseline = _emaAlpha * _baseline + (1 - _emaAlpha) * mag;
      _history.add(mag);
      if (_history.length > _maxHistory) {
        _history.removeAt(0);
      }
    });
  }

  EmfReading get reading {
    final double delta = _magnitude - _baseline;
    return EmfReading(
      magnitude: _magnitude,
      delta: delta,
      isAnomaly: delta.abs() > widget.anomalyThreshold,
    );
  }

  void reset() {
    setState(() {
      _magnitude = 0;
      _baseline = 50.0;
      _history.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    final r = reading;
    return Container(
      width: widget.width,
      height: widget.height,
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: const Color(0xFF0E0E0E),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: r.isAnomaly
              ? const Color(0xFFFF3B3B)
              : const Color(0xFF252525),
          width: r.isAnomaly ? 1.5 : 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'EMF',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Color(0xFF808080),
              fontSize: 9,
              fontWeight: FontWeight.bold,
            ),
          ),
          Expanded(
            child: CustomPaint(
              painter: _EmfMeterPainter(
                magnitude: r.magnitude,
                baseline: _baseline,
                history: _history,
                anomaly: r.isAnomaly,
                delta: r.delta,
              ),
              size: Size.infinite,
            ),
          ),
          Text(
            '${r.magnitude.toStringAsFixed(0)}µT',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: r.isAnomaly
                  ? const Color(0xFFFF3B3B)
                  : const Color(0xFFB0B0B0),
              fontSize: 9,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
    );
  }
}

class _EmfMeterPainter extends CustomPainter {
  _EmfMeterPainter({
    required this.magnitude,
    required this.baseline,
    required this.history,
    required this.anomaly,
    required this.delta,
  });

  final double magnitude;
  final double baseline;
  final List<double> history;
  final bool anomaly;
  final double delta;

  @override
  void paint(Canvas canvas, Size size) {
    // Vertical bar meter
    final double maxMag = math.max(magnitude, baseline) * 1.2;
    final double fillFrac = maxMag > 0 ? (magnitude / maxMag).clamp(0.0, 1.0) : 0;
    final double baselineFrac = maxMag > 0 ? (baseline / maxMag).clamp(0.0, 1.0) : 0;

    // Background
    final bg = Paint()..color = const Color(0xFF1A1A1A);
    canvas.drawRRect(
      RRect.fromRectAndRadius(Offset.zero & size, const Radius.circular(3)),
      bg,
    );

    // Filled portion
    final fillColor = anomaly
        ? const Color(0xFFFF3B3B)
        : (magnitude > baseline
            ? const Color(0xFFFFB300)
            : const Color(0xFF1FE033));
    if (fillFrac > 0) {
      final fillRect = Rect.fromLTWH(
        0,
        size.height * (1 - fillFrac),
        size.width,
        size.height * fillFrac,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(fillRect, const Radius.circular(3)),
        Paint()..color = fillColor,
      );
    }

    // Baseline line (horizontal)
    final baselineY = size.height * (1 - baselineFrac);
    canvas.drawLine(
      Offset(0, baselineY),
      Offset(size.width, baselineY),
      Paint()
        ..color = const Color(0xFF808080)
        ..strokeWidth = 1,
    );

    // Sparkline overlay (last 60 samples) at bottom 25%
    if (history.length > 1) {
      final sparkH = size.height * 0.25;
      final sparkY0 = size.height - sparkH;
      final path = Path();
      for (int i = 0; i < history.length; i++) {
        final double x = i * size.width / (history.length - 1);
        final double v = history[i] / maxMag;
        final double y = sparkY0 + (1 - v) * sparkH;
        if (i == 0) {
          path.moveTo(x, y);
        } else {
          path.lineTo(x, y);
        }
      }
      canvas.drawPath(
        path,
        Paint()
          ..color = const Color(0xFF00E5FF).withOpacity(0.7)
          ..strokeWidth = 1
          ..style = PaintingStyle.stroke,
      );
    }
  }

  @override
  bool shouldRepaint(_EmfMeterPainter old) {
    return old.magnitude != magnitude ||
        old.baseline != baseline ||
        old.anomaly != anomaly ||
        old.history.length != history.length;
  }
}
