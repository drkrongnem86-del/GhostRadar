// Ghost Radar v2.1.0 - Waterfall Spectrogram
//
// Real-time scrolling spectrogram (Sonic Visualiser / Praat style).
// Rolling buffer of FFT magnitudes, rendered as heatmap with viridis colormap.
//
// Data flow:
//   1. main.dart calls `pushFrame(magnitudes)` every FFT cycle (~2.56s)
//   2. Each frame: N magnitudes (0.39Hz/bin for 256-point FFT @ 100Hz)
//   3. Rolling buffer keeps last `maxFrames` frames (~77s for 30 frames)
//   4. CustomPainter renders as time × frequency heatmap
//
// Visual: newest frame on right, scrolls left. 0Hz at bottom, 50Hz at top.
// Red horizontal marker at 20Hz (the app's upper limit of interest).

import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';

class WaterfallSpectrogram extends StatefulWidget {
  const WaterfallSpectrogram({
    super.key,
    required this.numBins,
    this.maxFrames = 30,
    this.height = 120,
  });

  /// Number of frequency bins per frame (typically 256 from FFT).
  final int numBins;

  /// Number of frames in the rolling history (30 frames ≈ 77s @ 2.56s/frame).
  final int maxFrames;

  /// Widget height in logical pixels.
  final double height;

  @override
  State<WaterfallSpectrogram> createState() => _WaterfallSpectrogramState();
}

class _WaterfallSpectrogramState extends State<WaterfallSpectrogram> {
  /// Rolling buffer: oldest at index 0, newest at end.
  final List<Float32List> _frames = <Float32List>[];

  /// Push a new FFT magnitude frame. Normalizes via log scale to 0..1.
  void pushFrame(List<double> magnitudes) {
    if (magnitudes.length != widget.numBins) return;
    final Float32List normalized = Float32List(widget.numBins);
    for (int i = 0; i < widget.numBins; i++) {
      final double m = magnitudes[i].abs();
      // Empirical: most useful dynamic range for our 0-20Hz mic signal
      // log10(m) ranges from -8 (silence) to -2 (loud); map to 0..1
      final double v = m < 1e-10 ? 0 : (math.log(m) / math.ln10 + 8) / 6;
      normalized[i] = v.clamp(0.0, 1.0);
    }
    setState(() {
      _frames.add(normalized);
      if (_frames.length > widget.maxFrames) {
        _frames.removeAt(0);
      }
    });
  }

  /// Clear history (e.g., on scan stop).
  void clear() {
    setState(() => _frames.clear());
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: widget.height,
      decoration: BoxDecoration(
        color: const Color(0xFF000000),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xFF252525)),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(6),
        child: CustomPaint(
          painter: _SpectrogramPainter(
            frames: _frames,
            numBins: widget.numBins,
            maxFrames: widget.maxFrames,
          ),
          size: Size.infinite,
        ),
      ),
    );
  }
}

class _SpectrogramPainter extends CustomPainter {
  _SpectrogramPainter({
    required this.frames,
    required this.numBins,
    required this.maxFrames,
  });

  final List<Float32List> frames;
  final int numBins;
  final int maxFrames;

  /// Hz per bin (100Hz sample rate, 256-point FFT = 0.39Hz/bin).
  static const double _binHz = 100.0 / 256.0;

  @override
  void paint(Canvas canvas, Size size) {
    if (frames.isEmpty) {
      _drawPlaceholder(canvas, size);
      return;
    }

    final int colCount = frames.length;
    final double cellW = size.width / maxFrames;
    final double cellH = size.height / numBins;

    // Draw cells: column = frame (oldest left, newest right)
    // Frames list is oldest-to-newest, so draw offset = maxFrames - colCount
    final int startCol = maxFrames - colCount;
    for (int col = 0; col < colCount; col++) {
      final Float32List frame = frames[col];
      final int x = (startCol + col);
      for (int bin = 0; bin < numBins; bin++) {
        final double intensity = bin < frame.length ? frame[bin] : 0;
        if (intensity < 0.02) continue; // skip near-zero for speed
        final Color color = _viridis(intensity);
        // Bin 0 = 0Hz at bottom, bin N = N*binHz at top
        final double y = size.height - (bin + 1) * cellH;
        final rect = Rect.fromLTWH(
          x * cellW,
          y,
          cellW + 0.6, // overlap to hide antialiasing seams
          cellH + 0.6,
        );
        canvas.drawRect(rect, Paint()..color = color);
      }
    }

    // 20Hz marker line (red, semi-transparent)
    final int binFor20Hz = (20.0 / _binHz).round();
    final double y20 = size.height - binFor20Hz * cellH;
    final markerPaint = Paint()
      ..color = const Color(0xFFFF3B3B).withOpacity(0.55)
      ..strokeWidth = 1;
    canvas.drawLine(Offset(0, y20), Offset(size.width, y20), markerPaint);

    // Y-axis frequency labels
    final textStyle = TextStyle(
      color: const Color(0xFF808080).withOpacity(0.85),
      fontSize: 9,
      fontFamily: 'monospace',
    );
    _drawHzLabel(canvas, '0', 0, size, textStyle);
    _drawHzLabel(canvas, '10', (10.0 / _binHz).round(), size, textStyle);
    _drawHzLabel(canvas, '20', binFor20Hz, size, textStyle);
    _drawHzLabel(canvas, '50', (50.0 / _binHz).round(), size, textStyle);
  }

  void _drawHzLabel(Canvas canvas, String label, int bin, Size size, TextStyle style) {
    final tp = TextPainter(
      text: TextSpan(text: '$label Hz', style: style),
      textDirection: TextDirection.ltr,
    )..layout();
    final double y = size.height - (bin / numBins) * size.height - tp.height / 2;
    tp.paint(canvas, Offset(2, y.clamp(0, size.height - tp.height)));
  }

  void _drawPlaceholder(Canvas canvas, Size size) {
    final tp = TextPainter(
      text: const TextSpan(
        text: 'Đang chờ FFT frame… (bấm BẮT ĐẦU)',
        style: TextStyle(color: Color(0xFF606060), fontSize: 11),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(
      canvas,
      Offset((size.width - tp.width) / 2, (size.height - tp.height) / 2),
    );
  }

  /// Viridis-inspired colormap (matplotlib's viridis approximation).
  /// Goes: dark purple → blue → teal → green → yellow.
  Color _viridis(double t) {
    t = t.clamp(0.0, 1.0);
    if (t < 0.25) {
      return Color.lerp(
        const Color(0xFF440154), const Color(0xFF3B528B), t / 0.25)!;
    } else if (t < 0.5) {
      return Color.lerp(
        const Color(0xFF3B528B), const Color(0xFF21908C), (t - 0.25) / 0.25)!;
    } else if (t < 0.75) {
      return Color.lerp(
        const Color(0xFF21908C), const Color(0xFF5DC863), (t - 0.5) / 0.25)!;
    } else {
      return Color.lerp(
        const Color(0xFF5DC863), const Color(0xFFFDE725), (t - 0.75) / 0.25)!;
    }
  }

  @override
  bool shouldRepaint(_SpectrogramPainter old) {
    return old.frames.length != frames.length || old.frames != frames;
  }
}
