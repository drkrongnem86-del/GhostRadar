// Ghost Radar v2.0.2 - Detection Logger
//
// Ghi log detection events ra file JSON Lines format (mỗi event 1 dòng JSON).
// Dùng cho clinical review: sau ca trực, user mở file, share với đồng nghiệp,
// hoặc import vào tool analysis khác.
//
// File location: app docs directory (accessible via file picker / share).
// Format: ndjson (newline-delimited JSON) - 1 event per line
//   {"ts":"2026-08-26T15:00:00.000","type":"detect","class":"person","conf":0.87,"bbox":[x1,y1,x2,y2],"track_id":3,"heading":45.2,"lowfreq_hz":2.3}
//   {"ts":"...","type":"alarm","band":1,"hz":2.3,"ratio":3.5,"heading":45.2}
//   {"ts":"...","type":"combo","class":"person","conf":0.87,"track_id":3,"heading":45.2}
//
// Rotation: file theo ngày (ghost_radar_2026-08-26.log), auto rollover.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

class DetectionLogger {
  static const String _filePrefix = 'ghost_radar_';
  static const String _fileExt = '.log';

  /// Singleton - 1 logger per app
  static final DetectionLogger instance = DetectionLogger._();
  DetectionLogger._();

  File? _currentFile;
  String? _currentDate;
  IOSink? _sink;
  bool _enabled = true;
  int _eventCount = 0;

  bool get isEnabled => _enabled;
  int get eventCount => _eventCount;

  Future<void> setEnabled(bool value) async {
    _enabled = value;
    if (!value) {
      await _closeSink();
    }
  }

  Future<String> _getLogDir() async {
    final dir = await getApplicationDocumentsDirectory();
    final logDir = Directory('${dir.path}/logs');
    if (!await logDir.exists()) {
      await logDir.create(recursive: true);
    }
    return logDir.path;
  }

  String _todayStamp() {
    final now = DateTime.now();
    return '${now.year.toString().padLeft(4, '0')}-'
        '${now.month.toString().padLeft(2, '0')}-'
        '${now.day.toString().padLeft(2, '0')}';
  }

  Future<File> _ensureFile() async {
    final today = _todayStamp();
    if (_currentFile == null || _currentDate != today) {
      await _closeSink();
      final dir = await _getLogDir();
      final path = '$dir/$_filePrefix$today$_fileExt';
      _currentFile = File(path);
      _currentDate = today;
      _sink = _currentFile!.openWrite(mode: FileMode.append);
    }
    return _currentFile!;
  }

  Future<void> _closeSink() async {
    try {
      await _sink?.flush();
      await _sink?.close();
    } catch (_) {}
    _sink = null;
  }

  Future<void> _writeLine(Map<String, dynamic> event) async {
    if (!_enabled) return;
    try {
      final file = await _ensureFile();
      _sink ??= file.openWrite(mode: FileMode.append);
      _sink!.writeln(jsonEncode(event));
      // Flush every 5 events to avoid data loss
      _eventCount++;
      if (_eventCount % 5 == 0) {
        await _sink!.flush();
      }
    } catch (_) {
      // Silently ignore - logging is best-effort
    }
  }

  /// Log a detection event
  Future<void> logDetection({
    required String className,
    required double confidence,
    required List<double> bbox, // [x1, y1, x2, y2] in image coords
    required int? trackId,
    required double headingDeg,
    double? lowfreqHz,
  }) async {
    await _writeLine({
      'ts': DateTime.now().toIso8601String(),
      'type': 'detect',
      'class': className,
      'conf': double.parse(confidence.toStringAsFixed(3)),
      'bbox': bbox.map((v) => double.parse(v.toStringAsFixed(1))).toList(),
      'track_id': trackId,
      'heading': double.parse(headingDeg.toStringAsFixed(1)),
      if (lowfreqHz != null)
        'lowfreq_hz': double.parse(lowfreqHz.toStringAsFixed(2)),
    });
  }

  /// Log audio alarm event
  Future<void> logAudioAlarm({
    required int bandIndex,
    required double hz,
    required double ratio,
    required double headingDeg,
  }) async {
    await _writeLine({
      'ts': DateTime.now().toIso8601String(),
      'type': 'alarm',
      'band': bandIndex,
      'hz': double.parse(hz.toStringAsFixed(2)),
      'ratio': double.parse(ratio.toStringAsFixed(2)),
      'heading': double.parse(headingDeg.toStringAsFixed(1)),
    });
  }

  /// Log ghost combo event (audio alarm + visual detection)
  Future<void> logGhostCombo({
    required String className,
    required double confidence,
    required int trackId,
    required double headingDeg,
  }) async {
    await _writeLine({
      'ts': DateTime.now().toIso8601String(),
      'type': 'combo',
      'class': className,
      'conf': double.parse(confidence.toStringAsFixed(3)),
      'track_id': trackId,
      'heading': double.parse(headingDeg.toStringAsFixed(1)),
    });
  }

  /// Get all log file paths
  Future<List<String>> listLogFiles() async {
    final dir = Directory(await _getLogDir());
    final files = await dir
        .list()
        .where((e) => e is File && e.path.endsWith(_fileExt))
        .map((e) => e.path)
        .toList();
    files.sort();
    return files.reversed.toList();
  }

  /// Get current log file path
  Future<String?> getCurrentLogPath() async {
    await _ensureFile();
    await _sink?.flush();
    return _currentFile?.path;
  }

  /// Read tail of current log (last N lines)
  Future<List<String>> tailCurrentLog(int n) async {
    final path = await getCurrentLogPath();
    if (path == null) return [];
    final file = File(path);
    if (!await file.exists()) return [];
    final lines = await file.readAsLines();
    return lines.length <= n
        ? lines
        : lines.sublist(lines.length - n);
  }

  /// Clear all logs
  Future<void> clearAll() async {
    await _closeSink();
    final dir = Directory(await _getLogDir());
    if (await dir.exists()) {
      await for (final entity in dir.list()) {
        try {
          if (entity is File) await entity.delete();
        } catch (_) {}
      }
    }
    _currentFile = null;
    _currentDate = null;
    _eventCount = 0;
  }

  /// Close sink on app exit
  Future<void> dispose() async {
    await _closeSink();
  }
}
