// Ghost Radar v2.1.0 - EVP Auto-Recorder
//
// Maintains a 10-second rolling buffer of low-frequency audio samples.
// On demand (e.g., ghost combo trigger), dumps the buffer to a WAV file
// with a filename embedding timestamp + heading + class name.
//
// Inspired by EVP recorders (GhostTube EVP, EVP-2) where investigators
// capture 5-15s clips when "anomalies" are detected.
//
// API:
//   - EvpRecorder(lookbackSec: 10, sampleRate: 100)
//   - pushSample(value)        // call per 100Hz sample (after anti-alias/decimate)
//   - saveCurrentSnapshot(...) returns path to WAV file
//   - dispose()                 // cleanup

import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';

class EvpRecorder {
  EvpRecorder({this.lookbackSec = 10, this.sampleRate = 100})
      : _ring = Float64List(lookbackSec * sampleRate);

  /// Seconds of audio to keep in the rolling buffer.
  final int lookbackSec;

  /// Sample rate of the input stream (Hz). Our pipeline produces 100Hz after
  /// 441x decimation from 44.1kHz.
  final int sampleRate;

  /// Rolling buffer of float64 samples.
  final Float64List _ring;
  int _writeIdx = 0;
  int _filled = 0;

  /// Push a new sample into the rolling buffer.
  void pushSample(double value) {
    _ring[_writeIdx] = value;
    _writeIdx = (_writeIdx + 1) % _ring.length;
    if (_filled < _ring.length) _filled++;
  }

  int get bufferedSamples => _filled;
  int get bufferSize => _ring.length;
  bool get hasFullBuffer => _filled >= _ring.length;

  /// Linearize the ring into a contiguous list (oldest to newest).
  Float64List _linearize() {
    final out = Float64List(_filled);
    final int start = _filled < _ring.length ? 0 : _writeIdx;
    for (int i = 0; i < _filled; i++) {
      out[i] = _ring[(start + i) % _ring.length];
    }
    return out;
  }

  /// Save the current buffer as a 16-bit mono WAV file. Returns the file path.
  Future<String> saveCurrentSnapshot({
    double? heading,
    String? className,
    double? confidence,
    int? trackId,
  }) async {
    final samples = _linearize();
    final wavBytes = _encodeWav(samples, sampleRate);
    final dir = await _getEvpDir();
    final now = DateTime.now();
    String two(int v) => v.toString().padLeft(2, '0');
    String three(int v) => v.toString().padLeft(3, '0');
    final ts = '${now.year}'
        '${two(now.month)}'
        '${two(now.day)}_'
        '${two(now.hour)}'
        '${two(now.minute)}'
        '${two(now.second)}_'
        '${three(now.millisecond)}';
    final tag = className != null ? '_${_sanitize(className)}' : '';
    final headingTag =
        heading != null ? '_${heading.toStringAsFixed(0)}deg' : '';
    final filename = 'evp_${ts}${tag}${headingTag}.wav';
    final file = File('${dir.path}/$filename');
    await file.writeAsBytes(wavBytes, flush: true);
    return file.path;
  }

  static String _sanitize(String s) =>
      s.replaceAll(RegExp(r'[^a-zA-Z0-9_-]'), '_');

  static Future<Directory> _getEvpDir() async {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/evp');
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  /// Encode Float64 PCM as 16-bit mono WAV. Returns the byte buffer.
  Uint8List _encodeWav(Float64List samples, int sampleRate) {
    final int byteCount = samples.length * 2;
    final int dataSize = byteCount;
    final int fileSize = 36 + dataSize;
    final ByteData header = ByteData(44);

    // RIFF chunk
    header.setUint8(0, 0x52); // R
    header.setUint8(1, 0x49); // I
    header.setUint8(2, 0x46); // F
    header.setUint8(3, 0x46); // F
    header.setUint32(4, fileSize, Endian.little);
    header.setUint8(8, 0x57); // W
    header.setUint8(9, 0x41); // A
    header.setUint8(10, 0x56); // V
    header.setUint8(11, 0x45); // E

    // fmt sub-chunk
    header.setUint8(12, 0x66); // f
    header.setUint8(13, 0x6D); // m
    header.setUint8(14, 0x74); // t
    header.setUint8(15, 0x20); // (space)
    header.setUint32(16, 16, Endian.little); // Subchunk1Size for PCM
    header.setUint16(20, 1, Endian.little); // AudioFormat = PCM
    header.setUint16(22, 1, Endian.little); // NumChannels = mono
    header.setUint32(24, sampleRate, Endian.little);
    header.setUint32(28, sampleRate * 2, Endian.little); // ByteRate
    header.setUint16(32, 2, Endian.little); // BlockAlign
    header.setUint16(34, 16, Endian.little); // BitsPerSample

    // data sub-chunk
    header.setUint8(36, 0x64); // d
    header.setUint8(37, 0x61); // a
    header.setUint8(38, 0x74); // t
    header.setUint8(39, 0x61); // a
    header.setUint32(40, dataSize, Endian.little);

    final Uint8List output = Uint8List(44 + byteCount);
    output.setRange(0, 44, header.buffer.asUint8List());
    final ByteData sampleBytes = ByteData(byteCount);
    for (int i = 0; i < samples.length; i++) {
      double v = samples[i];
      if (v > 1.0) v = 1.0;
      if (v < -1.0) v = -1.0;
      final int s = (v * 32767).toInt();
      sampleBytes.setInt16(i * 2, s, Endian.little);
    }
    output.setRange(44, 44 + byteCount, sampleBytes.buffer.asUint8List());
    return output;
  }

  void dispose() {
    // Nothing to free - ring buffer is GC'd
  }
}
